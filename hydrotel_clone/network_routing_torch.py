"""Routeur réseau Hydrotel VECTORISÉ et DIFFÉRENTIABLE (PyTorch). Même physique
que network_routing.py (onde cinématique modifiée + lacs + topologie) mais
vectorisé par NIVEAU topologique (tronçons d'un même niveau = indépendants →
calcul tensoriel ; balayage séquentiel des niveaux avec scatter-add aval).

Newton déroulé sur MAXITER (pas de break, différentiable). Pour méandre : prend
des tenseurs par tronçon (depuis le graphe + territorial), gradient end-to-end.

Validé contre network_routing.py (numpy) sur DELISLE.
"""
from __future__ import annotations
import torch
from torch import Tensor

MAXITER = 20
TINY = 1.0e-20


def _celerite_vec(lng, lrg, pte, man, qamont, qaval):
    qmoy = (qamont + qaval) / 2.0
    alpha = (man * lrg ** 0.67) ** 0.6
    beta = 0.6
    r = alpha * pte ** (-beta / 2.0) / lrg
    sf = torch.clamp(pte - r * (qaval ** beta - qamont ** beta) / lng, min=0.00125)
    section = alpha * qmoy ** beta * sf ** (-beta / 2.0)
    return torch.where(section > 0, 1.67 * qmoy / torch.clamp(section, min=TINY), torch.zeros_like(section))


def _transfert_riviere_vec(dt, lng, lrg, pte, man, qa, ql, qb, qc, qm):
    beta = 0.6; s = 0.6
    alpha = (man * lrg ** (2.0 / 3.0)) ** 0.6
    r = (alpha / lrg * pte) ** (-0.3)
    c1 = 2.0 * alpha * lng / dt
    c2 = r / lng
    c3 = qb ** s
    c4 = qb ** beta
    c5 = qc - qb + qa + ql + qm
    qd = pte ** (beta / 2.0) / c1 * (2.0 * (qa - qb) + ql + qm) + c4
    qd = torch.where(qd <= 0, (qa + qb) / 2.0 + (ql + qm) / 2.0, qd)
    qd = torch.where(qd <= 0, (qa + qc) / 2.0 + (ql + qm) / 2.0, qd)
    qd = torch.where(qd <= 0, qc + qm, qd)
    qd = torch.clamp(qd, min=TINY) ** (1.0 / beta)
    c2e = c2.clone() if torch.is_tensor(c2) else torch.full_like(qd, c2)
    for _ in range(MAXITER):
        v1 = qd ** s
        v2 = qd ** beta
        v3 = pte - c2e * (v1 - c3)
        mask = v3 < pte
        v3 = torch.where(mask, pte if torch.is_tensor(pte) else torch.full_like(v3, pte), v3)
        c2e = torch.where(mask, torch.zeros_like(c2e), c2e)
        v4 = v3 ** (-beta / 2.0)
        f0 = qd + c1 * v4 * (v2 - c4) - c5
        f1 = beta / 2.0 * v4 / v3 * c2e * s * v1 / qd * (v2 - c4) + v4 * beta * v2 / qd
        f1 = 1.0 + c1 * f1
        qd = torch.clamp(qd - f0 / f1, min=TINY)
    return qd


def _transfert_lac_vec(dt, aire, c, k, qa, ql, qb, qc, qm):
    hb = torch.where(qb > 0, (qb / c) ** (1.0 / k), torch.zeros_like(qb))
    haut = torch.clamp(hb + ((qa + qc) / 2.0 - qb + (ql + qm) / 2.0) * dt / aire, min=0.0)
    for _ in range(MAXITER):
        f0 = haut - hb + (qb + c * haut ** k - qa - qc - ql - qm) * dt / aire / 2.0
        f1 = 1.0 + c * k * torch.clamp(haut, min=TINY) ** (k - 1.0) * dt / aire / 2.0
        haut = haut - f0 / f1
    return c * torch.clamp(haut, min=0.0) ** k


def route_network_torch(P, downstream, level_groups, apport_lateral, pdts=86400):
    """P : dict de tenseurs (nR,) : is_river (bool), lng, lrg, pte, man, surface_m2,
    c, k. downstream : long (nR,) (-1=exutoire). level_groups : liste de long
    tensors (indices par niveau topo amont→aval). apport_lateral : (T, nR).
    Retourne debit_aval (T, nR)."""
    dev = apport_lateral.device; dt_type = apport_lateral.dtype
    nR = apport_lateral.shape[1]; T = apport_lateral.shape[0]
    isr = P["is_river"]
    qa = torch.zeros(nR, device=dev, dtype=dt_type)
    qb = torch.zeros(nR, device=dev, dtype=dt_type)
    ql = torch.zeros(nR, device=dev, dtype=dt_type)
    river_idx = torch.nonzero(isr, as_tuple=True)[0]
    out = []

    for day in range(T):
        apl = apport_lateral[day]
        # nt global (rivières), détaché (contrôle discret)
        with torch.no_grad():
            c_riv = _celerite_vec(P["lng"][river_idx], P["lrg"][river_idx], P["pte"][river_idx],
                                  P["man"][river_idx], qa[river_idx], qb[river_idx])
            c_riv = torch.clamp(c_riv, min=1e-4)
            nt_i = (pdts / (P["lng"][river_idx] / c_riv + 1.0).floor()).floor() + 1
            ntmax = int(nt_i.max().item()) if nt_i.numel() else 1
        if pdts / max(ntmax, 1) < 1800:
            dt = 1800; nt = max(1, int(pdts / dt))
        else:
            nt = max(ntmax, 1); dt = int(pdts / nt)

        qaval_moy = torch.zeros(nR, device=dev, dtype=dt_type)
        for t in range(nt):
            qamont_acc = torch.zeros(nR, device=dev, dtype=dt_type)
            for g in level_groups:
                qc = qamont_acc[g]
                qm = apl[g]
                qa_g, qb_g, ql_g = qa[g], qb[g], ql[g]
                rmask = isr[g]
                qd = torch.zeros_like(qc)
                # rivières du niveau
                if bool(rmask.any()):
                    ri = g[rmask]
                    qd_r = _transfert_riviere_vec(dt, P["lng"][ri], P["lrg"][ri], P["pte"][ri], P["man"][ri],
                                                  qa[ri], ql[ri], qb[ri], qamont_acc[ri], apl[ri])
                    qd = qd.masked_scatter(rmask, qd_r)
                # lacs du niveau
                lmask = ~rmask
                if bool(lmask.any()):
                    li = g[lmask]
                    qd_l = _transfert_lac_vec(dt, P["surface_m2"][li], P["c"][li], P["k"][li],
                                              qa[li], ql[li], qb[li], qamont_acc[li], apl[li])
                    qd = qd.masked_scatter(lmask, qd_l)
                qd = torch.clamp(qd, min=0.0)
                # scatter qd vers le qamont du tronçon aval
                dg = downstream[g]
                valid = dg >= 0
                if bool(valid.any()):
                    qamont_acc = qamont_acc.index_add(0, dg[valid], qd[valid])
                # MAJ état (qa=qc courant, qb=qd, ql=qm) pour ces tronçons
                qa = qa.index_copy(0, g, qc)
                qb = qb.index_copy(0, g, qd)
                ql = ql.index_copy(0, g, qm)
                qaval_moy = qaval_moy.index_copy(
                    0, g, qd if t == 0 else (qaval_moy[g] * t + qd) / (t + 1.0))
        out.append(qaval_moy)
    return torch.stack(out, dim=0)
