"""Routage par opérateur : solve triangulaire précalculé au lieu du balayage par niveau.

Le balayage topologique de ``RoutingLayer._route_vectorized`` exécute ~n_levels
petites opérations GPU par pas de temps (109-167 niveaux sur SLSO), soit des
millions de lancements de kernels par epoch : le forward ET le backward sont
dominés par cette surcharge (profil 2026-06-10 : routage = 96 % du forward).

Muskingum étant linéaire et ses coefficients constants dans le temps (K, x
appris ne dépendent pas de t), la propagation intra-journalière à travers les
rivières est un système triangulaire inférieur :

    Q_out = D_alpha (A Q_out + net_W) + gamma . Q_prev + beta . q_lat
    (I - D_alpha A) Q_out = rhs(t)

où, pour n sous-pas Muskingum (récurrence Q <- c01*Q_in + c2*Q + q_lat/n) :
    S_geo = 1 + c2 + ... + c2^(n-1),  alpha = c01*S_geo,
    gamma = c2^n,  beta = S_geo/n.

W = (I - D_alpha A)^-1 est précalculé UNE FOIS par forward (par appel à
``simulate``, donc par chunk d'entraînement) via un solve triangulaire dense
différentiable ; chaque pas de temps devient un produit matrice-vecteur.

Les lacs sont non linéaires (tarage puissance, Newton implicite) et coupent la
linéarité. Deux modes :

* ``operator`` (étagé, sémantique identique au balayage) : les nœuds sont
  partitionnés en étages séparés par les lacs (stage(i) = max de lacs sur un
  chemin amont). Chaque étage est un bloc rivière linéaire résolu par son
  W_k, puis les lacs de l'étage sont mis à jour (même module Newton que le
  balayage, apport du jour). Profondeur séquentielle = nb d'étages (8 sur
  SLSO PHYSITEL, 46 sur l'open data 2222) au lieu du nb de niveaux (109/167).

* ``operator-lagged`` (un seul solve) : les lacs relâchent selon leur stockage
  de la VEILLE (tarage explicite plafonné à S_dispo/dt pour la stabilité),
  donc toutes les sorties de lacs sont connues en début de pas et le réseau
  rivière entier se résout en un seul W. Approximation : un jour de délai de
  réponse aux apports pour chaque lac (défendable pour de vrais lacs, à
  valider numériquement contre le balayage).

Le clamp >= 0 par sous-pas du balayage devient un relu post-solve (les valeurs
négatives ne peuvent venir que des prélèvements net_W < 0 ; l'écart est validé
numériquement contre le balayage).

La température n'est pas portée ici (inutilisée par la perte d'entraînement) :
RoutingLayer retombe sur le balayage par niveau quand la thermie est demandée.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from meandre.routing.graph import RiverGraph


class OperatorTopo(NamedTuple):
    """Topologie par étages, précalculée une fois par graphe (cache)."""
    n_stages: int
    # Rivières, ordonnées étage par étage (ordre topologique intra-étage)
    riv_idx: Tensor            # (n_riv,) indices globaux
    riv_offsets: list[int]     # (n_stages+1,) découpage de riv_idx par étage
    riv_pos: Tensor            # (n_nodes,) position locale dans son étage, -1 sinon
    # Arêtes rivière->rivière INTRA-étage (entrées de la matrice par étage)
    intra_src_pos: Tensor      # (n_intra,) position locale du src dans l'étage
    intra_dst_pos: Tensor      # (n_intra,) position locale du dst dans l'étage
    intra_dst_glob: Tensor     # (n_intra,) indice global du dst (porte alpha)
    intra_offsets: list[int]   # (n_stages+1,)
    # Arêtes "entrées connues" -> rivières (src résolu avant : rivière d'un
    # étage antérieur ou lac d'un étage antérieur), groupées par étage du dst
    known_src_glob: Tensor     # (n_known,)
    known_dst_pos: Tensor      # (n_known,) position locale du dst dans son étage
    known_offsets: list[int]   # (n_stages+1,)
    # Lacs groupés par étage, et leurs arêtes entrantes (src toujours résolu
    # avant le lac dans l'ordre étagé)
    lake_idx: Tensor           # (n_lakes,) indices globaux, groupés par étage
    lake_offsets: list[int]    # (n_stages+1,)
    lake_in_src_glob: Tensor   # (n_lake_in,)
    lake_in_dst_pos: Tensor    # (n_lake_in,) position dans lake_idx (globale)
    lake_in_offsets: list[int] # (n_stages+1,)


def build_operator_topo(graph: RiverGraph, lagged: bool) -> OperatorTopo:
    """Construit la topologie étagée. En mode lagged, un seul étage rivière
    (les lacs ne sont pas des barrières : leurs sorties sont connues)."""
    device = graph.edge_index.device
    n = graph.is_lake.shape[0]
    src = graph.edge_index[0].tolist()
    dst = graph.edge_index[1].tolist()
    is_lake = graph.is_lake.cpu().tolist()

    # Position topologique globale (pour l'ordre intra-étage)
    topo_pos = [0] * n
    for p, node in enumerate(graph.topo_order.cpu().tolist()):
        topo_pos[node] = p

    # stage(i) = max sur les chemins amont du nb de lacs traversés
    children: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for s, d in zip(src, dst):
        children[s].append(d)
        indeg[d] += 1
    stage = [0] * n
    if not lagged:
        from collections import deque
        q = deque(i for i in range(n) if indeg[i] == 0)
        ind = indeg[:]
        while q:
            u = q.popleft()
            for v in children[u]:
                cand = stage[u] + (1 if is_lake[u] else 0)
                if cand > stage[v]:
                    stage[v] = cand
                ind[v] -= 1
                if ind[v] == 0:
                    q.append(v)
    n_stages = max(stage) + 1 if n else 1

    # Rivières par étage, ordre topologique intra-étage
    riv_by_stage: list[list[int]] = [[] for _ in range(n_stages)]
    lake_by_stage: list[list[int]] = [[] for _ in range(n_stages)]
    for i in range(n):
        (lake_by_stage if is_lake[i] else riv_by_stage)[stage[i]].append(i)
    for k in range(n_stages):
        riv_by_stage[k].sort(key=lambda i: topo_pos[i])

    riv_idx: list[int] = []
    riv_offsets = [0]
    riv_pos = [-1] * n
    for k in range(n_stages):
        for j, i in enumerate(riv_by_stage[k]):
            riv_pos[i] = j
        riv_idx.extend(riv_by_stage[k])
        riv_offsets.append(len(riv_idx))

    lake_idx: list[int] = []
    lake_offsets = [0]
    lake_gpos = [-1] * n
    for k in range(n_stages):
        for i in lake_by_stage[k]:
            lake_gpos[i] = len(lake_idx)
            lake_idx.append(i)
        lake_offsets.append(len(lake_idx))

    # Classement des arêtes
    intra: list[list[tuple[int, int, int]]] = [[] for _ in range(n_stages)]
    known: list[list[tuple[int, int]]] = [[] for _ in range(n_stages)]
    lake_in: list[list[tuple[int, int]]] = [[] for _ in range(n_stages)]
    for s, d in zip(src, dst):
        if is_lake[d]:
            lake_in[stage[d]].append((s, lake_gpos[d]))
        elif (not is_lake[s]) and stage[s] == stage[d]:
            intra[stage[d]].append((riv_pos[s], riv_pos[d], d))
        else:
            known[stage[d]].append((s, riv_pos[d]))

    def flat3(groups):
        a, b, c, off = [], [], [], [0]
        for g in groups:
            for x, y, z in g:
                a.append(x); b.append(y); c.append(z)
            off.append(len(a))
        return a, b, c, off

    def flat2(groups):
        a, b, off = [], [], [0]
        for g in groups:
            for x, y in g:
                a.append(x); b.append(y)
            off.append(len(a))
        return a, b, off

    i_s, i_d, i_g, i_off = flat3(intra)
    k_s, k_d, k_off = flat2(known)
    l_s, l_d, l_off = flat2(lake_in)

    tl = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return OperatorTopo(
        n_stages=n_stages,
        riv_idx=tl(riv_idx), riv_offsets=riv_offsets, riv_pos=tl(riv_pos),
        intra_src_pos=tl(i_s), intra_dst_pos=tl(i_d), intra_dst_glob=tl(i_g),
        intra_offsets=i_off,
        known_src_glob=tl(k_s), known_dst_pos=tl(k_d), known_offsets=k_off,
        lake_idx=tl(lake_idx), lake_offsets=lake_offsets,
        lake_in_src_glob=tl(l_s), lake_in_dst_pos=tl(l_d), lake_in_offsets=l_off,
    )


class OperatorState(NamedTuple):
    """Opérateur par forward : coefficients affines + inverses par étage."""
    alpha: Tensor              # (n_nodes,)
    gamma: Tensor              # (n_nodes,)
    beta: Tensor               # (n_nodes,)
    W: list[Tensor]            # par étage : (N_k, N_k) = (I - D_alpha A_kk)^-1


def build_operator_state(
    topo: OperatorTopo, c01: Tensor, c2: Tensor, n_substeps: int,
) -> OperatorState:
    """Construit alpha/gamma/beta et les inverses de bloc. Différentiable
    (les gradients remontent vers K_musk/x_musk via c01/c2).

    Tout en float32 : sous autocast bf16, un solve/GEMV en demi-précision
    dégraderait les récurrences longues et fait des mélanges de dtypes.
    """
    c01 = c01.float()
    c2 = c2.float()
    s_geo = torch.ones_like(c2)
    acc = torch.ones_like(c2)
    for _ in range(n_substeps - 1):
        acc = acc * c2
        s_geo = s_geo + acc
    alpha = c01 * s_geo
    gamma = acc * c2                      # c2^n
    beta = s_geo / float(n_substeps)

    W: list[Tensor] = []
    for k in range(topo.n_stages):
        lo, hi = topo.riv_offsets[k], topo.riv_offsets[k + 1]
        nk = hi - lo
        if nk == 0:
            W.append(c01.new_zeros((0, 0)))
            continue
        e_lo, e_hi = topo.intra_offsets[k], topo.intra_offsets[k + 1]
        L = torch.eye(nk, dtype=c01.dtype, device=c01.device)
        if e_hi > e_lo:
            dpos = topo.intra_dst_pos[e_lo:e_hi]
            spos = topo.intra_src_pos[e_lo:e_hi]
            dglob = topo.intra_dst_glob[e_lo:e_hi]
            L = L.index_put((dpos, spos), -alpha[dglob], accumulate=True)
        W.append(torch.linalg.solve_triangular(
            L, torch.eye(nk, dtype=c01.dtype, device=c01.device), upper=False,
        ))
    return OperatorState(alpha=alpha, gamma=gamma, beta=beta, W=W)


def _lagged_lake_release(
    lake_module, S: Tensor, area_km2: Tensor,
    k_lake: Tensor | None = None, beta: Tensor | None = None,
) -> Tensor:
    """Tarage explicite sur le stockage de la veille, plafonné à S/dt.

    Q(S) = k_lake * ((S - S_dead)/A)^beta * A, S_dead = 0 (comme le balayage).
    Le plafond S/dt garantit la stabilité de l'Euler explicite (le lac ne peut
    pas relâcher plus que son stockage disponible en un jour). k_lake/beta par
    nœud (NeRF) ou scalaires globaux du module (None).
    """
    dt = 86400.0
    k = lake_module.k_lake if k_lake is None else k_lake
    b = lake_module.beta if beta is None else beta
    A_safe = (area_km2 * 1e6).clamp(min=1.0)
    depth = (S.clamp(min=0.0) / A_safe).clamp(min=1e-6)
    Q_rate = k * depth ** b * A_safe
    return torch.minimum(Q_rate, S.clamp(min=0.0) / dt)


def route_operator(
    layer,                      # RoutingLayer (donne .muskingum, .lake)
    topo: OperatorTopo,
    op: OperatorState,
    q_lat_m3s: Tensor,
    Q_out_prev: Tensor,
    net_W: Tensor,
    lake_storage_new: Tensor | None,
    area_km2: Tensor | None,
    dam_data,
    t: int,
    lagged: bool,
) -> Tensor:
    """Un pas de routage par opérateur. Retourne Q_out (n_nodes,).

    Met à jour lake_storage_new sur place (stockage détaché, comme le balayage).
    """
    # fp32 forcé : sous autocast bf16 la colonne verticale peut produire des
    # entrées en demi-précision ; le routage (récurrence + solve) reste fp32.
    q_lat_m3s = q_lat_m3s.float()
    Q_out_prev = Q_out_prev.float()
    net_W = net_W.float()
    n = q_lat_m3s.shape[0]
    Q_out = q_lat_m3s.new_zeros(n)
    dt = 86400.0

    # Params de lac par nœud (NeRF) ou scalaires globaux (None)
    lake_k_all = getattr(layer, "_lake_k", None)
    lake_b_all = getattr(layer, "_lake_beta", None)

    # Mode lagged : toutes les sorties de lacs sont connues d'emblée
    if lagged and topo.lake_idx.numel() > 0 and lake_storage_new is not None:
        li = topo.lake_idx
        area_l = area_km2[li] if area_km2 is not None else torch.ones_like(q_lat_m3s[li])
        Q_lake = _lagged_lake_release(
            layer.lake, lake_storage_new[li], area_l,
            k_lake=lake_k_all[li] if lake_k_all is not None else None,
            beta=lake_b_all[li] if lake_b_all is not None else None,
        )
        if dam_data is not None:
            forced = dam_data.releases[t][li]
            regulated = ~torch.isnan(forced)
            if regulated.any():
                Q_lake = torch.where(regulated, forced, Q_lake)
        Q_out[li] = torch.clamp(Q_lake, min=0.0)

    for k in range(topo.n_stages):
        # 1. Bloc rivière de l'étage k : rhs puis solve (matvec sur W_k)
        lo, hi = topo.riv_offsets[k], topo.riv_offsets[k + 1]
        if hi > lo:
            ri = topo.riv_idx[lo:hi]
            known_in = q_lat_m3s.new_zeros(hi - lo)
            e_lo, e_hi = topo.known_offsets[k], topo.known_offsets[k + 1]
            if e_hi > e_lo:
                known_in.scatter_add_(
                    0, topo.known_dst_pos[e_lo:e_hi],
                    Q_out[topo.known_src_glob[e_lo:e_hi]],
                )
            rhs = (op.alpha[ri] * (known_in + net_W[ri])
                   + op.gamma[ri] * Q_out_prev[ri]
                   + op.beta[ri] * q_lat_m3s[ri])
            q_unc = op.W[k] @ rhs
            # Clamp >= 0 : un nœud asséché par les prélèvements (net_W < 0)
            # doit sortir 0 ET propager 0 à l'aval, comme le clamp par nœud du
            # balayage. Complémentarité résolue par ensemble actif : épingler
            # les violations à 0 (solve sur la sous-matrice de W, encore
            # triangulaire unitaire), dés-épingler ce qui redevient positif
            # une fois l'amont épinglé. Coût nul quand aucun négatif (cas
            # courant) ; sinon 2-3 itérations de petits solves.
            if bool((q_unc < 0).any()):
                Wk = op.W[k]
                e_lo2, e_hi2 = topo.intra_offsets[k], topo.intra_offsets[k + 1]
                spos = topo.intra_src_pos[e_lo2:e_hi2]
                dpos = topo.intra_dst_pos[e_lo2:e_hi2]
                pinned = torch.zeros(hi - lo, dtype=torch.bool, device=rhs.device)
                q_cur = q_unc
                for _ in range(5):
                    viol = (~pinned) & (q_cur < 0)
                    # Valeur affine "libre" des épinglés avec l'état courant
                    if bool(pinned.any()):
                        ups = q_lat_m3s.new_zeros(hi - lo)
                        if spos.numel():
                            ups.scatter_add_(0, dpos, q_cur[spos])
                        free_val = (op.alpha[ri] * (ups + known_in + net_W[ri])
                                    + op.gamma[ri] * Q_out_prev[ri]
                                    + op.beta[ri] * q_lat_m3s[ri])
                        unpin = pinned & (free_val > 0)
                    else:
                        unpin = pinned
                    if not bool(viol.any()) and not bool(unpin.any()):
                        break
                    pinned = (pinned | viol) & ~unpin
                    idx = pinned.nonzero(as_tuple=True)[0]
                    if idx.numel() == 0:
                        q_cur = q_unc
                        continue
                    z = torch.linalg.solve_triangular(
                        Wk[idx][:, idx], q_unc[idx].unsqueeze(1), upper=False,
                    ).squeeze(1)
                    q_cur = q_unc - Wk[:, idx] @ z
                    q_cur = q_cur.index_put((idx,), torch.zeros_like(z))
                q_unc = q_cur
            Q_out[ri] = torch.clamp(q_unc, min=0.0)

        # 2. Lacs de l'étage k (mode étagé : Newton avec l'apport du jour)
        if lagged:
            continue
        l_lo, l_hi = topo.lake_offsets[k], topo.lake_offsets[k + 1]
        if l_hi > l_lo and lake_storage_new is not None:
            li = topo.lake_idx[l_lo:l_hi]
            n_l = l_hi - l_lo
            lake_in = q_lat_m3s.new_zeros(topo.lake_idx.shape[0])
            e_lo, e_hi = topo.lake_in_offsets[k], topo.lake_in_offsets[k + 1]
            if e_hi > e_lo:
                lake_in.scatter_add_(
                    0, topo.lake_in_dst_pos[e_lo:e_hi],
                    Q_out[topo.lake_in_src_glob[e_lo:e_hi]],
                )
            Q_in_total = lake_in[l_lo:l_hi] + q_lat_m3s[li] + net_W[li]
            zeros_l = q_lat_m3s.new_zeros(n_l)
            area_l = area_km2[li] if area_km2 is not None else torch.ones_like(zeros_l)
            Q_lake, S_lake = layer.lake(
                Q_in_total, lake_storage_new[li], area_l,
                E_lake=zeros_l, P_lake=zeros_l, S_dead=zeros_l,
                k_lake=lake_k_all[li] if lake_k_all is not None else None,
                beta=lake_b_all[li] if lake_b_all is not None else None,
            )
            if dam_data is not None:
                forced = dam_data.releases[t][li]
                regulated = ~torch.isnan(forced)
                if regulated.any():
                    Q_lake = torch.where(regulated, forced, Q_lake)
                    S_reg = torch.clamp(
                        lake_storage_new[li] + (Q_in_total - forced) * dt, min=0.0,
                    )
                    S_lake = torch.where(regulated, S_reg, S_lake)
            Q_out[li] = torch.clamp(Q_lake, min=0.0)
            lake_storage_new[li] = S_lake.detach()

    # Mode lagged : mise à jour des stockages avec l'apport du jour (connu)
    if lagged and topo.lake_idx.numel() > 0 and lake_storage_new is not None:
        li = topo.lake_idx
        lake_in = q_lat_m3s.new_zeros(li.shape[0])
        if topo.lake_in_src_glob.numel() > 0:
            lake_in.scatter_add_(
                0, topo.lake_in_dst_pos, Q_out[topo.lake_in_src_glob],
            )
        Q_in_total = lake_in + q_lat_m3s[li] + net_W[li]
        S_new = torch.clamp(
            lake_storage_new[li] + (Q_in_total - Q_out[li]) * dt, min=0.0,
        )
        lake_storage_new[li] = S_new.detach()

    return Q_out
