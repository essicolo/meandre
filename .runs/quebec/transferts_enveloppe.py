"""Les trois transferts que l'enveloppe probabiliste doit franchir, mesures separement.

La tete de quantiles est ajustee en post-traitement sur le debit deja simule, le modele
physique restant gele. Sa mediane est le debit deterministe, donc aucun score de debit
n'est modifie et seule l'enveloppe s'ajoute. Reste a savoir jusqu'ou elle s'exporte.

  TEMPS    ajustee sur la premiere moitie de la periode, jugee sur la seconde.
  STATIONS ajustee sur la moitie des stations, jugee sur les autres, toute la periode.
  REGIONS  ajustee sur huit regions, jugee sur la neuvieme.

La tete ne recoit ici AUCUNE identite de station : l'enveloppe est une fonction du seul
debit simule. C'est la condition pour qu'elle s'applique a un troncon non jauge, ou aucun
vecteur de station n'existe. Les deux derniers exercices mesurent precisement cela.

    .venv/Scripts/python.exe .runs/quebec/transferts_enveloppe.py
"""
import os

import numpy as np
import pandas as pd
import torch

from meandre.utils.quantile_head import QuantileHead

TAUS = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95)
T7 = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
REGS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cndb", "cndc"]
DOS = os.environ.get("MEANDRE_FLOTTE", "D:/meandre-data/quebec/flotte")
SORTIE = ".reports/quebec/caches/transferts-enveloppe.csv"
PAS = 600


def charger(reg):
    z = np.load(f"{DOS}/q-{reg}-A-v4.npz", allow_pickle=True)
    return torch.tensor(z["q_sim"], dtype=torch.float32), torch.tensor(z["q_obs"], dtype=torch.float32)


def ajuster(paires, pas=PAS):
    """Une seule tete, sans identite de station : un vecteur spatial partage et nul."""
    torch.manual_seed(0)
    sp = torch.zeros(1, 8)
    h = QuantileHead(n_spatial_params=8, taus=TAUS)
    opt = torch.optim.Adam(h.parameters(), lr=1e-2)
    for _ in range(pas):
        tot, npt = 0.0, 0
        for Qx, Yx in paires:
            o = h(sp, Qx.reshape(-1, 1)).squeeze(1)
            qv = Qx.reshape(-1, 1)
            q = torch.cat([qv + o[:, :3], qv, qv + o[:, 3:]], -1)
            yv = Yx.reshape(-1)
            m = torch.isfinite(yv) & torch.isfinite(q).all(-1)
            d = yv.unsqueeze(-1) - q
            tot = tot + torch.maximum(T7 * d, (T7 - 1) * d)[m].sum()
            npt += int(m.sum())
        opt.zero_grad()
        (tot / max(npt, 1)).backward()
        opt.step()
    return h, sp


def juger(h, sp, paires):
    ys, qs = [], []
    with torch.no_grad():
        for Qx, Yx in paires:
            o = h(sp, Qx.reshape(-1, 1)).squeeze(1)
            qv = Qx.reshape(-1, 1)
            q = torch.cat([qv + o[:, :3], qv, qv + o[:, 3:]], -1).numpy()
            yv = Yx.reshape(-1).numpy()
            m = np.isfinite(yv) & np.isfinite(q).all(-1)
            ys.append(yv[m])
            qs.append(q[m])
    y, q = np.concatenate(ys), np.concatenate(qs)
    return (len(y), float(np.mean((y > q[:, 2]) & (y < q[:, 4]))),
            float(np.mean((y > q[:, 0]) & (y < q[:, 6]))))


def main():
    don = {r: charger(r) for r in REGS}
    lignes = []

    for r in REGS:
        Q, Y = don[r]
        c = Q.shape[0] // 2
        h, sp = ajuster([(Q[:c], Y[:c])])
        n, c50, c90 = juger(h, sp, [(Q[c:], Y[c:])])
        lignes.append({"transfert": "temps", "region": r.upper(), "n": n, "c50": c50, "c90": c90})
        print(f"temps    {r.upper()} n={n:6d} | 50 % -> {c50:.3f} | 90 % -> {c90:.3f}")

    for r in REGS:
        Q, Y = don[r]
        ns = Q.shape[1]
        if ns < 4:
            continue
        idx = np.random.default_rng(0).permutation(ns)
        a, b = idx[: ns // 2], idx[ns // 2:]
        h, sp = ajuster([(Q[:, a], Y[:, a])])
        n, c50, c90 = juger(h, sp, [(Q[:, b], Y[:, b])])
        lignes.append({"transfert": "stations", "region": r.upper(), "n": n, "c50": c50, "c90": c90})
        print(f"stations {r.upper()} {len(a)}->{len(b)} n={n:6d} | 50 % -> {c50:.3f} | 90 % -> {c90:.3f}")

    for r in REGS:
        h, sp = ajuster([don[x] for x in REGS if x != r])
        n, c50, c90 = juger(h, sp, [don[r]])
        lignes.append({"transfert": "régions", "region": r.upper(), "n": n, "c50": c50, "c90": c90})
        print(f"régions  {r.upper()} n={n:6d} | 50 % -> {c50:.3f} | 90 % -> {c90:.3f}")

    os.makedirs(os.path.dirname(SORTIE), exist_ok=True)
    pd.DataFrame(lignes).to_csv(SORTIE, index=False)
    print(f"\n{SORTIE} ecrit ({len(lignes)} lignes)")


if __name__ == "__main__":
    main()
