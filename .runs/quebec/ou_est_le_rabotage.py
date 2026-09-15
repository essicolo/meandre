"""Quelles regions portent REELLEMENT le rabotage des pointes ?

Le 2026-09-15, l'epreuve appariee de l'ancrage du routage a ete menee sur le Saguenay et la
Gaspesie, dont les temoins avaient des pointes a 0,81 et 0,95 : ces regions ne rabotaient
presque pas, et le remede ne pouvait que depasser la cible. C'est la quatrieme condition
prealable du fichier de consignes, violee.

Avant toute nouvelle epreuve, il faut donc savoir ou le defaut se trouve. On le mesure sur
les caches par station de la periode d'evaluation, qui viennent d'etre refaits.

    .venv/Scripts/python.exe .runs/quebec/ou_est_le_rabotage.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm"]


def main():
    rep = f"{_paths.DATA_ROOT}/quebec/rapport"
    lignes = []
    for reg in REGIONS:
        f = f"{rep}/rap-{reg}-q.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        qs, qo = z["q_sim"], z["q_obs"]
        t = pd.DatetimeIndex([str(x)[:10] for x in z["dates"]])
        an = t.year.to_numpy()
        pics, nerf, plat, kges = [], [], [], []
        for j in range(qs.shape[1]):
            o, s = qo[:, j].astype(float), qs[:, j].astype(float)
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() < 365 or np.nanmean(o) <= 0:
                continue
            rp = []
            for a in np.unique(an[m]):
                k = m & (an == a)
                if k.sum() < 300 or np.nanmax(o[k]) <= 0:
                    continue
                rp.append(np.nanmax(s[k]) / np.nanmax(o[k]))
            if rp:
                pics.append(np.median(rp))
            vo, vs = np.std(np.diff(o[m])) / o[m].mean(), np.std(np.diff(s[m])) / s[m].mean()
            if vo > 0:
                nerf.append(vs / vo)
            ps = np.abs(np.diff(s[m])) / np.maximum(s[m][:-1], 1e-9) < 0.01
            po = np.abs(np.diff(o[m])) / np.maximum(o[m][:-1], 1e-9) < 0.01
            plat.append(ps.mean() - po.mean())
            r = np.corrcoef(o[m], s[m])[0, 1]
            b = s[m].mean() / o[m].mean()
            g = (s[m].std() / s[m].mean()) / (o[m].std() / o[m].mean())
            kges.append(1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))
        if not pics:
            continue
        lignes.append({"region": reg, "stations": len(kges),
                       "pointes": float(np.median(pics)),
                       "nervosite": float(np.median(nerf)) if nerf else np.nan,
                       "exces_de_platitude_pts": 100 * float(np.median(plat)),
                       "kge": float(np.median(kges))})
    d = pd.DataFrame(lignes).sort_values("pointes")
    print("période d'évaluation 2022-2024, médiane des stations de chaque région\n")
    print(f"{'région':>7} {'stations':>9} {'pointes sim/obs':>16} {'nervosité':>10} "
          f"{'excès de platitude':>19} {'KGE':>7}")
    for r in d.itertuples():
        print(f"{r.region:>7} {r.stations:9d} {r.pointes:16.2f} {r.nervosite:10.2f} "
              f"{r.exces_de_platitude_pts:17.1f} pt {r.kge:7.3f}")
    print(f"\nrégions dont les pointes sont sous 0,75 : "
          f"{', '.join(d[d.pointes < 0.75].region) or 'aucune'}")
    print(f"régions dont la nervosité est sous 0,70 : "
          f"{', '.join(d[d.nervosite < 0.70].region) or 'aucune'}")
    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        d.to_csv(f"{sortie}/ou-est-le-rabotage.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
