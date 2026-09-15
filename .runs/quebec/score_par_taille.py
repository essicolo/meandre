"""Le modele est-il moins bon sur les petits bassins versants ?

Question d'Essi, 2026-09-15 : l'Ontario dispose de stations sur de petits bassins, et un
collegue propose d'y employer un modele purement appris. Avant d'en discuter, il faut
savoir ce que vaut meandre en fonction de la taille du bassin controle.

On croise le KGE de la periode d'evaluation 2022-2024, par station, avec l'aire drainee
declaree de la station. Aucune simulation : les series sont celles deja calculees.

    .venv/Scripts/python.exe .runs/quebec/score_par_taille.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]


def kge(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 365:
        return None
    o, s = o[m].astype(float), s[m].astype(float)
    if o.std() < 1e-9 or s.std() < 1e-9 or o.mean() <= 0:
        return None
    r = float(np.corrcoef(o, s)[0, 1])
    b = float(s.mean() / o.mean())
    g = float((s.std() / s.mean()) / (o.std() / o.mean()))
    return 1 - float(np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)), r, b, g


def main():
    racine = _paths.DATA_ROOT
    lignes = []
    for reg in REGIONS:
        f = f"{racine}/quebec/flotte/q-{reg}-B.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        cx = duckdb.connect(f"{racine}/quebec/{reg}.duckdb", read_only=True)
        st = cx.sql("select station_id, drainage_area_km2 from stations").fetchdf()
        cx.close()
        aire = dict(zip(st.station_id.astype(str), st.drainage_area_km2))
        for j, sid in enumerate(z["station_ids"]):
            k = kge(z["q_obs"][:, j], z["q_sim"][:, j])
            a = aire.get(str(sid))
            if k is None or not np.isfinite(k[0]) or a is None or not np.isfinite(a):
                continue
            lignes.append({"region": reg, "station": str(sid), "aire_km2": float(a),
                           "kge": k[0], "r": k[1], "beta": k[2], "gamma": k[3]})
    d = pd.DataFrame(lignes)
    if d.empty:
        print("aucune station")
        return 1
    print(f"{len(d)} stations, {d.region.nunique()} régions, période d'évaluation 2022-2024")
    print(f"aire drainée : médiane {d.aire_km2.median():.0f} km², "
          f"q05 {d.aire_km2.quantile(.05):.0f}, q95 {d.aire_km2.quantile(.95):.0f} km²\n")

    # Corrélation de rang entre l'aire et chaque terme, sans découper en classes.
    from scipy.stats import spearmanr
    la = np.log10(d.aire_km2.values)
    print("corrélation de rang de Spearman avec le logarithme de l'aire drainée")
    for col in ("kge", "r", "beta", "gamma"):
        rho, p = spearmanr(la, d[col].values)
        print(f"  {col:>6} : rho {rho:+.3f} | p {p:.4f}")

    print("\nmédiane par tranche d'aire, pour la lecture seulement")
    for lo, hi in ((0, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 1e9)):
        q = d[(d.aire_km2 >= lo) & (d.aire_km2 < hi)]
        if len(q) < 3:
            continue
        et = f"{lo:.0f}-{hi:.0f}" if hi < 1e9 else f"plus de {lo:.0f}"
        print(f"  {et:>12} km² : n {len(q):3d} | KGE méd {q.kge.median():+.3f} "
              f"| r {q.r.median():.3f} | beta {q.beta.median():.3f} | gamma {q.gamma.median():.3f}")

    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        d.to_csv(f"{sortie}/score-par-taille.csv", index=False)
        print(f"\ncache : {sortie}/score-par-taille.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
