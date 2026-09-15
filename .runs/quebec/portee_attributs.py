"""A quelle echelle les attributs territoriaux portent-ils encore de l'information ?

Question d'Essi, 2026-09-15 : les cartes de sol et la topographie pourraient-elles aider
les petits bassins ? Le champ spatial ne cale pas les parametres d'un bassin, il les deduit
de seize attributs. Si deux troncons voisins recoivent la meme valeur d'attribut, le champ
ne peut pas les distinguer, et aucun progres de l'inversion n'y changera rien.

On mesure la correlation de chaque attribut entre troncons separes par une distance
croissante. Une correlation proche de un a quelques kilometres signifie une carte lisse a
cette echelle, donc sans pouvoir de distinction pour un bassin de quelques troncons.

    .venv/Scripts/python.exe .runs/quebec/portee_attributs.py
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

ATTRIBUTS = ["mean_slope_pct", "mean_elevation_m", "f_forest", "f_agriculture", "f_urban",
             "f_wetland", "f_water", "f_sand", "f_silt", "f_clay", "depth_to_bedrock_m",
             "lake_fraction"]
BANDES = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 300)]


def main(regions):
    racine = _paths.DATA_ROOT
    acc = {a: {b: [] for b in BANDES} for a in ATTRIBUTS}
    rng = np.random.default_rng(1234)
    for reg in regions:
        base = f"{racine}/quebec/{reg}.duckdb"
        if not os.path.exists(base):
            continue
        cx = duckdb.connect(base, read_only=True)
        nd = cx.sql("select node_idx, lon, lat from nodes order by node_idx").fetchdf()
        cx.close()
        rw = pd.read_parquet(f"{racine}/quebec/territorial-raw-QC.parquet")
        rw = rw[rw.region == reg].reset_index(drop=True)
        if len(rw) != len(nd):
            continue
        n = len(nd)
        idx = rng.choice(n, size=min(n, 1200), replace=False)
        lat0 = float(nd.lat.mean())
        kx = 111.0 * np.cos(np.radians(lat0))
        x = nd.lon.values[idx] * kx
        y = nd.lat.values[idx] * 111.0
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        iu = np.triu_indices(len(idx), 1)
        dd = d[iu]
        for a in ATTRIBUTS:
            v = rw[a].values[idx].astype(float)
            if not np.isfinite(v).all() or v.std() < 1e-12:
                continue
            vi, vj = v[iu[0]], v[iu[1]]
            for b in BANDES:
                m = (dd >= b[0]) & (dd < b[1])
                if m.sum() < 200:
                    continue
                acc[a][b].append(float(np.corrcoef(vi[m], vj[m])[0, 1]))
    print(f"corrélation d'un attribut entre deux tronçons séparés de la distance indiquée,")
    print(f"médiane sur {len(regions)} régions\n")
    en = "  ".join(f"{b[0]:>3}-{b[1]:<3}" for b in BANDES)
    print(f"{'attribut':>20}  {en}")
    for a in ATTRIBUTS:
        vals = []
        for b in BANDES:
            vals.append(np.median(acc[a][b]) if acc[a][b] else np.nan)
        ligne = "  ".join(f"{v:7.2f}" for v in vals)
        print(f"{a:>20}  {ligne}")
    print("\nles colonnes sont des kilomètres ; un tronçon médian de rivière fait 5,7 km")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["outv", "gasp", "sagu", "mont", "slno", "abit"]))
