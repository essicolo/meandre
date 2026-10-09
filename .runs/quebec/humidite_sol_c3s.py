"""Humidité du sol satellitaire C3S (ESA CCI, combinée) sur les nœuds d'un territoire (2026-10-09).

Surface (SSMV) et zone racinaire 0-10, 10-40, 40-100 cm et 0-1 m (RZSMV), moyennes mensuelles
2005-2024, en m³/m³, échantillonnées au nœud le plus proche de la grille de 0,25°. Imprime le cycle
de mai à octobre (moyenne des années) et, pour juillet-août, chaque année de 2015 à 2024. Question :
la zone racinaire sèche-t-elle en été, et le modèle (qui reste au-dessus de la capacité au champ,
R339) le fait-il ?

    python .runs/quebec/humidite_sol_c3s.py mont outv
"""
import glob
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
REP = f"{DATA}/esa_cci_sm/mensuel/nc"
VARS = {"surface": ("SSMV", "sm"), "0-10": ("RZSMV", "rzsm_1"), "10-40": ("RZSMV", "rzsm_2"), "40-100": ("RZSMV", "rzsm_3")}


def series(lon, lat):
    out = {k: {} for k in VARS}
    for nom, (motif, var) in VARS.items():
        for f in sorted(glob.glob(f"{REP}/*{motif}*.nc")):
            t = pd.Timestamp(os.path.basename(f).split("-")[-3][:8])
            with xr.open_dataset(f) as ds:
                v = ds[var].isel(time=0).sel(lon=xr.DataArray(lon, dims="n"), lat=xr.DataArray(lat, dims="n"), method="nearest").values.astype(float)
            out[nom][t] = np.nanmean(v)
    return {k: pd.Series(v).sort_index() for k, v in out.items()}


def main(territoires):
    for reg in territoires:
        xy = BasinCache(f"{DATA}/quebec/{reg}.duckdb").load(device="cpu")["node_coords"].cpu().numpy()
        s = series(xy[:, 0], xy[:, 1])
        print(f"\n{reg} : humidité du sol C3S, m³/m³, moyenne des nœuds")
        print("  couche     " + " ".join(f"{m:6d}" for m in range(5, 11)) + "   chute mai-août")
        for nom, x in s.items():
            cyc = [x[x.index.month == m].mean() for m in range(5, 11)]
            print(f"  {nom:9s}  " + " ".join(f"{c:6.3f}" for c in cyc) + f"   {cyc[3] / cyc[0] - 1:+.0%}")
        print("  juillet-août par année (0-1 m approché par la moyenne 10-40 et 40-100) :")
        z = (s["10-40"] + s["40-100"]) / 2
        print("   " + " ".join(f"{a}:{z[(z.index.year == a) & z.index.month.isin([7, 8])].mean():.3f}" for a in range(2015, 2025)))


if __name__ == "__main__":
    main(sys.argv[1:] or ["mont", "outv"])
