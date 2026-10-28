"""Evapotranspiration mensuelle de trois sources sur les noeuds d'un territoire : SSEBop v6.1
(bilan d'energie, temperature de surface VIIRS), MOD16 (Penman-Monteith) et, si fourni, le
modele (bilan mensuel ETL_BILAN). Moyenne des noeuds, et separement sur les noeuds agricoles
(part > 0,5) et forestiers (part > 0,7), 2013-2024, en mm/mois.

Question (2026-10-09) : MOD16 est-il trop haut en ete dans le sud agricole ? SSEBop voit le
stress hydrique par la temperature de surface, MOD16 non.

    python .runs/quebec/et_sources_comparees.py mont outv
"""
import glob
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import duckdb
import numpy as np
import pandas as pd
import rasterio

from meandre.data.basin_cache import BasinCache

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
SSEBOP = os.environ.get("MEANDRE_SSEBOP", f"{DATA}/ssebop/v61")


def ssebop_aux_noeuds(lon, lat):
    """Serie mensuelle SSEBop (mm/mois) aux noeuds : dict {Timestamp: valeurs}."""
    out = {}
    for z in sorted(glob.glob(f"{SSEBOP}/m*.zip")):
        ym = os.path.basename(z)[1:7]
        tif = f"/vsizip/{z}/m{ym}_viirsSSEBopETv61_actual_mm.tif"
        try:
            with rasterio.open(tif) as r:
                v = np.array([x[0] for x in r.sample(zip(lon, lat))], dtype=float)
                if r.nodata is not None:
                    v[v == r.nodata] = np.nan
        except Exception:
            continue
        v[(v < 0) | (v > 400)] = np.nan
        out[pd.Timestamp(f"{ym[:4]}-{ym[4:]}-01")] = v
    return out


def main(territoires):
    raw = pd.read_parquet(f"{DATA}/quebec/territorial-raw-QC.parquet")
    for reg in territoires:
        db = f"{DATA}/quebec/{reg}.duckdb"
        h = BasinCache(db).load(device="cpu")
        xy = h["node_coords"].cpu().numpy()
        lon, lat = xy[:, 0], xy[:, 1]
        rr = raw[raw.region == reg].reset_index(drop=True)
        groupes = {"tous": np.ones(len(lon), bool), "agricoles": rr.f_agriculture.to_numpy() > 0.5, "forestiers": rr.f_forest.to_numpy() > 0.7}
        s = ssebop_aux_noeuds(lon, lat)
        con = duckdb.connect(db, read_only=True)
        m16 = con.execute("select date, node_idx, etr_mm_day from modis_et where quality_ok and year(date) between 2013 and 2024").fetchdf()
        con.close()
        m16["mois"] = pd.to_datetime(m16.date).dt.month
        print(f"\n{reg} : {len(lon)} noeuds ({int(groupes['agricoles'].sum())} agricoles, {int(groupes['forestiers'].sum())} forestiers), mm/mois, 2013-2024")
        print("  " + "groupe      source   " + " ".join(f"{m:5d}" for m in range(4, 11)) + "   avr-oct")
        for g, masque in groupes.items():
            if masque.sum() == 0:
                continue
            ligne_s, ligne_m = [], []
            for m in range(4, 11):
                vs = [np.nanmean(v[masque]) for t, v in s.items() if t.month == m and 2013 <= t.year <= 2024]
                ligne_s.append(np.nanmean(vs) if vs else np.nan)
                sel = m16[(m16.mois == m) & m16.node_idx.isin(np.where(masque)[0])]
                ligne_m.append(sel.etr_mm_day.mean() * 30.4 if len(sel) else np.nan)
            print(f"  {g:10s}  SSEBop   " + " ".join(f"{x:5.0f}" for x in ligne_s) + f"   {np.nansum(ligne_s):6.0f}")
            print(f"  {'':10s}  MOD16    " + " ".join(f"{x:5.0f}" for x in ligne_m) + f"   {np.nansum(ligne_m):6.0f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["mont", "outv"])
