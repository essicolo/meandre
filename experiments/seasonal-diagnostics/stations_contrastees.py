"""Par station, deux territoires : KGE et ete de meandre et d'Hydrotel contre les attributs du bassin amont.

Question d'Essi (2026-10-09) : pourquoi le sud de l'Outaouais fonctionne et pas la Monteregie ?
Pour chaque station : aire drainee, parts agricole et forestiere et pente moyennes du bassin
amont, fraction de lac, KGE de meandre (export ETL_DUMP_Q) et du calage LN24HA d'Hydrotel sur
les memes jours, rapport simule/observe de juin a aout, et KGE de l'observe decale d'un jour
contre lui-meme (nervosite).

    python .runs/quebec/stations_contrastees.py <q-mont.npz> <q-outv.npz>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import duckdb
import numpy as np
import pandas as pd
import torch
import xarray as xr

from meandre.data.basin_cache import BasinCache
from trajectoire_champ import bassin_amont

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
RQH = os.environ.get("MEANDRE_RQH", "D:/rqh")


def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o)
    s, o = s[m], o[m]
    if len(s) < 60 or o.std() < 1e-9 or s.std() < 1e-9:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2 + ((s.std() / s.mean()) / (o.std() / o.mean()) - 1) ** 2)


def main(fichiers):
    z = xr.open_zarr(f"{RQH}/rqh_2026-04/data/06_posttraitement/posttraitement_LN24HA.zarr")
    zp = {t: i for i, t in enumerate(z["troncon_id"].values.astype(str))}
    zt = pd.to_datetime(z["time"].values)
    raw = pd.read_parquet(f"{DATA}/quebec/territorial-raw-QC.parquet")
    rows = []
    for f in fichiers:
        reg = os.path.basename(f).split("-")[1]
        db = f"{DATA}/quebec/{reg}.duckdb"
        h = BasinCache(db).load(device=torch.device("cpu"))
        graph, nid = h["graph"], h["node_ids"]
        n = h["node_coords"].shape[0]
        con = duckdb.connect(db, read_only=True)
        idx = dict(con.execute("select station_id, node_idx from stations").fetchall())
        con.close()
        rr = raw[raw.region == reg].reset_index(drop=True)
        d = np.load(f, allow_pickle=True)
        dates = pd.to_datetime(d["dates"])
        ete = np.isin(dates.month, (6, 7, 8))
        for k, sid in enumerate(d["station_ids"]):
            s, o = d["q_sim"][:, k], d["q_obs"][:, k]
            if np.isfinite(o).sum() < 300 or str(sid) not in idx:
                continue
            ni = int(idx[str(sid)])
            am = bassin_amont(graph.edge_index, n, ni)
            tid = f"{reg.upper()}{int(nid[ni]):05d}"
            kh = eteh = np.nan
            if tid in zp:
                hy = pd.Series(z["Dis"][zp[tid], :].values, index=zt).reindex(dates).to_numpy()
                kh = kge(hy, o)
                m = ete & np.isfinite(o) & np.isfinite(hy)
                eteh = hy[m].mean() / o[m].mean() if m.sum() > 60 else np.nan
            m = ete & np.isfinite(o) & np.isfinite(s)
            rows.append(dict(territoire=reg, station=str(sid), aire_km2=float(rr.drainage_area_km2.iloc[ni]),
                             agri=float(rr.f_agriculture.iloc[am].mean()), foret=float(rr.f_forest.iloc[am].mean()),
                             pente=float(rr.mean_slope_pct.iloc[am].mean()), lac=float(rr.lake_fraction.iloc[am].mean()),
                             kge=kge(s, o), kge_hydrotel=kh, ete=s[m].mean() / o[m].mean(), ete_hydrotel=eteh,
                             kge_decal_1j=kge(o[1:], o[:-1])))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.sort_values(["territoire", "agri"]).round(2).to_string(index=False))
    print()
    cols = ["kge", "kge_hydrotel", "ete", "ete_hydrotel", "kge_decal_1j"]
    print(pd.DataFrame({a: t[cols].corrwith(t[a] if a != "aire_km2" else np.log(t[a])) for a in ("agri", "foret", "pente", "lac", "aire_km2")}).round(2).to_string())
    return t


if __name__ == "__main__":
    main(sys.argv[1:])
