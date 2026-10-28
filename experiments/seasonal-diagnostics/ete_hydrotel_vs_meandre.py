"""L'ensemble Hydrotel a-t-il l'ete d'un territoire ? Rapport simule sur observe par mois.

Pour chaque station du territoire, sur 2022-2024 et les memes jours mesures : les six calages
d'Hydrotel (posttraitement_*.zarr, debit par troncon) et meandre (export de stations
ETL_DUMP_Q). Mediane des rapports par station et par mois. Pose pour la Monteregie, dont
l'ete simule par meandre vaut la moitie de l'observe (R326, R329) : si Hydrotel, meme
PHYSITEL et krigeage des stations, a l'ete juste, PHYSITEL n'est pas la limite.

    python .runs/quebec/ete_hydrotel_vs_meandre.py mont <q-mont.npz> [<q-mont-autre.npz> ...]
"""
import glob
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import duckdb
import numpy as np
import pandas as pd
import xarray as xr

from meandre.data.basin_cache import BasinCache

_DATA_ROOT = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
_RQH = os.environ.get("MEANDRE_RQH", "D:/rqh")
T0, T1 = os.environ.get("MEANDRE_ETE_T0", "2022-01-01"), os.environ.get("MEANDRE_ETE_T1", "2024-12-31")


def rapports_mensuels(sim: pd.Series, obs: pd.Series) -> np.ndarray:
    """Rapport des moyennes mensuelles simule / observe, douze valeurs, NaN si moins de 20 jours."""
    j = pd.concat([sim, obs], axis=1, join="inner").dropna()
    out = np.full(12, np.nan)
    for m in range(1, 13):
        s = j[j.index.month == m]
        if len(s) >= 20 and s.iloc[:, 1].mean() > 0:
            out[m - 1] = s.iloc[:, 0].mean() / s.iloc[:, 1].mean()
    return out


def main(reg: str, exports: list[str]):
    REG = reg.upper()
    db = f"{_DATA_ROOT}/quebec/{reg.lower()}.duckdb"
    node_ids = BasinCache(db).load(device="cpu")["node_ids"]
    c = duckdb.connect(db, read_only=True)
    stations = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
    obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
    c.close()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    membres = sorted(glob.glob(f"{_RQH}/rqh_2026-04/data/06_posttraitement/posttraitement_*.zarr"))
    lignes = {}
    for zp in membres:
        z = xr.open_zarr(zp)
        tid = z["troncon_id"].values.astype(str)
        pos = {t: i for i, t in enumerate(tid)}
        temps = pd.to_datetime(z["time"].values)
        sel = (temps >= T0) & (temps <= T1)
        r = []
        for _, st in stations.iterrows():
            o = obs[obs.station_id == st.station_id].set_index("date")["discharge"]
            t = f"{REG}{int(node_ids[int(st.node_idx)]):05d}"
            if o.notna().sum() < 60 or t not in pos:
                continue
            dis = pd.Series(z["Dis"][pos[t], sel].values, index=temps[sel])
            r.append(rapports_mensuels(dis, o))
        z.close()
        nom = os.path.basename(zp).replace("posttraitement_", "").replace(".zarr", "")
        lignes[f"Hydrotel {nom}"] = (np.nanmedian(np.array(r), axis=0), len(r))
    for ex in exports:
        d = np.load(ex, allow_pickle=True)
        dates = pd.to_datetime(d["dates"])
        r = []
        for k, sid in enumerate(d["station_ids"]):
            o = obs[obs.station_id == str(sid)].set_index("date")["discharge"]
            if o.notna().sum() < 60:
                continue
            s = pd.Series(d["q_sim"][:, k], index=dates)
            r.append(rapports_mensuels(s, o))
        lignes[f"meandre {os.path.basename(ex)[:-4]}"] = (np.nanmedian(np.array(r), axis=0), len(r))
    print(f"{REG}, rapport simule / observe par mois, mediane des stations, {T0[:4]}-{T1[:4]}, memes jours mesures")
    print(f"{'':42s} n   " + " ".join(f"{m:5d}" for m in range(1, 13)))
    for nom, (v, n) in lignes.items():
        print(f"{nom:42s} {n:2d}  " + " ".join(f"{x:5.2f}" for x in v))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
