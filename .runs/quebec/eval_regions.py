"""Tableau récapitulatif Québec : held-out 2022-2024 par région, méandre vs Hydrotel BRUT.
- méandre : reach parquet (D:/meandre-data/quebec/results/reach-<reg>.parquet, reach_id = node_idx+1) ;
  SLSO utilise le champion (.runs/slso/results/reach-physitel-hydrotel-casr-zn.parquet).
- Hydrotel brut : posttraitement_LN24HA.zarr (Dis par troncon_id, 2020-2026, non interpolé).
- obs : duckdb de chaque région.
  python .runs/quebec/eval_regions.py [REG ...]
Sortie : reports/quebec_heldout.csv + tableau console.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb, xarray as xr
from meandre.data.basin_cache import BasinCache

QC = "D:/meandre-data/quebec"
ZARR = r"C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data/06_posttraitement/posttraitement_LN24HA.zarr"
T0, T1 = "2022-01-01", "2024-12-31"
REGIONS = [r.upper() for r in sys.argv[1:]] or ["LABI", "CNDC", "CNDA", "CNDE", "CNDB", "CNDD",
    "ABIT", "MONT", "SAGU", "OUTM", "SLNO", "OUTV", "GASP", "SLSO"]

def kge(qs, qo):
    m = np.isfinite(qs) & np.isfinite(qo)
    qs, qo = qs[m], qo[m]
    if len(qs) < 60 or qo.std() < 1e-9 or qs.std() < 1e-9: return np.nan
    r = np.corrcoef(qs, qo)[0, 1]; b = qs.mean() / qo.mean()
    g = (qs.std() / qs.mean()) / (qo.std() / qo.mean())
    return 1 - np.sqrt((r - 1)**2 + (b - 1)**2 + (g - 1)**2)

z = xr.open_zarr(ZARR)
z_tid = z["troncon_id"].values.astype(str)
z_pos = {t: i for i, t in enumerate(z_tid)}
z_time = pd.to_datetime(z["time"].values)
z_sl = (z_time >= T0) & (z_time <= T1)
z_dates = z_time[z_sl]

rows = []
for reg in REGIONS:
    db = f"{QC}/{reg.lower()}.duckdb" if reg != "SLSO" else ".runs/slso/data/slso.duckdb"
    pq = (f"{QC}/results/reach-{reg.lower()}.parquet" if reg != "SLSO"
          else ".runs/slso/results/reach-physitel-hydrotel-casr-zn.parquet")
    if not os.path.exists(pq):
        print(f"{reg}: parquet absent (pas encore entraîné)"); continue
    node_ids = BasinCache(db).load(device="cpu")["node_ids"]
    c = duckdb.connect(db, read_only=True)
    stations = c.execute("SELECT station_id, node_idx FROM stations").fetchdf()
    obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
    c.close()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{pq}' WHERE date>='{T0}' AND date<='{T1}'").df()
    sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()
    k_me, k_hy = [], []
    for _, st in stations.iterrows():
        o = obs[obs.station_id == st.station_id].set_index("date")["discharge"]
        if o.notna().sum() < 60: continue
        # méandre
        s = sim[sim.reach_id == int(st.node_idx) + 1].set_index("date")["Q_sim_m3s"]
        j = pd.concat([s, o], axis=1, join="inner").dropna()
        km = kge(j.iloc[:, 0].values, j.iloc[:, 1].values) if len(j) else np.nan
        # hydrotel brut (troncon_id du node : node_ids sont des ints -> "REG#####")
        tid = f"{reg}{int(node_ids[int(st.node_idx)]):05d}"
        kh = np.nan
        if tid in z_pos:
            dis = pd.Series(z["Dis"][z_pos[tid], z_sl].values, index=z_dates)
            j2 = pd.concat([dis, o], axis=1, join="inner").dropna()
            kh = kge(j2.iloc[:, 0].values, j2.iloc[:, 1].values) if len(j2) else np.nan
        if np.isfinite(km) or np.isfinite(kh):
            k_me.append(km); k_hy.append(kh)
    k_me = np.array(k_me, float); k_hy = np.array(k_hy, float)
    both = np.isfinite(k_me) & np.isfinite(k_hy)
    rows.append({"region": reg, "n_sta": int(both.sum()),
                 "meandre_med": np.nanmedian(k_me[both]) if both.any() else np.nan,
                 "hydrotel_med": np.nanmedian(k_hy[both]) if both.any() else np.nan,
                 "meandre_gagne": int((k_me[both] > k_hy[both]).sum()) if both.any() else 0})
    print(f"{reg}: n={rows[-1]['n_sta']} | méandre {rows[-1]['meandre_med']:.3f} vs Hydrotel {rows[-1]['hydrotel_med']:.3f} "
          f"| méandre gagne {rows[-1]['meandre_gagne']}/{rows[-1]['n_sta']}")
z.close()
df = pd.DataFrame(rows)
if len(df):
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/quebec_heldout.csv", index=False)
    b = df.dropna()
    print(f"\nGLOBAL ({b.n_sta.sum()} stations, {len(b)} régions) : "
          f"méandre méd-des-méd {b.meandre_med.median():.3f} vs Hydrotel {b.hydrotel_med.median():.3f} "
          f"| stations gagnées {b.meandre_gagne.sum()}/{b.n_sta.sum()}")
    print("-> reports/quebec_heldout.csv")
