"""Comparaison HEAD-TO-HEAD : Hydrotel vs meandre vs obs, par station, test 2022-2024.
Repond a "meandre est-il pire qu'Hydrotel, et dans quelle composante (r/beta/gamma) ?"
Hydrotel = sortie officielle LN24HA (Z:). Matche chaque jauge CEHQ au troncon Hydrotel
par coords + aire. ZERO run.

  python .runs/slso/diag_vs_hydrotel.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import duckdb
import xarray as xr

DB = ".runs/slso/data/slso.duckdb"
PARQUET = ".runs/slso/results/reach-physitel-hydrotel-overnight.parquet"
HYDRO = "Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_LN24HA_GCQ_HC_04032025.nc"
T0, T1 = "2022-01-01", "2024-12-31"

def kge(qs, qo):
    m = np.isfinite(qs) & np.isfinite(qo)
    qs, qo = qs[m], qo[m]
    if len(qs) < 30 or qo.std() < 1e-9 or qs.std() < 1e-9:
        return (np.nan,)*4
    r = np.corrcoef(qs, qo)[0, 1]
    beta = qs.mean()/qo.mean()
    gamma = (qs.std()/qs.mean())/(qo.std()/qo.mean())
    return 1-np.sqrt((r-1)**2+(beta-1)**2+(gamma-1)**2), r, beta, gamma

c = duckdb.connect(DB, read_only=True)
st = c.execute("SELECT station_id, node_idx, lon, lat, drainage_area_km2 FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id, date, discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()

# meandre sim
sim = duckdb.sql(f"SELECT date, reach_id, Q_sim_m3s FROM '{PARQUET}' WHERE date>='{T0}' AND date<='{T1}'").df()
sim["date"] = pd.to_datetime(sim["date"]).dt.normalize()

# Hydrotel : sous-ensemble SLSO + fenetre, charge seulement ce qu'il faut
ds = xr.open_dataset(HYDRO)
sid_h = ds["station_id"].values.astype(str)
is_slso = np.char.startswith(sid_h, "SLSO")
idx_slso = np.where(is_slso)[0]
lon_h = ds["lon"].values[idx_slso]; lat_h = ds["lat"].values[idx_slso]
area_h = ds["drainage_area"].values[idx_slso]
dis = ds["Dis"].isel(station=idx_slso).sel(time=slice(T0, T1))
th = pd.to_datetime(dis["time"].values).normalize()
dis_v = dis.values   # (n_slso, n_time)
ds.close()
print(f"troncons SLSO Hydrotel: {len(idx_slso)}  fenetre {th.min().date()}..{th.max().date()} ({len(th)} j)")

rows = []
for _, s in st.iterrows():
    sid, nidx = s["station_id"], int(s["node_idx"])
    o = obs[obs["station_id"] == sid][["date", "discharge"]]
    if len(o) < 30:
        continue
    # match Hydrotel : plus proche en coords, aire compatible (<35%)
    d2 = (lon_h - s["lon"])**2 + (lat_h - s["lat"])**2
    arerr = np.abs(area_h - s["drainage_area_km2"]) / max(s["drainage_area_km2"], 1)
    cand = np.where(arerr < 0.35)[0]
    if len(cand) == 0:
        j = int(np.argmin(d2)); aflag = "*"   # pas de match d'aire : plus proche coords
    else:
        j = cand[int(np.argmin(d2[cand]))]; aflag = ""
    qh = pd.DataFrame({"date": th, "qh": dis_v[j]})
    qm = sim[sim["reach_id"] == nidx + 1][["date", "Q_sim_m3s"]].rename(columns={"Q_sim_m3s": "qm"})
    m = o.merge(qh, on="date", how="inner").merge(qm, on="date", how="inner")
    if len(m) < 30:
        continue
    kh, rh, bh, gh = kge(m["qh"].to_numpy(), m["discharge"].to_numpy())
    km, rm, bm, gm = kge(m["qm"].to_numpy(), m["discharge"].to_numpy())
    rows.append((sid, s["drainage_area_km2"], area_h[j], aflag, kh, rh, bh, gh, km, rm, bm, gm))

df = pd.DataFrame(rows, columns=["station","area_cehq","area_hyd","flag",
                                 "KGE_H","r_H","b_H","g_H","KGE_M","r_M","b_M","g_M"]).dropna(subset=["KGE_H","KGE_M"])
df = df.sort_values("KGE_M")
pd.set_option("display.float_format", lambda x: f"{x:6.2f}")
pd.set_option("display.width", 200)
print(f"\n=== Hydrotel (H) vs meandre (M), test {T0}..{T1}, {len(df)} stations ===")
print(df.to_string(index=False))
print(f"\n           Hydrotel   meandre")
print(f"KGE   med  {df.KGE_H.median():7.3f}  {df.KGE_M.median():7.3f}")
print(f"r     med  {df.r_H.median():7.3f}  {df.r_M.median():7.3f}")
print(f"beta  med  {df.b_H.median():7.3f}  {df.b_M.median():7.3f}")
print(f"gamma med  {df.g_H.median():7.3f}  {df.g_M.median():7.3f}")
print(f"\nstations ou meandre >= Hydrotel : {(df.KGE_M >= df.KGE_H).sum()}/{len(df)}")
print(f"flag * = pas de match d'aire fiable ({(df.flag=='*').sum()} stations)")
