"""Le "je ne sais quoi" hydrologique de quebec.zarr = cohérence de VOLUME ?
Coefficient de ruissellement RC = Q_obs / P par station, pour CaSR vs quebec.zarr.
Plage physique QC boréal ~0.45-0.65. Si CaSR (plus humide) donne RC trop bas et
quebec.zarr un RC cohérent, CaSR a un excès de précip que le modèle doit "perdre"
(ET/stockage) -> beta distordu. Corrigeable par prétraitement hydrologique
(sous-captage, bilan). CPU seulement.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, duckdb, xarray as xr

T0, T1 = "2004-01-01", "2021-12-31"
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
st = c.execute("SELECT station_id, node_idx, drainage_area_km2 AS area FROM stations").fetchdf()
obs = c.execute(f"SELECT station_id,date,discharge FROM observations WHERE date>='{T0}' AND date<='{T1}'").fetchdf()
c.close()
obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
# débit moyen par station (m3/s) -> lame (mm/an) = Q * 31557.6 / area_km2
qbar = obs.groupby("station_id").discharge.mean()

def pbar_at_nodes(fc):
    d = xr.open_dataset(f".runs/slso/data/{fc}")
    t = pd.to_datetime(d["time"].values).normalize()
    sl = (t >= pd.Timestamp(T0)) & (t <= pd.Timestamp(T1))
    P = d["forcing"].values[sl][..., 0].mean(axis=0) * 365.25   # mm/an par nœud
    d.close(); return P

Pc = pbar_at_nodes("forcing-casr-riox-intens.nc")
Pq = pbar_at_nodes("forcing.nc")

rows = []
for _, s in st.iterrows():
    if s.station_id not in qbar.index: continue
    q_mm = qbar[s.station_id] * 31557.6 / s.area          # lame écoulée mm/an
    ni = int(s.node_idx)
    rows.append(dict(sid=s.station_id, area=s.area, q_mm=q_mm,
                     P_casr=Pc[ni], P_qz=Pq[ni],
                     RC_casr=q_mm / Pc[ni], RC_qz=q_mm / Pq[ni]))
df = pd.DataFrame(rows)
print(f"{len(df)} stations | période {T0[:4]}-{T1[:4]}\n")
print(f"lame écoulée Q médiane      : {df.q_mm.median():.0f} mm/an")
print(f"P médian  CaSR {df.P_casr.median():.0f}  |  quebec.zarr {df.P_qz.median():.0f} mm/an\n")
print(f"COEFF RUISSELLEMENT RC = Q/P (physique QC boréal ~0.45-0.65) :")
print(f"  CaSR         : médian {df.RC_casr.median():.3f}  [{df.RC_casr.quantile(.25):.2f}-{df.RC_casr.quantile(.75):.2f}]")
print(f"  quebec.zarr  : médian {df.RC_qz.median():.3f}  [{df.RC_qz.quantile(.25):.2f}-{df.RC_qz.quantile(.75):.2f}]")
print()
# combien de stations avec RC physiquement invraisemblable (<0.4 = trop humide, >0.8 = trop sec)
for tag, col in [("CaSR", "RC_casr"), ("quebec.zarr", "RC_qz")]:
    lo = (df[col] < 0.40).sum(); hi = (df[col] > 0.80).sum()
    print(f"  {tag:11s} : {lo} stations RC<0.40 (forçage trop humide) | {hi} RC>0.80 (trop sec)")
# excès de précip CaSR implicite pour ramener RC à ~0.55
target = 0.55
print(f"\nP requis pour RC={target} (médian) : {df.q_mm.median()/target:.0f} mm/an")
print(f"  -> CaSR à {df.P_casr.median():.0f} est {'TROP HUMIDE de '+str(int(df.P_casr.median()-df.q_mm.median()/target))+' mm/an' if df.P_casr.median()>df.q_mm.median()/target else 'cohérent'}")
print(f"  -> quebec.zarr à {df.P_qz.median():.0f} est {'plus proche' if abs(df.P_qz.median()-df.q_mm.median()/target)<abs(df.P_casr.median()-df.q_mm.median()/target) else 'plus loin'}")
