"""Bilan d'eau MODÈLE-LIBRE : les trois produits indépendants (CaSR P, MODIS ET,
débits mesurés) ferment-ils P − ET − Q ≈ 0 ? Un résidu important = données
incompatibles, le modèle est sommé de réconcilier l'irréconciliable. Par station
sur son aire de drainage, période commune MODIS (2013-2024), moyennes long terme.
  python .runs/slso/diag_closure.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, duckdb

Y0, Y1 = "2013-01-01", "2021-12-31"   # commun MODIS, hors dérive test 2022-24
DB = ".runs/slso/data/slso.duckdb"
c = duckdb.connect(DB, read_only=True)
st = c.execute("SELECT station_id, node_idx, drainage_area_km2 FROM stations").fetchdf()
# ET MODIS moyen par noeud (mm/j -> mm/an), qualité OK, période commune
et = c.execute(f"""SELECT node_idx, AVG(etr_mm_day)*365.25 AS et_yr FROM modis_et
                   WHERE quality_ok AND date>='{Y0}' AND date<='{Y1}' GROUP BY node_idx""").fetchdf()
et_map = dict(zip(et.node_idx, et.et_yr))
# Q moyen par station (m3/s -> mm/an sur son aire)
obs = c.execute(f"""SELECT station_id, AVG(discharge) AS q FROM observations
                    WHERE date>='{Y0}' AND date<='{Y1}' GROUP BY station_id""").fetchdf()
q_map = dict(zip(obs.station_id, obs.q))
c.close()

# P CaSR moyen par noeud (mm/an) sur la période
ds = xr.open_dataset(".runs/slso/data/forcing-casr-riox.nc")
t = pd.to_datetime(ds["time"].values); sl = (t >= pd.Timestamp(Y0)) & (t <= pd.Timestamp(Y1))
P_node = ds["forcing"].values[sl][..., 0].mean(axis=0) * 365.25   # (n,) mm/an
ds.close()

rows = []
for _, s in st.iterrows():
    ni = int(s.node_idx); sid = s.station_id; area = s.drainage_area_km2
    if sid not in q_map or area is None or area <= 0:
        continue
    P = float(P_node[ni]); ET = et_map.get(ni, np.nan)
    Q = q_map[sid] * 31557.6 / area          # mm/an
    if not np.isfinite(ET):
        continue
    resid = P - ET - Q                        # devrait être ~0 (ΔS long terme ~0)
    rows.append((sid, area, P, ET, Q, resid, resid / P * 100, Q / P, ET / P))
df = pd.DataFrame(rows, columns=["station", "area_km2", "P", "ET", "Q", "resid", "resid_pct", "RC", "ET_frac"])
df = df.sort_values("area_km2", ascending=False)
pd.set_option("display.width", 200)
print(f"Bilan modèle-libre {Y0[:4]}-{Y1[:4]} (mm/an) — {len(df)} stations\n")
print(df.round(0).to_string(index=False))
print(f"\n— Médianes — P {df.P.median():.0f}  ET {df.ET.median():.0f}  Q {df.Q.median():.0f}  "
      f"résidu {df.resid.median():.0f} ({df.resid_pct.median():.0f}% de P)")
print(f"  coeff ruissellement Q/P médian {df.RC.median():.2f}  |  ET/P médian {df.ET_frac.median():.2f}")
print(f"  ET/P + Q/P médian = {df.ET_frac.median()+df.RC.median():.2f}  (1.0 = fermeture parfaite)")
print(f"\nLecture : résidu>0 => P > ET+Q (CaSR trop humide OU ET/Q sous-estimés) ;")
print(f"          résidu<0 => P < ET+Q (CaSR trop sec OU ET/Q surestimés).")
