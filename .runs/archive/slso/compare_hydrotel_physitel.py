"""Validation PHYSITEL : obs vs méandre vs Hydrotel (débit BRUT DEBITS_SIM, avant
interpolation optimale) aux jauges SLSO du bassin PHYSITEL. Apparie les jauges
aux tronçons Hydrotel par aire+proximité. Mesure ENFIN proprement l'écart
méandre/Hydrotel sur le même bassin, les mêmes jauges, l'observé commun.

  python .runs/slso/compare_hydrotel_physitel.py [reach_parquet]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, duckdb, xarray as xr
import os
os.chdir(Path(__file__).resolve().parents[2])

REACH = sys.argv[1] if len(sys.argv) > 1 else ".runs/slso/results/reach-phase1-grace.parquet"
DB = ".runs/slso/data/slso.duckdb"
PLAT = "LN24HA"   # SLSO est dans la plateforme LN24HA
NC = f"Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_{PLAT}_GCQ_HC_04032025.nc"
D0, D1 = "2001-01-01", "2024-10-30"

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = ~np.isnan(o) & ~np.isnan(s); o, s = o[m], s[m]
    if len(o) < 100 or o.std() == 0 or s.std() == 0: return np.nan
    r = np.corrcoef(o, s)[0, 1]; b = s.mean()/o.mean()
    g = (s.std()/s.mean())/(o.std()/o.mean())
    return 1 - np.sqrt((r-1)**2 + (b-1)**2 + (g-1)**2)

def peak_ratio(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = ~np.isnan(o) & ~np.isnan(s)
    if m.sum() < 100: return np.nan
    o, s = o[m], s[m]
    hi = o >= np.quantile(o, 0.99)
    if hi.sum() < 5: return np.nan
    return s[hi].mean() / (o[hi].mean() + 1e-9)

# 1. Jauges + obs + mapping node_idx -> node_id (reach_id du parquet)
con = duckdb.connect(DB, read_only=True)
sta = con.execute("select station_id sid, node_idx, lon, lat, drainage_area_km2 area from stations").df()
nmap = con.execute("select node_idx, node_id from nodes").df()
obs = con.execute(f"select station_id sid, date, discharge q from observations where date between '{D0}' and '{D1}'").df()
con.close()
sta = sta.merge(nmap, on="node_idx", how="left")
obs['date'] = pd.to_datetime(obs['date']).dt.normalize()

# 2. méandre Q par reach_id (= node_id)
mreach = pd.read_parquet(REACH, columns=['date', 'reach_id', 'Q_sim_m3s'])
mreach['date'] = pd.to_datetime(mreach['date']).dt.normalize()

# 3. Hydrotel : apparier jauges -> tronçons (aire + proximité)
ds = xr.open_dataset(NC)
hlon = ds['lon'].values; hlat = ds['lat'].values; harea = ds['drainage_area'].values
cos = np.cos(np.radians(np.nanmedian(hlat)))
print(f"Plateforme {PLAT} : {len(hlon)} tronçons. Appariement de {sta.sid.nunique()} jauges SLSO...", flush=True)
match = {}
for r in sta.itertuples():
    if not (r.area and r.area > 0): continue
    dx = (hlon - r.lon) * 111.0 * cos; dy = (hlat - r.lat) * 111.0
    dist = np.hypot(dx, dy); ratio = harea / max(float(r.area), 1e-3)
    valid = (ratio < 2.0) & (ratio > 0.5) & (dist <= 5.0)
    if not valid.any(): continue
    cost = np.where(valid, dist + 50*np.abs(np.log(np.clip(ratio, 1e-6, 1e6))), np.inf)
    match[r.sid] = int(np.argmin(cost))
print(f"  appariés Hydrotel : {len(match)}/{sta.sid.nunique()}", flush=True)

htime = pd.to_datetime(ds['time'].values)
tmask = (htime >= D0) & (htime <= D1)
hidx = sorted(set(match.values()))
print("  extraction Dis (lent, réseau)...", flush=True)
Dis = ds['Dis'].isel(station=hidx, time=np.where(tmask)[0]).values
ds.close()
htimes = pd.DatetimeIndex(htime[tmask]).normalize()
ipos = {gi: i for i, gi in enumerate(hidx)}
hydf = {sid: pd.Series(Dis[ipos[gi]], index=htimes) for sid, gi in match.items()}

# 4. comparaison par station
rows = []
for r in sta.drop_duplicates('sid').itertuples():
    sid = r.sid
    o = obs[obs.sid == sid].set_index('date')['q']
    if len(o) < 365: continue
    ms = mreach[mreach.reach_id == r.node_id].set_index('date')['Q_sim_m3s']
    df = pd.DataFrame({'o': o}); df['m'] = ms.reindex(df.index)
    km = kge(df.o, df.m); pm = peak_ratio(df.o, df.m)
    kh = ph = np.nan
    if sid in hydf:
        df['h'] = hydf[sid].reindex(df.index)
        kh = kge(df.o, df.h); ph = peak_ratio(df.o, df.h)
    rows.append((sid, r.area, km, kh, pm, ph))

res = pd.DataFrame(rows, columns=['sid', 'area', 'KGE_meandre', 'KGE_hydrotel', 'peak_meandre', 'peak_hydrotel'])
res = res.dropna(subset=['KGE_hydrotel', 'KGE_meandre']).sort_values('area', ascending=False)
pd.set_option('display.width', 170)
print(f"\n=== {len(res)} stations SLSO (obs vs meandre vs Hydrotel {PLAT}, {D0}..{D1}) ===")
print(f"meandre = {Path(REACH).name}")
print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\n-- Medianes per-station --")
print(f"KGE   meandre {res.KGE_meandre.median():.3f}  |  Hydrotel {res.KGE_hydrotel.median():.3f}  |  ecart {(res.KGE_hydrotel-res.KGE_meandre).median():+.3f}")
print(f"peak  meandre {res.peak_meandre.median():.3f}  |  Hydrotel {res.peak_hydrotel.median():.3f}  |  ecart {(res.peak_hydrotel-res.peak_meandre).median():+.3f}")
print(f"Hydrotel > meandre (KGE) : {int((res.KGE_hydrotel>res.KGE_meandre).sum())}/{len(res)} stations")
