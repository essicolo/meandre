"""Comparaison par station : obs vs méandre vs Hydrotel (débit BRUT, avant
interpolation optimale). Apparie nos jauges SLSO aux tronçons Hydrotel par
aire+proximité, extrait le débit Hydrotel, calcule KGE/peak_ratio des deux
modèles contre l'observé. Trouve ce que Hydrotel capte et que méandre rate.

  uv run python .runs/slso-od/compare_hydrotel.py [PLATEFORME]   (défaut MG24HS)
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd, duckdb, xarray as xr
os.chdir(Path(__file__).resolve().parents[2])

PLAT = sys.argv[1] if len(sys.argv) > 1 else "MG24HS"
NC = f"Z:/Atlas_hydro/SRH/DEBITS_SIM/A20_HYDREP_QCMERI_XXX_DEBITJ_HIS_XXX_XXX_XXX_XXX_XXX_XXX_HYD_{PLAT}_GCQ_HC_04032025.nc"
D0, D1 = "2001-01-01", "2024-10-30"

def kge(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = ~np.isnan(o) & ~np.isnan(s)
    o, s = o[m], s[m]
    if len(o) < 100 or o.std() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    b = s.mean() / o.mean()
    g = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1 - np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

def peak_ratio(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = ~np.isnan(o) & ~np.isnan(s)
    if m.sum() < 100:
        return np.nan
    return np.nanmax(s[m]) / (np.nanmax(o[m]) + 1e-9)

# 1. Nos jauges + obs + node_idx
con = duckdb.connect('.runs/slso-od/data/basin.duckdb', read_only=True)
sta = con.execute("select station_id sid, node_idx, lon, lat, drainage_area_km2 area from stations").df()
obs = con.execute(f"select station_id sid, date, discharge q from observations where date between '{D0}' and '{D1}'").df()
con.close()
obs['date'] = pd.to_datetime(obs['date']).dt.normalize()

# 2. méandre Q par nœud (reach-posenc)
mreach = pd.read_parquet('.runs/slso-od/results/reach-posenc.parquet', columns=['date', 'reach_id', 'Q_sim_m3s'])
mreach['date'] = pd.to_datetime(mreach['date']).dt.normalize()

# 3. Hydrotel : lon/lat/aire de tous les tronçons, puis apparier
ds = xr.open_dataset(NC)
hlon = ds['lon'].values; hlat = ds['lat'].values; harea = ds['drainage_area'].values
cos = np.cos(np.radians(np.nanmedian(hlat)))
print(f"Plateforme {PLAT} : {len(hlon)} tronçons. Appariement de {sta.sid.nunique()} jauges...", flush=True)

# index Hydrotel pour chaque jauge (aire+proximité, filtres ratio<2 dist<5km)
match = {}
for r in sta.itertuples():
    if not (r.area and r.area > 0): continue
    dx = (hlon - r.lon) * 111.0 * cos; dy = (hlat - r.lat) * 111.0
    dist = np.hypot(dx, dy)
    ratio = harea / max(float(r.area), 1e-3)
    valid = (ratio < 2.0) & (ratio > 0.5) & (dist <= 5.0)
    if not valid.any(): continue
    cost = np.where(valid, dist + 50 * np.abs(np.log(np.clip(ratio, 1e-6, 1e6))), np.inf)
    match[r.sid] = int(np.argmin(cost))
print(f"  appariés : {len(match)}/{sta.sid.nunique()}", flush=True)

# extraire Hydrotel Dis pour les tronçons appariés + slice temps
htime = pd.to_datetime(ds['time'].values)
tmask = (htime >= D0) & (htime <= D1)
hidx = sorted(set(match.values()))
print("  extraction Dis (peut être lent sur réseau)...", flush=True)
Dis = ds['Dis'].isel(station=hidx, time=np.where(tmask)[0]).values  # (n_match, T)
ds.close()
htimes = pd.DatetimeIndex(htime[tmask]).normalize()
idx_pos = {gi: i for i, gi in enumerate(hidx)}
hydf = {sid: pd.Series(Dis[idx_pos[gi]], index=htimes) for sid, gi in match.items()}

# 4. comparaison par station
rows = []
for r in sta.drop_duplicates('sid').itertuples():
    sid = r.sid
    o = obs[obs.sid == sid].set_index('date')['q']
    if len(o) < 365: continue
    # méandre au nœud
    ms = mreach[mreach.reach_id == r.node_idx].set_index('date')['Q_sim_m3s']
    # aligner sur dates obs
    df = pd.DataFrame({'o': o})
    df['m'] = ms.reindex(df.index)
    kge_m = kge(df.o, df.m); pr_m = peak_ratio(df.o, df.m)
    kge_h = pr_h = np.nan
    if sid in hydf:
        df['h'] = hydf[sid].reindex(df.index)
        kge_h = kge(df.o, df.h); pr_h = peak_ratio(df.o, df.h)
    rows.append((sid, r.area, kge_m, kge_h, pr_m, pr_h))

res = pd.DataFrame(rows, columns=['sid', 'area', 'KGE_meandre', 'KGE_hydrotel', 'peak_meandre', 'peak_hydrotel'])
res = res.dropna(subset=['KGE_hydrotel', 'KGE_meandre']).sort_values('area', ascending=False)
res.to_parquet(f'.runs/slso-od/results/compare_hydrotel_{PLAT}.parquet', index=False)
pd.set_option('display.width', 160)
print(f"\n=== {len(res)} stations (obs vs meandre vs Hydrotel {PLAT}) ===")
print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print("\n-- Medianes --")
print(f"KGE   meandre {res.KGE_meandre.median():.3f}  |  Hydrotel {res.KGE_hydrotel.median():.3f}")
print(f"peak  meandre {res.peak_meandre.median():.3f}  |  Hydrotel {res.peak_hydrotel.median():.3f}")
print(f"Hydrotel > meandre (KGE) : {int((res.KGE_hydrotel>res.KGE_meandre).sum())}/{len(res)}")
print(f"ecart KGE median (Hydrotel - meandre) : {(res.KGE_hydrotel-res.KGE_meandre).median():+.3f}")
