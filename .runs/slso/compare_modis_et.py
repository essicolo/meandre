"""Écart MODIS ETR (MOD16A3GF, ANNUEL) vs ETR simulé (fields nc).

La table modis_et est annuelle (date = 1er janvier, etr_mm_day = taux annuel
moyen). On compare donc la moyenne ANNUELLE de l'ETR simulé par nœud×année.
"""
import duckdb, numpy as np, pandas as pd, xarray as xr

FIELDS = ".runs/slso/results/fields-kendall-gal-v3-phase2-boxcox-nll.nc"

# ETR simulé (time, node) → moyenne annuelle par nœud
ds = xr.open_dataset(FIELDS)
sim_etr = ds["etr"].values  # (T, N) mm/day
dates = pd.to_datetime(xr.open_dataset(".runs/slso/data/forcing.nc")["time"].values)
ds.close()
years = dates.year.values
sim_annual = {}  # year -> (N,) moyenne annuelle
for y in np.unique(years):
    sim_annual[y] = np.nanmean(sim_etr[years == y, :], axis=0)

# MODIS annuel (date = 1er janvier de l'année)
con = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
mod = con.execute(
    "select date, node_idx, etr_mm_day from modis_et "
    "where quality_ok = true and etr_mm_day is not null"
).fetchdf()
con.close()
mod["date"] = pd.to_datetime(mod["date"])
mod = mod[(mod["date"].dt.month == 1) & (mod["date"].dt.day == 1)].copy()  # garde l'annuel
mod["year"] = mod["date"].dt.year
print(f"MODIS annuel valide : {len(mod)} (nœud×année)  années {mod['year'].min()}-{mod['year'].max()}  "
      f"nœuds {mod['node_idx'].nunique()}")

rows = []
for _, r_ in mod.iterrows():
    y, ni = int(r_["year"]), int(r_["node_idx"])
    if y in sim_annual and ni < sim_etr.shape[1]:
        rows.append((y, ni, float(r_["etr_mm_day"]), float(sim_annual[y][ni])))

df = pd.DataFrame(rows, columns=["year", "node", "modis", "sim"]).dropna()
bias = (df["sim"] - df["modis"]).mean()
rmse = np.sqrt(((df["sim"] - df["modis"]) ** 2).mean())
r = np.corrcoef(df["sim"], df["modis"])[0, 1]
print(f"\n{'='*64}\nETR SIMULÉ vs MODIS ANNUEL — n={len(df)} paires (nœud×année)\n{'='*64}")
print(f"biais (sim−modis) = {bias:+.3f} mm/j   RMSE = {rmse:.3f}   r = {r:.3f}")
print(f"moyennes : sim {df['sim'].mean():.3f}  modis {df['modis'].mean():.3f} mm/j")
print(f"  → sim/modis = {df['sim'].mean()/df['modis'].mean():.2f}\n")
yc = df.groupby("year").agg(sim=("sim", "mean"), modis=("modis", "mean"))
print("  année ", " ".join(f"{y:>6d}" for y in yc.index))
print("  sim   ", " ".join(f"{v:6.2f}" for v in yc['sim']))
print("  modis ", " ".join(f"{v:6.2f}" for v in yc['modis']))
