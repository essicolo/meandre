"""Formate les entrées PyGMET pour UNE région (design : CaSR+PyGMET reproductible).
Depuis les stations ECCC (cache eccc_loader) + le MNT PHYSITEL (altitude.tif) :
  - liste stations CSV : stnid,lat,lon,elev,slp_n,slp_e  (prédicteurs statiques)
  - un netcdf de séries par station (prcp/tmin/tmax, dim time)
  - gridinfo netcdf : grille régulière (GRID_DEG) avec elev + gradients + lat/lon 2D + mask
  - config PyGMET .toml
Le krigeage lui-même = PyGMET (LOO cross-val = juge anti-yeux-de-bœuf).

  python .runs/quebec/build_pygmet_inputs.py gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr, rioxarray
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from meandre.data.basin_cache import BasinCache
from meandre.data.eccc_loader import fetch_stations, fetch_daily

REG = sys.argv[1].lower()
Y0, Y1 = int(os.environ.get("Y0", 2000)), int(os.environ.get("Y1", 2024))
GRID_DEG = float(os.environ.get("GRID_DEG", 0.05))
MIN_DAYS = int(os.environ.get("MIN_DAYS", 730))
DEM = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020/physitel/altitude.tif"
OUT = Path(f"D:/meandre-data/pygmet/{REG}")
(OUT / "stndata").mkdir(parents=True, exist_ok=True)
DB = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"D:/meandre-data/quebec/{REG}.duckdb"

nc = BasinCache(DB).load(device="cpu")["node_coords"].numpy()
bbox = (round(float(nc[:, 0].min()) - 0.3, 1), round(float(nc[:, 1].min()) - 0.3, 1),
        round(float(nc[:, 0].max()) + 0.3, 1), round(float(nc[:, 1].max()) + 0.3, 1))
print(f"[{REG}] bbox {bbox} | grille {GRID_DEG}°", flush=True)

# ── MNT -> lat/lon régulier (élévation + gradients pour prédicteurs) ──────────
dem = rioxarray.open_rasterio(DEM).squeeze("band", drop=True)
dem = dem.where(dem > -900)
lon_g = np.arange(bbox[0], bbox[2] + GRID_DEG, GRID_DEG)
lat_g = np.arange(bbox[1], bbox[3] + GRID_DEG, GRID_DEG)
demll = dem.rio.reproject("EPSG:4326")
demll = demll.where(demll > -900)
xi = demll["x"].values; yi = demll["y"].values
order_y = np.argsort(yi)
fE = RegularGridInterpolator((yi[order_y], xi), demll.values[order_y], bounds_error=False, fill_value=np.nan)
LON, LAT = np.meshgrid(lon_g, lat_g)
elev = fE(np.c_[LAT.ravel(), LON.ravel()]).reshape(LAT.shape)
# gradients (m par degré) -> normalisés en m/km approx pour rester O(1-10) comme le cas-test
dz_dy, dz_dx = np.gradient(np.nan_to_num(elev, nan=np.nanmean(elev)), lat_g, lon_g)
grad_ns = dz_dy / 111.0   # m/km sud->nord
grad_we = dz_dx / (111.0 * np.cos(np.radians(LAT)))   # m/km ouest->est
mask = (~np.isnan(elev)).astype(np.int32)
elev = np.nan_to_num(elev, nan=0.0)

grid = xr.Dataset(
    {"elev": (("y", "x"), elev.astype(np.float32)),
     "gradient_n_s": (("y", "x"), grad_ns.astype(np.float32)),
     "gradient_w_e": (("y", "x"), grad_we.astype(np.float32)),
     "latitude": (("y", "x"), LAT.astype(np.float32)),
     "longitude": (("y", "x"), LON.astype(np.float32)),
     "mask": (("y", "x"), mask)},
    coords={"x": lon_g.astype(np.float32), "y": lat_g.astype(np.float32)})
grid["dx"] = GRID_DEG; grid["dy"] = GRID_DEG
grid["startx"] = float(lon_g[0]); grid["starty"] = float(lat_g[0])
grid_fp = OUT / f"{REG}.gridinfo.nc"
grid.to_netcdf(grid_fp)
print(f"[{REG}] gridinfo {LAT.shape} | {int(mask.sum())} cellules valides -> {grid_fp}", flush=True)

# ── stations ECCC -> liste + fichiers de séries ──────────────────────────────
df = fetch_daily(bbox, Y0, Y1)
fgrad_ns = RegularGridInterpolator((lat_g, lon_g), grad_ns, bounds_error=False, fill_value=0.0)
fgrad_we = RegularGridInterpolator((lat_g, lon_g), grad_we, bounds_error=False, fill_value=0.0)
felev = RegularGridInterpolator((lat_g, lon_g), elev, bounds_error=False, fill_value=0.0)
dates = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31", freq="D")
rows = []
for sid, g in df.groupby("climate_id"):
    g = g.dropna(subset=["date"]).drop_duplicates("date").set_index("date").reindex(dates)
    nval = int(g["P"].notna().sum() + g["Tmax"].notna().sum())
    if g["P"].notna().sum() < MIN_DAYS and g["Tmax"].notna().sum() < MIN_DAYS:
        continue
    lat = float(np.nanmedian(g["lat"])); lon = float(np.nanmedian(g["lon"]))
    if not (bbox[1] <= lat <= bbox[3] and bbox[0] <= lon <= bbox[2]):
        continue
    st = xr.Dataset(
        {"prcp": ("time", g["P"].values.astype(np.float32)),
         "tmin": ("time", g["Tmin"].values.astype(np.float32)),
         "tmax": ("time", g["Tmax"].values.astype(np.float32))},
        coords={"time": dates})
    st.to_netcdf(OUT / "stndata" / f"{sid}.nc")
    rows.append({"stnid": sid, "lat": round(lat, 4), "lon": round(lon, 4),
                 "elev": round(float(felev([[lat, lon]])[0]), 1),
                 "slp_n": round(float(fgrad_ns([[lat, lon]])[0]), 3),
                 "slp_e": round(float(fgrad_we([[lat, lon]])[0]), 3)})
stn = pd.DataFrame(rows)
stn_fp = OUT / f"{REG}.stn_list.csv"
stn.to_csv(stn_fp, index=False)
print(f"[{REG}] {len(stn)} stations retenues (>= {MIN_DAYS} j) -> {stn_fp}", flush=True)
print(f"[{REG}] densité : 1 station / {int((bbox[2]-bbox[0])*111*(bbox[3]-bbox[1])*80/max(len(stn),1))} km²", flush=True)
print(f"[{REG}] PRÊT pour PyGMET. Config à générer : date {Y0}-01-01..{Y1}-12-31", flush=True)
