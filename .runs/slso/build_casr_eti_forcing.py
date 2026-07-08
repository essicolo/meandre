"""Forçage ETI : canaux du champion CaSR-corr (P corrigé volume+timing) + FB (courte
longueur d'onde incidente, W/m2, moyenne journalière) au canal 6 pour melt_mode='eti'.
hydrotel_column attend sw_in = canal 6 quand melt_mode='eti'.
Sortie : forcing-casr-eti.nc (7 canaux : P, Tmin, Tmax, R_n, u2, e_a, FB).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache

CASR = ".runs/slso/data/casr"
BASE = os.environ.get("BASE", "D:/meandre-data/slso/forcing-casr-corr.nc")
OUT = ".runs/slso/data/forcing-casr-eti.nc"
RLON_BLOCKS = ["rlon526-560", "rlon561-595"]; RLAT_BLOCKS = ["rlat351-385", "rlat386-420"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]

h = BasinCache(".runs/slso/data/slso.duckdb").load(device="cpu"); nc = h["node_coords"].numpy()
_g = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_TT_1.5m_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}_2000-2003.nc")["rotated_pole"].attrs
rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0))})
geo = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
nrlon, nrlat = Transformer.from_crs(geo, rp, always_xy=True).transform(nc[:, 0], nc[:, 1])
nrlon_da = xr.DataArray(nrlon, dims="node"); nrlat_da = xr.DataArray(nrlat, dims="node")

daily = []
for ch in CHUNKS:
    times = None; rows = []
    for ab in RLAT_BLOCKS:
        cols = []
        for rb in RLON_BLOCKS:
            ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_P_FB_SFC_{rb}_{ab}_{ch}.nc")
            v = [x for x in ds.data_vars if "CaSR" in x][0]
            cols.append(ds[v].values); times = pd.to_datetime(ds.time.values); ds.close()
        rows.append(np.concatenate(cols, axis=2))
    merged = np.concatenate(rows, axis=1)
    _d = xr.open_dataset(f"{CASR}/CaSR_v3.2_P_FB_SFC_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}_{ch}.nc")
    rlon0 = _d.rlon.values; rlat0 = _d.rlat.values; _d.close()
    _d2 = xr.open_dataset(f"{CASR}/CaSR_v3.2_P_FB_SFC_{RLON_BLOCKS[1]}_{RLAT_BLOCKS[0]}_{ch}.nc"); rlon1 = _d2.rlon.values; _d2.close()
    _d3 = xr.open_dataset(f"{CASR}/CaSR_v3.2_P_FB_SFC_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[1]}_{ch}.nc"); rlat1 = _d3.rlat.values; _d3.close()
    da = xr.DataArray(merged, dims=("time", "rlat", "rlon"),
                      coords={"time": times, "rlat": np.concatenate([rlat0, rlat1]), "rlon": np.concatenate([rlon0, rlon1])})
    samp = da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear")
    daily.append(pd.DataFrame(samp.values, index=times).resample("1D").mean())
fb = pd.concat(daily); fb = fb.groupby(fb.index).mean()
fb = fb[(fb.index >= pd.Timestamp("2000-01-01")) & (fb.index <= pd.Timestamp("2024-12-31"))]
print(f"FB : {fb.values.mean():.0f} W/m2 moyen | min {np.nanmin(fb.values):.0f} max {np.nanmax(fb.values):.0f}")

b = xr.open_dataset(BASE); F = b["forcing"].values; VARS = list(b["var"].values.astype(str)); t = b["time"].values; b.close()
assert F.shape[0] == fb.shape[0], f"{F.shape} vs {fb.shape}"
F2 = F.copy(); F2[:, :, 6] = fb.values.astype(np.float32)  # remplace DT_eff par FB
VARS2 = VARS[:6] + ["FB"]
if os.path.exists(OUT): os.remove(OUT)
xr.Dataset({"forcing": (("time", "node", "var"), F2.astype(np.float32))},
           coords={"time": t, "node": np.arange(F2.shape[1]), "var": VARS2}).to_netcdf(OUT)
print(f"[ok] {OUT} : {VARS2}")
