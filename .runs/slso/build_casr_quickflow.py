"""Quickflow d'intensité RÉELLE depuis l'HORAIRE de CaSR (test sous-journalier scalable).
Pour chaque jour/nœud : fraction de la pluie qui RUISSELLE par excès d'infiltration,
   frac = somme_h max(0, P_h - infil_cap) / somme_h P_h
calculée sur la vraie séquence horaire (pas le proxy DT_eff). Précalcul OFFLINE : le
modèle reste JOURNALIER, on injecte frac comme signal. Scalable (calcul une fois).
Sauve casr_quickflow_frac.npy (T, N) + dates. INFIL_CAP en mm/h (défaut 5).
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, xarray as xr
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache

CASR = ".runs/slso/data/casr"
DB = os.environ.get("CASR_DB", ".runs/slso/data/slso.duckdb")
INFIL_CAP = float(os.environ.get("INFIL_CAP", "5.0"))    # mm/h — capacité d'infiltration de surface
RLON_BLOCKS = ["rlon526-560", "rlon561-595"]
RLAT_BLOCKS = ["rlat351-385", "rlat386-420"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]

h = BasinCache(DB).load(device="cpu")
nc = h["node_coords"].numpy(); nlon, nlat = nc[:, 0], nc[:, 1]
print(f"noeuds : {len(nc)} | infil_cap {INFIL_CAP} mm/h")
_t0 = f"{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}"
_g = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_TT_1.5m_{_t0}_2000-2003.nc")["rotated_pole"].attrs
rp_crs = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0))})
geo_crs = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
nrlon, nrlat = Transformer.from_crs(geo_crs, rp_crs, always_xy=True).transform(nlon, nlat)
nrlon_da = xr.DataArray(nrlon, dims="node"); nrlat_da = xr.DataArray(nrlat, dims="node")

daily = []
for ch in CHUNKS:
    times = None; rows = []
    for ab in RLAT_BLOCKS:
        cols = []
        for rb in RLON_BLOCKS:
            ds = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{rb}_{ab}_{ch}.nc")
            vname = [x for x in ds.data_vars if "CaSR" in x][0]
            cols.append(ds[vname].values); times = pd.to_datetime(ds.time.values); ds.close()
        rows.append(np.concatenate(cols, axis=2))
    merged = np.concatenate(rows, axis=1)
    da = xr.DataArray(merged, dims=("time", "rlat", "rlon"),
                      coords={"time": times, "rlat": np.arange(merged.shape[1]),
                              "rlon": np.arange(merged.shape[2])})
    # refaire les axes réguliers pour l'interp
    _dsr = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[0]}_{ch}.nc")
    rlon0 = _dsr.rlon.values; rlat0 = _dsr.rlat.values; _dsr.close()
    _dsr2 = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{RLON_BLOCKS[1]}_{RLAT_BLOCKS[0]}_{ch}.nc")
    rlon1 = _dsr2.rlon.values; _dsr2.close()
    _dsr3 = xr.open_dataset(f"{CASR}/CaSR_v3.2_A_PR0_SFC_{RLON_BLOCKS[0]}_{RLAT_BLOCKS[1]}_{ch}.nc")
    rlat1 = _dsr3.rlat.values; _dsr3.close()
    RLON = np.concatenate([rlon0, rlon1]); RLAT = np.concatenate([rlat0, rlat1])
    da = da.assign_coords(rlon=RLON, rlat=RLAT)
    samp = da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear")     # (time, node) m/h
    df = pd.DataFrame(samp.values * 1000.0, index=times)               # mm/h
    excess = (df - INFIL_CAP).clip(lower=0.0)                          # excès horaire
    dqf = excess.resample("1D").sum()                                  # quickflow journalier mm/j
    dsum = df.resample("1D").sum()                                     # précip journalière mm/j
    frac = (dqf / dsum.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(0.0)
    daily.append(frac)
    print(f"  chunk {ch} : {len(frac)} jours, frac médiane {float(frac.values[frac.values>0].mean()) if (frac.values>0).any() else 0:.3f}")

out = pd.concat(daily); out = out.groupby(out.index).max()             # jour-frontière
sl = (out.index >= pd.Timestamp("2000-01-01")) & (out.index <= pd.Timestamp("2024-12-31"))
out = out[sl]
np.save(".runs/slso/data/casr_quickflow_frac.npy", out.values.astype(np.float32))
np.save(".runs/slso/data/casr_quickflow_dates.npy", out.index.values.astype("datetime64[D]"))
print(f"[ok] fraction quickflow : {out.shape} | jours avec quickflow>0 : {(out.values>0.01).mean()*100:.0f}% "
      f"| frac moyenne (jours actifs) {out.values[out.values>0.01].mean():.3f}")
