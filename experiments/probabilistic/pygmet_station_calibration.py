"""Calibration de l'ensemble PyGMET aux stations météo ECCC, sur les jours réellement mesurés.

Deux mesures, pour la précipitation et la température moyenne :
- écart réduit en validation croisée : (observé - estimation sans la station) / incertitude de
  PyGMET à la maille de la station. Calibré si son écart-type vaut 1 ;
- rang de l'observé parmi les membres à la maille de la station. La station a servi à construire les
  membres, donc ce rang est optimiste : un ensemble trop étroit même ainsi l'est à coup sûr.

Les jours comblés par interpolation IDW avant PyGMET sont écartés : seuls comptent les jours où la
station ECCC a une mesure dans le cache brut.

Usage : python pygmet_station_calibration.py <territoire> <nom de cas> <cache ECCC brut .parquet>
"""
import glob
import sys

import numpy as np
import pandas as pd
import xarray as xr

territory, case, raw_path = sys.argv[1:4]
root = f"D:/meandre-data/pygmet/{territory}/PyGMET_output"
stations = xr.open_dataset(f"{root}/stn_info/all_stn.nc")
ids = [s.decode() if isinstance(s, bytes) else str(s) for s in stations["stnid"].values]
cv = xr.open_dataset(sorted(glob.glob(f"{root}/regression_outputs/{case}_CrossValidation_Regression_*.nc"))[0])
aux = xr.open_dataset(sorted(glob.glob(f"{root}/regression_outputs/{case}_Auxiliary_*.nc"))[0])
days = pd.DatetimeIndex(cv["time"].values)

# Jours mesurés : présents et non manquants dans le cache ECCC brut.
raw = pd.read_parquet(raw_path)
raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
raw["climate_id"] = raw["climate_id"].astype(str)
raw = raw[raw["climate_id"].isin(ids) & raw["date"].isin(days)]

# Maille de chaque station : la plus proche en longitude et latitude.
lat_grid, lon_grid = aux["latitude"].values, aux["longitude"].values
cell = [np.unravel_index(np.nanargmin((lat_grid - la) ** 2 + (lon_grid - lo) ** 2), lat_grid.shape) for la, lo in zip(stations["lat"].values, stations["lon"].values)]
rows, cols = np.array([c[0] for c in cell]), np.array([c[1] for c in cell])

member_files = sorted(glob.glob(f"{root}/ensemble_outputs/{case}_Ensemble_*.nc"))
members = {"prcp": [], "tmean": []}
for f in member_files:
    ds = xr.open_dataset(f)
    for v in members:
        members[v].append(ds[v].transpose("time", "y", "x").values[:, rows, cols])
    ds.close()
members = {v: np.stack(a) for v, a in members.items()}

stn_index = {s: i for i, s in enumerate(ids)}
day_index = pd.Series(np.arange(len(days)), index=days)
measured = {"prcp": raw.dropna(subset=["P"]), "tmean": raw.dropna(subset=["Tmin", "Tmax"])}
print(f"{territory} : {len(ids)} stations, {len(member_files)} membres, {len(days)} jours")
for var, obs_column in (("prcp", "P"), ("tmean", None)):
    m = measured[var]
    s = m["climate_id"].map(stn_index).to_numpy()
    t = day_index[m["date"]].to_numpy()
    observed = m["P"].to_numpy() if obs_column else 0.5 * (m["Tmin"] + m["Tmax"]).to_numpy()
    estimate = cv[var].values[s, t]
    uncertainty = aux[f"uncert_{var}"].transpose("time", "y", "x").values[t, rows[s], cols[s]]
    keep = np.isfinite(observed) & np.isfinite(estimate) & np.isfinite(uncertainty) & (uncertainty > 0)
    if var == "prcp":
        keep &= (observed > 0.5) | (estimate > 0.5)
    reduced = (observed[keep] - estimate[keep]) / uncertainty[keep]
    ensemble = members[var][:, t[keep], s[keep]]
    rank = (ensemble < observed[keep]).sum(axis=0)
    counts = np.bincount(rank, minlength=len(member_files) + 1) / len(rank)
    label = "précipitation, jours où l'observé ou l'estimation dépasse 0,5 mm" if var == "prcp" else "température moyenne"
    print(f"\n{label} : {keep.sum()} jours-stations mesurés")
    print(f"  validation croisée : écart réduit moyen {reduced.mean():+.2f}, écart-type {reduced.std():.2f} (1 si calibré), part hors de ±1,64 : {100 * (np.abs(reduced) > 1.645).mean():.1f} % (10 attendus)")
    print(f"  rang parmi les membres (optimiste) : classes {np.round(100 * counts, 1)} | attendu {100 / len(counts):.1f} chacune")
