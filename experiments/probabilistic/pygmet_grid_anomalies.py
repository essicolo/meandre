"""Écarts de chaque membre PyGMET à la moyenne des membres, lus directement dans les sorties de grille
de PyGMET (ensemble_outputs), au point de chaque tronçon d'un territoire.

Même rôle que member_anomalies.py, sans passer par des forçages par membre : chaque tronçon prend la
cellule valide la plus proche de la grille PyGMET (0,05 degré), et pour chaque membre on garde un
facteur multiplicatif sur la précipitation et une différence sur Tmin et Tmax, calculés sur la
moyenne des membres. member_forcing.py applique ensuite ces écarts à CaSR.

Usage : python pygmet_grid_anomalies.py <territoire> <nom de cas PyGMET> <dossier de sortie>
"""
import glob
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

from meandre.data.basin_cache import BasinCache

territory, case, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
data = "D:/meandre-data"
files = sorted(glob.glob(f"{data}/pygmet/{territory}/PyGMET_output/ensemble_outputs/{case}_Ensemble_*.nc"))
assert files, "aucun membre PyGMET"

nodes = BasinCache(f"{data}/quebec/{territory}.duckdb").load(device="cpu")["node_coords"].numpy()
first = xr.open_dataset(files[0])
lon_grid, lat_grid = first["longitude"].values, first["latitude"].values
valid = np.isfinite(first["prcp"].isel(time=0).values)
first.close()
# Cellule valide la plus proche du point de chaque tronçon.
cells = np.flatnonzero(valid.ravel())
tree = cKDTree(np.c_[lon_grid.ravel()[cells], lat_grid.ravel()[cells]])
_, nearest = tree.query(nodes[:, :2])
cell = cells[nearest]
print(f"{len(files)} membres, {len(nodes)} tronçons, {len(cells)} cellules valides, distance médiane au point {np.median(_):.3f} degré")

members = []
for f in files:
    ds = xr.open_dataset(f)
    p = ds["prcp"].values.reshape(ds.sizes["time"], -1)[:, cell]
    tmean = ds["tmean"].values.reshape(ds.sizes["time"], -1)[:, cell]
    trange = ds["trange"].values.reshape(ds.sizes["time"], -1)[:, cell]
    times = ds["time"].values
    ds.close()
    members.append(np.stack([p, tmean - trange / 2, tmean + trange / 2], axis=-1))
members = np.stack(members).astype(np.float32)
mean = np.nanmean(members, axis=0)
print(f"précipitation annuelle moyenne des membres {np.nanmean(mean[..., 0]) * 365.25:.0f} mm, NaN {np.isnan(members).mean() * 100:.2f} %")

for m in range(len(files)):
    factor = np.where(mean[..., 0] >= 0.1, members[m, ..., 0] / np.maximum(mean[..., 0], 1e-6), 1.0)
    factor = np.clip(np.nan_to_num(factor, nan=1.0), 0.2, 5.0).astype(np.float32)
    delta_tmin = np.nan_to_num(members[m, ..., 1] - mean[..., 1]).astype(np.float32)
    delta_tmax = np.nan_to_num(members[m, ..., 2] - mean[..., 2]).astype(np.float32)
    xr.Dataset(
        {"p_factor": (("time", "node"), factor), "tmin_delta": (("time", "node"), delta_tmin), "tmax_delta": (("time", "node"), delta_tmax)},
        coords={"time": times, "node": np.arange(len(nodes))},
    ).to_netcdf(out_dir / f"anomalies-{territory}-ens{m + 1:03d}.nc", encoding={k: {"zlib": True, "complevel": 4} for k in ("p_factor", "tmin_delta", "tmax_delta")})
    rainy = mean[..., 0] >= 1
    print(f"membre {m + 1} : facteur P médian les jours de pluie {np.median(factor[rainy]):.2f}, écart-type {factor[rainy].std():.2f} ; dTmin écart-type {delta_tmin.std():.2f} °C")
