"""Écarts de chaque membre PyGMET à la moyenne des membres, pour recentrer l'ensemble sur CaSR.

Les membres PyGMET d'un territoire (forcing-<terr>-ensNNN.nc, krigeage des stations) ont leur propre
climat : sur l'Outaouais en 2021-2024, 11 % plus secs et 1,3 °C plus froids que CaSR. Le modèle a
appris sur CaSR ; évalué sur ces membres, il mesurerait leur biais et non leur dispersion. On ne garde
donc de chaque membre que son écart à la moyenne des membres : un facteur multiplicatif sur la
précipitation, une différence sur les températures. Ces écarts, appliqués à CaSR par
member_forcing.py, donnent des membres centrés sur CaSR avec la dispersion spatio-temporelle de
PyGMET.

Le facteur de précipitation est borné entre 0,2 et 5, et vaut 1 quand la moyenne des membres est
sous 0,1 mm (pas de pluie à répartir). Un fichier par membre, sur la période d'ensemble seulement.

Usage : python member_anomalies.py <territoire> <dossier de sortie> <début> <fin> <n membres>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

territory, out_dir, start, end, n_members = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
out_dir.mkdir(parents=True, exist_ok=True)
data = "D:/meandre-data/quebec"

members = []
for m in range(1, n_members + 1):
    ds = xr.open_dataset(f"{data}/forcing-{territory}-ens{m:03d}.nc")
    days = pd.DatetimeIndex(ds.time.values)
    kept = (days >= start) & (days <= end)
    names = list(ds["var"].values)
    f = ds.forcing.values[kept][:, :, [names.index("P"), names.index("Tmin"), names.index("Tmax")]].astype(np.float32)
    ds.close()
    members.append(f)
members = np.stack(members)
mean = members.mean(axis=0)
print(f"{n_members} membres, {members.shape[1]} jours, {members.shape[2]} tronçons ; NaN : {np.isnan(members).mean() * 100:.3f} %")

for m in range(n_members):
    factor = np.where(mean[..., 0] >= 0.1, members[m, ..., 0] / np.maximum(mean[..., 0], 1e-6), 1.0)
    factor = np.clip(np.nan_to_num(factor, nan=1.0), 0.2, 5.0).astype(np.float32)
    delta_tmin = np.nan_to_num(members[m, ..., 1] - mean[..., 1]).astype(np.float32)
    delta_tmax = np.nan_to_num(members[m, ..., 2] - mean[..., 2]).astype(np.float32)
    xr.Dataset(
        {"p_factor": (("time", "node"), factor), "tmin_delta": (("time", "node"), delta_tmin), "tmax_delta": (("time", "node"), delta_tmax)},
        coords={"time": days[kept], "node": np.arange(members.shape[2])},
    ).to_netcdf(out_dir / f"anomalies-{territory}-ens{m + 1:03d}.nc", encoding={k: {"zlib": True, "complevel": 4} for k in ("p_factor", "tmin_delta", "tmax_delta")})
    print(f"membre {m + 1} : facteur P médian les jours de pluie {np.median(factor[mean[..., 0] >= 1]):.2f}, écart-type {factor[mean[..., 0] >= 1].std():.2f} ; dTmin écart-type {delta_tmin.std():.2f} °C")
