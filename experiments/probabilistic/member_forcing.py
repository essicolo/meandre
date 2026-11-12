"""Forçage d'un membre d'ensemble recentré sur CaSR : CaSR (vent en m/s) partout, et sur la période
d'ensemble, la précipitation multipliée par le facteur du membre et les températures décalées de son
écart (écarts calculés par member_anomalies.py). Avant la période d'ensemble, tous les membres sont
identiques à CaSR : la mise en régime est commune.

Usage : python member_forcing.py <forçage CaSR .nc> <anomalies du membre .nc> <sortie .nc>
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

casr_path, anomalies_path, out_path = sys.argv[1:4]
ds = xr.open_dataset(casr_path).load()
names = list(ds["var"].values)
anomalies = xr.open_dataset(anomalies_path)
days = pd.DatetimeIndex(ds.time.values)
position = pd.Series(np.arange(len(days)), index=days)[pd.DatetimeIndex(anomalies.time.values)].to_numpy()
forcing = ds.forcing.values.copy()
forcing[position, :, names.index("P")] *= anomalies["p_factor"].values
forcing[position, :, names.index("Tmin")] += anomalies["tmin_delta"].values
forcing[position, :, names.index("Tmax")] += anomalies["tmax_delta"].values
# Un écart peut inverser Tmin et Tmax sur un jour : on les remet dans l'ordre.
tmin, tmax = forcing[..., names.index("Tmin")], forcing[..., names.index("Tmax")]
forcing[..., names.index("Tmin")], forcing[..., names.index("Tmax")] = np.minimum(tmin, tmax), np.maximum(tmin, tmax)
ds["forcing"].values[:] = forcing.astype(np.float32)
ds.to_netcdf(out_path)
print(f"{out_path} : P annuel {forcing[position, :, 0].mean() * 365.25:.0f} mm sur la période d'ensemble, NaN {np.isnan(forcing).any()}")
