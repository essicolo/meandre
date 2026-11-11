"""Écart d'un forçage de modèle climatique à CaSR, par mois, sur la période commune (2001-2020 par défaut) :
moyenne sur les tronçons du territoire, rapport pour la précipitation, le rayonnement, le vent et la
pression de vapeur, différence en °C pour les températures ; corrélation des anomalies journalières de
température et de précipitation (accord jour par jour, attendu seulement pour une simulation pilotée
par une réanalyse).

Usage : python compare_forcing.py <forcing CaSR .nc> <forcing modèle .nc> [début fin]
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

start, end = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("2001-01-01", "2020-12-31")
a = xr.open_dataset(sys.argv[1]).forcing.sel(time=slice(start, end))
b = xr.open_dataset(sys.argv[2]).forcing.sel(time=slice(start, end))
a, b = xr.align(a, b, join="inner")
m = a.time.dt.month
ma = a.mean("node").groupby(m).mean().to_pandas()
mb = b.mean("node").groupby(m).mean().to_pandas()
out = pd.DataFrame(index=ma.index)
for v in ("P", "R_n", "u2", "e_a"):
    out[f"{v} rapport"] = mb[v] / ma[v]
for v in ("Tmin", "Tmax"):
    out[f"{v} écart °C"] = mb[v] - ma[v]
pd.set_option("display.width", 200)
print(f"{a.sizes['time']} jours communs, {a.sizes['node']} tronçons")
print("Annuel : P " + f"{float(a.sel(var='P').mean()) * 365.25:.0f} contre {float(b.sel(var='P').mean()) * 365.25:.0f} mm/an (CaSR, modèle)")
print(out.round(2).T.to_string())
for v in ("Tmax", "P"):
    x = a.sel(var=v).mean("node").to_pandas()
    y = b.sel(var=v).mean("node").to_pandas()
    ax = x - x.groupby(x.index.dayofyear).transform("mean")
    ay = y - y.groupby(y.index.dayofyear).transform("mean")
    print(f"corrélation jour par jour des anomalies, {v} moyenne du territoire : {np.corrcoef(ax, ay)[0, 1]:.2f}")
