"""Quantifie l'écart proxy-vs-ERA5-Land sur le rayonnement, l'humidité et le
vent pour SLSO, via l'API time-series ARCO (reanalysis-era5-land-timeseries).

Pourquoi time-series : un point sur une année complète d'un coup, sans limite
de coût (contrairement aux requêtes grille horaire).

Le dataset time-series n'expose pas le rayonnement NET — seulement le shortwave
incident (SSRD). C'est l'idéal pour le diagnostic : l'erreur du proxy est
précisément dans l'estimation Hargreaves du shortwave (R_s = kRS·√dT·R_a) ;
le terme longwave est calculé identiquement des deux côtés. On compare donc le
shortwave INCIDENT proxy-vs-mesuré, ce qui isole l'erreur Hargreaves.

  R_s : proxy Hargreaves (kRS·√dT·R_a)   vs   ERA5 SSRD          (MJ/m²/day)
  e_a : proxy es(Tmin)                    vs   ERA5 ex-dewpoint   (kPa)
  u2  : proxy constante 2.0               vs   ERA5 ex-vent       (m/s)
  T_min/T_max : contrôle de cohérence (point ERA5 vs moyenne-bassin proxy)

Usage :
  PYTHONUTF8=1 python .runs/slso/compare_era5_proxy.py download
  PYTHONUTF8=1 python .runs/slso/compare_era5_proxy.py compare
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from meandre.data.gridded_forcing import (
    _extraterrestrial_radiation,
    _saturation_vapour_pressure,
)

# Point de comparaison : centroïde du bassin SLSO
LON, LAT = -71.348, 45.946
YEAR = "2018"
KRS = 0.17  # idem gridded_forcing (continental)
CACHE = Path(".runs/slso/data/era5")
TS = CACHE / f"era5_ts_{YEAR}.nc"
PROXY = Path(".runs/slso/data/forcing.nc")


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    if TS.exists():
        print(f"[era5] time-series déjà présent : {TS}")
        return
    import cdsapi

    c = cdsapi.Client()
    print(f"[era5] download time-series {YEAR} @ ({LON},{LAT}) → {TS}")
    c.retrieve(
        "reanalysis-era5-land-timeseries",
        {
            "variable": [
                "2m_temperature",
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_solar_radiation_downwards",
            ],
            "location": {"longitude": LON, "latitude": LAT},
            "date": [f"{YEAR}-01-01/{YEAR}-12-31"],
            "data_format": "netcdf",
        },
        str(TS),
    )
    print(f"[era5] OK → {TS}")


def _era5_daily() -> pd.DataFrame:
    # L'API time-series livre un ZIP de NetCDF (un par groupe de variables).
    import zipfile
    extract_dir = TS.parent / f"ts_{YEAR}"
    if zipfile.is_zipfile(TS):
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(TS) as z:
            z.extractall(extract_dir)
        members = sorted(extract_dir.glob("*.nc"))
        ds = xr.merge([xr.open_dataset(p) for p in members], compat="override")
    else:
        ds = xr.open_dataset(TS)
    tdim = "valid_time" if "valid_time" in ds.dims else ("time" if "time" in ds.dims else list(ds.dims)[0])

    def g(names):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"{names} absent de {list(ds.data_vars)}")

    t2m = g(["t2m", "2m_temperature"])
    d2m = g(["d2m", "2m_dewpoint_temperature"])
    u10 = g(["u10", "10m_u_component_of_wind"])
    v10 = g(["v10", "10m_v_component_of_wind"])
    ssrd = g(["ssrd", "surface_solar_radiation_downwards"])

    rs = lambda da: da.resample({tdim: "1D"})  # noqa: E731
    # L'API time-series livre le flux HORAIRE dé-accumulé (J/m² par heure) →
    # total journalier = somme des 24 heures. J → MJ.
    R_s = (rs(ssrd).sum() / 1e6)
    T_min = rs(t2m).min() - 273.15
    T_max = rs(t2m).max() - 273.15
    wind10 = np.sqrt(u10 ** 2 + v10 ** 2)
    u2 = rs(wind10 * (4.87 / np.log(67.8 * 10 - 5.42))).mean()
    d2m_C = d2m - 273.15
    e_a = rs(0.6108 * np.exp(17.27 * d2m_C / (d2m_C + 237.3))).mean()

    df = pd.DataFrame({
        "R_s": R_s.squeeze().values,
        "T_min": T_min.squeeze().values,
        "T_max": T_max.squeeze().values,
        "u2": u2.squeeze().values,
        "e_a": e_a.squeeze().values,
    }, index=pd.to_datetime(R_s[tdim].values))
    ds.close()
    return df


def _proxy_daily() -> pd.DataFrame:
    pf = xr.open_dataset(PROXY)
    f = pf["forcing"]
    tp = pd.to_datetime(pf["time"].values)
    mask = tp.year == int(YEAR)
    out = {}
    for v in ["T_min", "T_max", "u2", "e_a"]:
        out[v] = f.sel(variable=v).mean(dim="node").values[mask]
    df = pd.DataFrame(out, index=tp[mask])
    pf.close()
    # R_s proxy = Hargreaves reconstruit depuis Tmin/Tmax + R_a au point
    doy = df.index.dayofyear.values.astype(float)
    R_a = _extraterrestrial_radiation(np.deg2rad(np.array([LAT])), doy)[:, 0]
    dT = np.maximum(df["T_max"].values - df["T_min"].values, 0.0)
    df["R_s"] = KRS * np.sqrt(dT) * R_a
    return df


def compare() -> None:
    era = _era5_daily()
    proxy = _proxy_daily()
    # forcing.nc est horodaté à 05:00 UTC, ERA5 à 00:00 → aligner sur la date
    era.index = era.index.normalize()
    proxy.index = proxy.index.normalize()
    print(f"\n{'='*70}\nÉCART PROXY vs ERA5-Land — SLSO {YEAR} @ ({LON},{LAT}) — mensuel\n{'='*70}")
    print("(proxy = moyenne-bassin ; ERA5 = point centroïde ; T = contrôle cohérence)\n")
    units = {"R_s": "MJ/m²/d", "e_a": "kPa", "u2": "m/s", "T_min": "°C", "T_max": "°C"}
    for v in ["R_s", "e_a", "u2", "T_min", "T_max"]:
        j = proxy[[v]].rename(columns={v: "p"}).join(
            era[[v]].rename(columns={v: "e"}), how="inner").dropna()
        bias = (j["e"] - j["p"]).mean()
        rmse = np.sqrt(((j["e"] - j["p"]) ** 2).mean())
        pc = j["p"].groupby(j.index.month).mean()
        ec = j["e"].groupby(j.index.month).mean()
        mm = sorted(pc.index)
        print(f"{v}  [{units[v]}]  biais ERA5−proxy={bias:+.3f}  RMSE={rmse:.3f}  n={len(j)}")
        print("   mois ", " ".join(f"{m:>6d}" for m in mm))
        print("   proxy", " ".join(f"{pc[m]:6.2f}" for m in mm))
        print("   era5 ", " ".join(f"{ec[m]:6.2f}" for m in mm))
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    {"download": download, "compare": compare}.get(cmd, lambda: print(f"inconnu: {cmd}"))()
