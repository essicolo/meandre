"""Download ERA5-Land forcing data from the Copernicus Climate Data Store.

Requires a CDS API key in ``~/.cdsapirc``. Register at:
https://cds.climate.copernicus.eu/how-to-api

Downloads daily aggregates of the 6 forcing variables used by meandre:
    P       total_precipitation           (m/day → mm/day)
    T_min   2m_temperature min            (K → °C)
    T_max   2m_temperature max            (K → °C)
    R_n     surface_net_solar_radiation   (J/m² → MJ/m²/day)
    u2      10m wind → 2m wind           (m/s)
    e_a     2m dewpoint → actual vapour   (kPa)

Usage::

    from meandre.data.era5_loader import download_era5

    ds = download_era5(
        bbox=(-70.5, 46.5, -68.5, 48.5),
        date_start="2000-01-01",
        date_end="2019-12-31",
        cache_dir="data/era5/",
    )
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr


# ERA5-Land variables to download (hourly on CDS)
_ERA5_VARIABLES = [
    "total_precipitation",
    "2m_temperature",
    "surface_net_solar_radiation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
]


def download_era5(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cache_dir: str | Path,
    dataset: str = "reanalysis-era5-land",
) -> xr.Dataset:
    """Download ERA5-Land forcing and convert to meandre convention.

    Parameters
    ----------
    bbox : (west, south, east, north) in EPSG:4326.
    date_start, date_end : ISO date strings.
    cache_dir : Directory for cached NetCDF.
    dataset : CDS dataset name.

    Returns
    -------
    xr.Dataset with variables (P, T_min, T_max, R_n, u2, e_a),
    dimensions (time, latitude, longitude), daily resolution.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "era5_forcing.nc"

    if out_path.exists():
        print(f"[era5] Cached forcing: {out_path}")
        return xr.open_dataset(out_path)

    # Download raw hourly data year by year
    raw_path = cache_dir / "era5_raw.nc"
    if not raw_path.exists():
        _download_raw(bbox, date_start, date_end, raw_path, dataset)

    # Convert to daily meandre variables
    ds = _convert_to_daily(raw_path)
    ds.to_netcdf(out_path)
    print(f"[era5] Daily forcing saved: {out_path}")
    return ds


def _download_raw(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    out_path: Path,
    dataset: str,
) -> None:
    """Download raw hourly ERA5-Land data via CDS API."""
    try:
        import cdsapi
    except ImportError:
        raise ImportError(
            "cdsapi is required for ERA5 downloads. "
            "Install with: pip install cdsapi\n"
            "Then set up your API key: https://cds.climate.copernicus.eu/how-to-api"
        )

    west, south, east, north = bbox
    # CDS expects [north, west, south, east]
    area = [north, west, south, east]

    # Parse date range into years/months
    from datetime import datetime
    start = datetime.fromisoformat(date_start)
    end = datetime.fromisoformat(date_end)

    years = list(range(start.year, end.year + 1))
    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{h:02d}:00" for h in range(24)]

    c = cdsapi.Client()

    print(f"[era5] Downloading {len(years)} year(s) of ERA5-Land...")
    print(f"  bbox: {bbox}")
    print(f"  period: {date_start} to {date_end}")

    c.retrieve(
        dataset,
        {
            "product_type": "reanalysis",
            "variable": _ERA5_VARIABLES,
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": times,
            "area": area,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"[era5] Raw download saved: {out_path}")


def _convert_to_daily(raw_path: Path) -> xr.Dataset:
    """Convert hourly ERA5-Land to daily meandre forcing variables."""
    ds = xr.open_dataset(raw_path)

    # ERA5-Land variable names (may vary by download method)
    # Try common names
    def _get(names: list[str]):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"None of {names} found in ERA5 dataset: {list(ds.data_vars)}")

    tp = _get(["tp", "total_precipitation"])
    t2m = _get(["t2m", "2m_temperature"])
    ssr = _get(["ssr", "surface_net_solar_radiation"])
    u10 = _get(["u10", "10m_u_component_of_wind"])
    v10 = _get(["v10", "10m_v_component_of_wind"])
    d2m = _get(["d2m", "2m_dewpoint_temperature"])

    # Group by day
    daily = ds.resample(time="1D")

    # P: total precipitation (m → mm/day, cumulative → sum)
    P = tp.resample(time="1D").sum() * 1000.0  # m → mm

    # T_min, T_max (K → °C)
    T_min = (t2m.resample(time="1D").min()) - 273.15
    T_max = (t2m.resample(time="1D").max()) - 273.15

    # R_n: net solar radiation (J/m² cumulative → MJ/m²/day)
    R_n = ssr.resample(time="1D").sum() / 1e6

    # u2: 10m wind → 2m wind (log wind profile, FAO-56)
    wind10 = np.sqrt(u10**2 + v10**2)
    u2_hourly = wind10 * (4.87 / np.log(67.8 * 10 - 5.42))  # FAO conversion
    u2 = u2_hourly.resample(time="1D").mean()

    # e_a: actual vapour pressure from dewpoint (kPa, FAO-56 Eq. 14)
    d2m_C = d2m - 273.15
    e_a_hourly = 0.6108 * np.exp(17.27 * d2m_C / (d2m_C + 237.3))
    e_a = e_a_hourly.resample(time="1D").mean()

    # Build output dataset
    out = xr.Dataset({
        "P": P,
        "T_min": T_min,
        "T_max": T_max,
        "R_n": R_n,
        "u2": u2,
        "e_a": e_a,
    })

    out.attrs["description"] = "meandre forcing from ERA5-Land"
    out.attrs["units"] = "P:mm/day, T:C, R_n:MJ/m2/day, u2:m/s, e_a:kPa"

    ds.close()
    return out


# ---------------------------------------------------------------------------
# Gridded ERA5-Land over a basin bbox — for config-driven forcing override.
# Net radiation = net shortwave (ssr) + net longwave (str). Variables fetched
# month-by-month (CDS cost limits) and cached per month so runs are resumable.
# ---------------------------------------------------------------------------

_ERA5_GRID_VARIABLES = [
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
]


def _month_range(date_start: str, date_end: str):
    from datetime import datetime
    s = datetime.fromisoformat(date_start)
    e = datetime.fromisoformat(date_end)
    y, m = s.year, s.month
    while (y, m) <= (e.year, e.month):
        yield y, m
        m = 1 if m == 12 else m + 1
        y = y + 1 if m == 1 else y


def fetch_era5_grid(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cache_dir: str | Path,
) -> list[Path]:
    """Download gridded ERA5-Land month-by-month over ``bbox`` (W,S,E,N).

    Returns the list of cached monthly NetCDF paths. Skips months already
    present. Raises if cdsapi/API key are missing.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = bbox
    paths: list[Path] = []
    client = None
    for y, m in _month_range(date_start, date_end):
        out = cache_dir / f"era5_grid_{y}{m:02d}.nc"
        paths.append(out)
        if out.exists():
            continue
        if client is None:
            try:
                import cdsapi
            except ImportError as exc:
                raise ImportError(
                    "cdsapi requis pour le fetch ERA5-Land. pip install cdsapi "
                    "+ clé ~/.cdsapirc (https://cds.climate.copernicus.eu/how-to-api)"
                ) from exc
            client = cdsapi.Client()
        print(f"[era5] grille {y}-{m:02d} bbox={bbox} → {out}")
        client.retrieve(
            "reanalysis-era5-land",
            {
                "variable": _ERA5_GRID_VARIABLES,
                "year": str(y),
                "month": f"{m:02d}",
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [north, west, south, east],  # N, W, S, E
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            str(out),
        )
    return paths


def _is_accumulated(da, tdim: str) -> bool:
    """Devine si ``da`` est un cumul depuis 00 UTC (intra-jour monotone) plutôt
    qu'un flux horaire (cycle diurne).

    ERA5-Land : la valeur à 00:00 = total de la veille (artefact de reset), donc
    on EXCLUT les diffs de transition de minuit et on teste la constance de
    signe des diffs *intra-jour* uniquement."""
    import numpy as np
    import pandas as pd
    n = min(96, da.sizes[tdim])  # ~4 premiers jours
    sample = da.isel({tdim: slice(0, n)})
    spatial = [d for d in sample.dims if d != tdim]
    s = sample.mean(dim=spatial).values if spatial else sample.values
    hours = pd.to_datetime(sample[tdim].values).hour
    d = np.diff(s)
    # diff[i] relie hours[i] → hours[i+1] ; on jette toute transition touchant
    # 00:00 (reset : 23→00 ET 00→01, ce dernier va du total-veille à ~0).
    keep = (hours[1:] != 0) & (hours[:-1] != 0)
    d = d[keep]
    d = d[np.abs(d) > 1e-6]
    if d.size == 0:
        return False
    frac_same_sign = max((d > 0).mean(), (d < 0).mean())
    return frac_same_sign > 0.9  # quasi-monotone intra-jour → cumul


def _daily_accumulated(da, tdim: str):
    """Total journalier d'une variable cumulée depuis 00 UTC = valeur en fin de
    jour (dernier pas). Robuste au signe (ssr>0, str<0)."""
    return da.resample({tdim: "1D"}).last()


_EDH_URL = "https://data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr"


def _edh_token() -> str:
    """Token DestinE (Bearer) depuis ~/.netrc (machine data.earthdatahub.destine.eu).
    Cherche aussi le .netrc Windows depuis WSL. login VIDE, password = token."""
    import netrc
    candidates = [None, "/mnt/c/Users/parse01/.netrc"]
    for p in candidates:
        try:
            nr = netrc.netrc(p) if p else netrc.netrc()
            auth = nr.authenticators("data.earthdatahub.destine.eu")
            if auth and auth[2]:
                return auth[2]
        except Exception:
            continue
    raise RuntimeError(
        "Token EarthDataHub introuvable (~/.netrc ou /mnt/c/Users/parse01/.netrc, "
        "machine data.earthdatahub.destine.eu). Régénérer un token DestinE si expiré."
    )


def era5_grid_daily(
    bbox: tuple[float, float, float, float],
    date_start: str,
    date_end: str,
    cache_dir: str | Path,
):
    """ERA5 single-levels journalier (EarthDataHub ARCO Zarr) sur le bbox :
    R_n (net = ssr+str, MJ/m²/j), e_a (kPa), u2 (m/s). Dims (time, lat, lon).

    Lecture cloud lazy via xarray+token Bearer (pas de file CDS). Le résultat
    journalier (petit) est caché ; le 1er appel télécharge le subset horaire
    (~heure(s) selon la période)."""
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_f = cache_dir / f"era5_edh_daily_{date_start}_{date_end}.nc"
    if cache_f.exists():
        print(f"[era5/EDH] cache journalier : {cache_f}")
        return xr.open_dataset(cache_f)

    west, south, east, north = bbox
    token = _edh_token()
    print(f"[era5/EDH] ouverture {_EDH_URL} (lazy)…")
    ds = xr.open_zarr(
        _EDH_URL, consolidated=True, chunks={},
        storage_options={"headers": {"Authorization": f"Bearer {token}"}},
    )
    tdim = "valid_time" if "valid_time" in ds.dims else "time"
    # lon ERA5 en 0-360 ; lat décroissante (90→-90)
    sub = ds[["ssr", "str", "d2m", "u10", "v10"]].sel({
        tdim: slice(date_start, date_end),
        "latitude": slice(north, south),
        "longitude": slice(west % 360, east % 360),
    })
    if "number" in sub.dims:
        sub = sub.isel(number=0, drop=True)
    print(f"[era5/EDH] subset {dict(sub.sizes)} — téléchargement (chunk-aligné)…")

    ssr, strad, d2m, u10, v10 = sub.ssr, sub["str"], sub.d2m, sub.u10, sub.v10
    # ERA5 single-levels : accumulés à l'HEURE (flux horaire) → daily = somme.
    # _is_accumulated auto-détecte cumul-depuis-00 (ERA5-Land) vs flux (ERA5).
    if _is_accumulated(ssr, tdim):
        ssr_d, str_d = _daily_accumulated(ssr, tdim), _daily_accumulated(strad, tdim)
    else:
        ssr_d = ssr.resample({tdim: "1D"}).sum()
        str_d = strad.resample({tdim: "1D"}).sum()
    R_n = (ssr_d + str_d) / 1e6  # J/m² → MJ/m²/j

    d2m_C = d2m - 273.15
    e_a = (0.6108 * np.exp(17.27 * d2m_C / (d2m_C + 237.3))).resample({tdim: "1D"}).mean()
    wind10 = np.sqrt(u10 ** 2 + v10 ** 2)
    u2 = (wind10 * (4.87 / np.log(67.8 * 10 - 5.42))).resample({tdim: "1D"}).mean()

    out = xr.Dataset({"R_n": R_n, "e_a": e_a, "u2": u2}).rename({tdim: "time"})
    out = out.compute()  # streaming dask : télécharge + agrège sans tout charger en RAM
    # lon 0-360 → -180..180 pour matcher node_coords
    out = out.assign_coords(longitude=((out.longitude + 180) % 360 - 180)).sortby("longitude")
    out.to_netcdf(cache_f)
    print(f"[era5/EDH] journalier mis en cache : {cache_f}")
    return out
