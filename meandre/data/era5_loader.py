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
