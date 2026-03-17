"""Download daily streamflow from Environment Canada HYDAT.

Uses the public Datamart CSV endpoint — no API key required.

Usage::

    from meandre.data.hydat_loader import download_hydat_station

    df = download_hydat_station("02RH035", "2000-01-01", "2019-12-31")
    # Returns DataFrame with columns: date, discharge (m³/s)
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd


# Environment Canada wateroffice CSV endpoint (public)
_EC_BASE = (
    "https://wateroffice.ec.gc.ca/report/real_time_e.html"
    "?stn={station_id}&mode=Table&type=realTime"
    "&startDate={start}&endDate={end}&prm1=47"  # 47 = discharge
)

# Historical daily data endpoint
_EC_HIST = (
    "https://wateroffice.ec.gc.ca/download/report_e.html"
    "?type=h2oArc&stn={station_id}"
    "&startDate={start}&endDate={end}"
    "&parameterType=Flow&resolution=daily"
)


def download_hydat_station(
    station_id: str,
    date_start: str,
    date_end: str,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download daily discharge for one HYDAT station.

    Parameters
    ----------
    station_id :
        HYDAT station ID (e.g., "02RH035").
    date_start, date_end :
        ISO 8601 date strings (inclusive).
    cache_dir :
        Optional directory to cache downloaded CSV.

    Returns
    -------
    DataFrame with columns ``date`` (datetime64) and ``discharge`` (float, m³/s).
    NaN for missing days.
    """
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{station_id}_{date_start}_{date_end}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["date"])

    df = _download_csv(station_id, date_start, date_end)

    if cache_dir and not df.empty:
        df.to_csv(cache_path, index=False)
        print(f"[hydat] Cached: {cache_path}")

    return df


def _download_csv(
    station_id: str, date_start: str, date_end: str,
) -> pd.DataFrame:
    """Fetch daily discharge CSV from Environment Canada."""
    import urllib.request

    url = _EC_HIST.format(
        station_id=station_id, start=date_start, end=date_end,
    )
    print(f"[hydat] Downloading {station_id} ({date_start} to {date_end})...")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "meandre/1.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            text = response.read().decode("utf-8")
    except Exception as e:
        print(f"[hydat] Warning: download failed for {station_id}: {e}")
        return pd.DataFrame(columns=["date", "discharge"])

    # Parse the CSV (EC format has header lines to skip)
    lines = text.strip().split("\n")

    # Find the header line (contains "Date" or "DATE")
    header_idx = 0
    for i, line in enumerate(lines):
        if "Date" in line or "DATE" in line:
            header_idx = i
            break

    csv_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        print(f"[hydat] Warning: could not parse CSV for {station_id}")
        return pd.DataFrame(columns=["date", "discharge"])

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find date and flow columns
    date_col = next((c for c in df.columns if "date" in c), None)
    flow_col = next(
        (c for c in df.columns if "flow" in c or "discharge" in c or "value" in c),
        None,
    )

    if date_col is None or flow_col is None:
        print(f"[hydat] Warning: unexpected columns: {list(df.columns)}")
        return pd.DataFrame(columns=["date", "discharge"])

    result = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "discharge": pd.to_numeric(df[flow_col], errors="coerce"),
    })
    result = result.dropna(subset=["date"])
    result = result.sort_values("date").reset_index(drop=True)

    # Filter to requested date range
    mask = (result["date"] >= date_start) & (result["date"] <= date_end)
    result = result[mask].reset_index(drop=True)

    print(f"[hydat] {station_id}: {len(result)} daily values")
    return result


def download_hydat_stations(
    station_ids: list[str],
    date_start: str,
    date_end: str,
    cache_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Download daily discharge for multiple stations.

    Returns dict mapping station_id → DataFrame.
    """
    results = {}
    for sid in station_ids:
        results[sid] = download_hydat_station(sid, date_start, date_end, cache_dir)
    return results
