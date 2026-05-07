"""Build a basin DuckDB from a bounding box, fully automated.

Pipeline:
    bbox  ->  download_all (Planetary Computer + SoilGrids + OSM rivers)
          ->  build_basin (D8 + auto-detected outlet + zonal stats)
          ->  basin.duckdb (consumable by slso.py via basin_cache.load())

    Optionally, with --with-forcing:
          ->  download_forcing_open_meteo (Open-Meteo ERA5 archive)
          ->  forcing.nc (consumable by gridded_forcing.extract_forcing())

Usage
-----
::

    # Auto-detected outlet (highest-accumulation cell on bbox edge)
    python scripts/bbox_to_basin.py \\
        --bbox=-71.5,46.55,-71.3,46.75 \\
        --output=notebooks/test/data/basin.duckdb

    # With forcing (Open-Meteo, daily pr/tasmin/tasmax over the period)
    python scripts/bbox_to_basin.py \\
        --bbox=-73.0,44.5,-69.6,47.7 \\
        --output=notebooks/slso/data/slso-od.duckdb \\
        --with-forcing \\
        --start=2000-01-01 --end=2024-12-31

NOTE on PowerShell: values starting with ``-`` (negative coords) must use
``--bbox=...`` syntax (with ``=``) to avoid being parsed as new flags.

The geo cache (DEM, landcover, soil, forcing...) is reused across runs —
only the first call hits the network.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--bbox needs 4 comma-separated floats, got {s!r}")
    w, s_, e, n = parts
    if w >= e or s_ >= n:
        raise ValueError(f"--bbox must be west,south,east,north (got {parts})")
    return (w, s_, e, n)


def _parse_outlet(s: str | None) -> tuple[float, float] | None:
    if s is None or s.lower() == "auto":
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--outlet needs lon,lat, got {s!r}")
    return (parts[0], parts[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a basin DuckDB from a bounding box.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--bbox", required=True,
        help="west,south,east,north in EPSG:4326 (e.g. -71.5,46.55,-71.3,46.75)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output DuckDB path (e.g. notebooks/slso/data/basin.duckdb)",
    )
    parser.add_argument(
        "--outlet", default=None,
        help="lon,lat of outlet, or 'auto' (default) for max-acc cell on bbox edge",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Geo raster cache (default: <output_parent>/geo_cache)",
    )
    parser.add_argument(
        "--min-area-km2", type=float, default=1.5,
        help="Minimum subcatchment area for stream threshold (default: 1.5)",
    )
    parser.add_argument(
        "--max-subcatchments", type=int, default=3500,
        help="Maximum number of subcatchments (default: 3500)",
    )
    parser.add_argument(
        "--with-forcing", action="store_true",
        help="Also download daily forcing from Open-Meteo ERA5 archive",
    )
    parser.add_argument(
        "--start", default=None,
        help="Forcing start date (YYYY-MM-DD), required with --with-forcing",
    )
    parser.add_argument(
        "--end", default=None,
        help="Forcing end date (YYYY-MM-DD), required with --with-forcing",
    )
    parser.add_argument(
        "--forcing-resolution-deg", type=float, default=0.1,
        help="Open-Meteo grid spacing in degrees (default: 0.1 ≈ 11 km)",
    )
    args = parser.parse_args(argv)

    if args.with_forcing and (args.start is None or args.end is None):
        parser.error("--with-forcing requires --start and --end")

    bbox = _parse_bbox(args.bbox)
    outlet = _parse_outlet(args.outlet)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = (
        Path(args.cache_dir) if args.cache_dir
        else output.parent / "geo_cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"BBOX               : {bbox}")
    print(f"Outlet             : {outlet if outlet else 'AUTO (max-acc on edge)'}")
    print(f"Output DB          : {output}")
    print(f"Geo cache          : {cache_dir}")
    print(f"min_area_km2       : {args.min_area_km2}")
    print(f"max_subcatchments  : {args.max_subcatchments}")
    print(f"Surface estimée    : ~{(bbox[2]-bbox[0])*111:.0f} × {(bbox[3]-bbox[1])*111:.0f} km")
    print()

    from meandre.data.open_data import download_all
    from meandre.data.basin_builder import build_basin

    print("=== Step 1: Download geo rasters ===")
    rasters = download_all(bbox=bbox, cache_dir=cache_dir)
    for k, v in rasters.items():
        print(f"  {k:18s} : {v}")
    print()

    if rasters["dem"] is None:
        print("[FATAL] DEM unavailable — cannot build basin", file=sys.stderr)
        return 2

    print("=== Step 2: Build basin (delineate + zonal stats) ===")
    if output.exists():
        output.unlink()
        print(f"  removed existing: {output}")

    cache = build_basin(
        dem_path              = rasters["dem"],
        landcover_path        = rasters["landcover"],
        soil_dir              = rasters["soil_dir"],
        outlet                = outlet,
        basin_db              = output,
        min_area_km2          = args.min_area_km2,
        max_subcatchments     = args.max_subcatchments,
        water_occurrence_path = rasters["water_occurrence"],
        lai_path              = rasters["lai"],
        nrcan_lc_path         = rasters["nrcan_lc"],
        water_polygons_path   = rasters["water_polygons"],
    )

    if args.with_forcing:
        print()
        print(f"=== Step 3: Download forcing ({args.start} -> {args.end}) ===")
        from meandre.data.open_data import download_forcing_open_meteo
        forcing_path = download_forcing_open_meteo(
            bbox           = bbox,
            start_date     = args.start,
            end_date       = args.end,
            cache_dir      = cache_dir,
            resolution_deg = args.forcing_resolution_deg,
        )
        if forcing_path is None:
            print("[!] Forcing download failed", file=sys.stderr)
            return 3
        print(f"  forcing.nc: {forcing_path}")
        print(f"  use with slso.py via [paths] forcing_cache={forcing_path}")

    print()
    print(f"=== Done: {output} ===")
    print(f"  use with slso.py via basin_db={output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
