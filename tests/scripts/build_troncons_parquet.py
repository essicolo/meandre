"""Convertit les SHP PHYSITEL en parquet GeoParquet pour les diagnostics.

Combine rivieres.shp (LineStrings, ident positif) et lacs.shp (Polygones,
ident négatif) en un seul fichier avec :
  - node_id : int  — = ident pour rivières, |ident| pour lacs (matche slso.duckdb)
  - is_lake : bool
  - geometry : LineString | Polygon (EPSG:4326)

Lance une fois après modification des SHP. Le parquet résultant est consommé
par diagnostics.qmd.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent
SHP_DIR = ROOT / "physitel"
OUT = ROOT / "data" / "troncons.parquet"


def main() -> None:
    riv = gpd.read_file(SHP_DIR / "rivieres.shp").to_crs("EPSG:4326")
    lacs = gpd.read_file(SHP_DIR / "lacs.shp").to_crs("EPSG:4326")

    riv_out = gpd.GeoDataFrame(
        {"node_id": riv["ident"].astype("int32"), "is_lake": False},
        geometry=riv.geometry, crs="EPSG:4326",
    )
    lacs_out = gpd.GeoDataFrame(
        {"node_id": lacs["ident"].abs().astype("int32"), "is_lake": True},
        geometry=lacs.geometry, crs="EPSG:4326",
    )
    troncons = pd.concat([riv_out, lacs_out], ignore_index=True)
    troncons = gpd.GeoDataFrame(troncons, geometry="geometry", crs="EPSG:4326")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    troncons.to_parquet(OUT)
    print(f"Écrit : {OUT}  ({len(troncons)} géométries, "
          f"{(~troncons['is_lake']).sum()} rivières, "
          f"{troncons['is_lake'].sum()} lacs)")


if __name__ == "__main__":
    main()
