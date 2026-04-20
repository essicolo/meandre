"""Geology enrichment — add lithological fractions to a DuckDB territorial table.

Performs a spatial join between node centroids and geology polygons (e.g. SIGEOM)
and writes one-hot fraction columns into the existing ``territorial`` table.

This module is **optional**: it requires ``geopandas`` at runtime but the rest
of meandre works without it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def enrich_territorial_with_geology(
    db_path: str | Path,
    gpkg_path: str | Path,
    layer: str,
    class_col: str,
    prefix: str = "geo",
    top_k: int = 8,
    coarsen: str | None = None,
) -> list[str]:
    """Add geology fraction columns to a DuckDB territorial table.

    For each node centroid, a point-in-polygon join assigns the dominant
    geological class.  The ``top_k`` most frequent classes become one-hot
    columns (prefixed with *prefix*); remaining classes are lumped into
    ``{prefix}_other``.

    Parameters
    ----------
    db_path : path
        DuckDB database containing ``nodes`` (lon, lat) and ``territorial``.
    gpkg_path : path
        GeoPackage file with geology polygons.
    layer : str
        Layer name inside the GeoPackage.
    class_col : str
        Column in the geology layer that holds the class label.
    prefix : str
        Column name prefix for the one-hot columns (e.g. ``"bedrock"``).
    top_k : int
        Number of most-frequent classes to keep; the rest become ``_other``.
    coarsen : str or None
        Optional Python expression applied to each class label to reduce
        cardinality **before** counting.  The variable ``x`` holds the raw
        label string.  Example: ``"x[:2]"`` keeps the first two characters.

    Returns
    -------
    list[str]
        Names of the columns added to ``territorial``.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for geology enrichment.  "
            "Install it with:  pip install geopandas"
        ) from exc

    db_path = Path(db_path)
    gpkg_path = Path(gpkg_path)

    # ------------------------------------------------------------------
    # 1. Read node centroids from DuckDB
    # ------------------------------------------------------------------
    con = duckdb.connect(str(db_path), read_only=True)
    nodes = con.execute(
        "SELECT node_idx, lon, lat FROM nodes ORDER BY node_idx"
    ).df()
    con.close()

    logger.info("Loaded %d node centroids from %s", len(nodes), db_path.name)

    pts = gpd.GeoDataFrame(
        nodes,
        geometry=[Point(lon, lat) for lon, lat in zip(nodes["lon"], nodes["lat"])],
        crs="EPSG:4326",
    )

    # ------------------------------------------------------------------
    # 2. Load geology polygons (only the class column + geometry)
    # ------------------------------------------------------------------
    logger.info("Reading layer '%s' from %s ...", layer, gpkg_path.name)
    geo = gpd.read_file(gpkg_path, layer=layer, columns=[class_col])
    logger.info("Loaded %d geology polygons", len(geo))

    # Reproject points to match geology CRS if needed
    if pts.crs != geo.crs:
        pts = pts.to_crs(geo.crs)

    # ------------------------------------------------------------------
    # 3. Spatial join (point-in-polygon)
    # ------------------------------------------------------------------
    joined = gpd.sjoin(pts, geo, how="left", predicate="within")

    # Resolve duplicates: keep first match per node
    joined = joined.drop_duplicates(subset="node_idx", keep="first")

    raw_class = joined[class_col].fillna("unknown").astype(str)

    # Optional coarsening
    if coarsen is not None:
        raw_class = raw_class.apply(lambda x: eval(coarsen))  # noqa: S307

    joined["geo_class"] = raw_class

    # ------------------------------------------------------------------
    # 4. Identify top_k classes
    # ------------------------------------------------------------------
    counts = joined["geo_class"].value_counts()
    top_classes = list(counts.index[:top_k])

    def _safe_col(cls: str) -> str:
        """Sanitise a class label for use as a SQL column name."""
        return cls.lower().replace(" ", "_").replace("-", "_").replace("/", "_")

    col_names = [f"{prefix}_{_safe_col(c)}" for c in top_classes]
    col_names.append(f"{prefix}_other")

    # ------------------------------------------------------------------
    # 5. Build one-hot DataFrame
    # ------------------------------------------------------------------
    one_hot = pd.DataFrame(0.0, index=nodes["node_idx"], columns=col_names)
    for idx, cls in zip(joined["node_idx"], joined["geo_class"]):
        if cls in top_classes:
            col = f"{prefix}_{_safe_col(cls)}"
        else:
            col = f"{prefix}_other"
        one_hot.at[idx, col] = 1.0

    one_hot = one_hot.reset_index().rename(columns={"index": "node_idx"})

    # Z-score normalise so geology features have the same scale as the
    # original territorial features (mean~0, std~1).
    for col in col_names:
        mu = one_hot[col].mean()
        sigma = one_hot[col].std()
        if sigma > 1e-8:
            one_hot[col] = (one_hot[col] - mu) / sigma
        else:
            one_hot[col] = 0.0  # constant column — zero out

    logger.info(
        "Classes kept: %s  (+ other: %d nodes)",
        {c: int(counts[c]) for c in top_classes},
        int((joined["geo_class"].isin(top_classes) == False).sum()),  # noqa: E712
    )

    # ------------------------------------------------------------------
    # 6. Write columns into DuckDB territorial table
    # ------------------------------------------------------------------
    con = duckdb.connect(str(db_path))

    # Check which columns already exist
    existing = set(
        con.execute("PRAGMA table_info('territorial')").df()["name"]
    )

    for col in col_names:
        if col in existing:
            logger.info("Column '%s' already exists — will overwrite values", col)
        else:
            con.execute(f"ALTER TABLE territorial ADD COLUMN \"{col}\" DOUBLE DEFAULT 0")

    # Bulk update via a temporary table
    con.register("_geo_onehot", one_hot)
    for col in col_names:
        con.execute(f"""
            UPDATE territorial
            SET \"{col}\" = g.\"{col}\"
            FROM _geo_onehot g
            WHERE territorial.node_idx = g.node_idx
        """)
    con.unregister("_geo_onehot")

    con.close()
    logger.info("Wrote %d columns to territorial in %s", len(col_names), db_path.name)

    return col_names
