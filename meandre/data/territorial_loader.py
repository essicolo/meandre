"""Load territorial features from GIS-derived tables.

Expects a CSV or GeoPackage with one row per subbasin/reach node and
columns matching the TerritorialFeatures feature names.
Values are z-score normalised before being passed to the spatial encoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from meandre.spatial.territorial import TerritorialFeatures, DEFAULT_PHYSICAL_COLUMNS

# Column names in the source table -> canonical feature names
COLUMN_MAP = {
    "drainage_area_km2": "drainage_area_km2",
    "strahler": "strahler_order",
    "slope_pct": "mean_slope_pct",
    "elevation_m": "mean_elevation_m",
    "sin_aspect": "sin_aspect",
    "cos_aspect": "cos_aspect",
    "f_forest": "f_forest",
    "f_agriculture": "f_agriculture",
    "f_urban": "f_urban",
    "f_wetland": "f_wetland",
    "f_water": "f_water",
    "f_sand": "f_sand",
    "f_silt": "f_silt",
    "f_clay": "f_clay",
    "depth_bedrock_m": "depth_to_bedrock_m",
    "dist_outlet_km": "dist_to_outlet_km",
    "lake_frac": "lake_fraction",
}

# Canonical feature columns (fed to the spatial network, normalised)
FEATURE_COLUMNS = [
    "drainage_area_km2",
    "strahler_order",
    "mean_slope_pct",
    "mean_elevation_m",
    "sin_aspect",
    "cos_aspect",
    "f_forest",
    "f_agriculture",
    "f_urban",
    "f_wetland",
    "f_water",
    "f_sand",
    "f_silt",
    "f_clay",
    "depth_to_bedrock_m",
    "dist_to_outlet_km",
    "lake_fraction",
]


def load_territorial(
    path: str | Path,
    normalise: bool = True,
    device: torch.device | None = None,
) -> TerritorialFeatures:
    """Load and optionally z-score normalise territorial features.

    Args:
        path:      CSV or GeoPackage path.
        normalise: If True, apply z-score per feature column.
        device:    Target device.
    Returns:
        TerritorialFeatures with (n_nodes, n_features) data tensor.
    """
    import pandas as pd

    p = Path(path)
    if p.suffix in {".gpkg", ".shp"}:
        import geopandas as gpd
        df = gpd.read_file(p).drop(columns="geometry")
    else:
        df = pd.read_csv(p)

    # Rename to canonical field names
    df = df.rename(columns={v: k2 for k2, v in COLUMN_MAP.items() if v in df.columns})

    if normalise:
        for col in df.select_dtypes(include=[np.number]).columns:
            mu, sig = df[col].mean(), df[col].std()
            if sig > 0:
                df[col] = (df[col] - mu) / sig

    n = len(df)

    def _t(col: str) -> torch.Tensor:
        if col in df.columns:
            return torch.from_numpy(df[col].values.astype(np.float32)).to(device)
        return torch.zeros(n, device=device)

    # Build feature tensor from canonical columns
    columns = FEATURE_COLUMNS
    data = torch.stack([_t(col) for col in columns], dim=-1)

    return TerritorialFeatures(data=data, columns=columns)
