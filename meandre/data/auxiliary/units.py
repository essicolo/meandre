"""Unités hydrologiques de PHYSITEL et agrégation au tronçon, communes à toutes les sources."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

PLATEFORME = "LN24HA"


def project_dir(region: str) -> Path:
    return Path(_paths.PLATFORMS_ROOT) / PLATEFORME / f"{region.upper()}_{PLATEFORME}_2020"


def hydro_units(region: str):
    """Unités hydrologiques d'une région, une ligne par identifiant.

    Une unité peut compter plusieurs polygones dans `uhrh.shp` : ils sont fusionnés, et
    l'aire est celle de l'union.
    """
    import geopandas as gpd

    uh = gpd.read_file(project_dir(region) / "physitel" / "uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    uh = uh.dissolve(by="uhrh").reset_index()
    uh["area_m2"] = uh.geometry.area
    return uh


def to_reaches(region: str, numerators: pd.DataFrame, denominators: pd.DataFrame, unit_area: pd.Series) -> pd.DataFrame:
    """Moyennes pondérées par tronçon à partir de sommes par unité hydrologique.

    `numerators` et `denominators` sont indexés par unité, une colonne par variable : somme de
    valeur × aire valide et aire valide. La couverture est l'aire valide rapportée à l'aire du
    tronçon.
    """
    rows = []
    for t in _parse_troncon(project_dir(region) / "physitel" / "troncon.trl"):
        ids = [u for u in t["uhrh_ids"] if u in unit_area.index]
        if not ids:
            continue
        area = float(unit_area.loc[ids].sum())
        num = numerators.reindex(ids).fillna(0.0).sum()
        den = denominators.reindex(ids).fillna(0.0).sum()
        row = {"region": region, "troncon": int(t["id"]), "area_m2": area}
        for v in numerators.columns:
            row[v] = num[v] / den[v] if den[v] > 0 else np.nan
            row[f"couv_{v}"] = den[v] / area if area > 0 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
