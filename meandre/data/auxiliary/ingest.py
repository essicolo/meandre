"""Ingestion d'une source auxiliaire au tronçon, selon le type déclaré dans son source.toml."""
from __future__ import annotations

import numpy as np
import pandas as pd

from meandre.data.auxiliary.catalog import Source
from meandre.data.auxiliary.units import hydro_units, to_reaches


def _ilr(parts: np.ndarray) -> np.ndarray:
    import nuee

    return nuee.ilr(nuee.multiplicative_replacement(nuee.closure(parts)))


def _transform(values: np.ndarray, name: str | None, floor: float) -> np.ndarray:
    if name in (None, "aucune"):
        return values
    if name == "log":
        return np.log(np.maximum(values, floor))
    raise ValueError(f"transformation inconnue : {name}")


def ingest_raster_bands(source: Source, region: str) -> pd.DataFrame:
    """Rasters multibandes alignés : compositions en ilr et variables simples, bande par bande."""
    import rasterio

    from meandre.data.auxiliary.raster import UnitGrid

    ing = source.ingestion
    bandes = ing["bandes"]
    units = hydro_units(region)
    area = units.set_index("uhrh").area_m2
    num, den = {}, {}
    ouverts = {}

    def ouvrir(fichier):
        if fichier not in ouverts:
            ouverts[fichier] = rasterio.open(source.path(fichier))
        return ouverts[fichier]

    premier = ing.get("composition", [{}])[0].get("parts", {}) or {}
    reference = ouvrir(next(iter(premier.values())) if premier else ing["variable"][0]["fichier"])
    grille = UnitGrid(units, reference)

    for b, etiquette in enumerate(bandes, start=1):
        for comp in ing.get("composition", []):
            fichiers = list(comp["parts"].values())
            pixels = np.stack([grille.read(ouvrir(f), b) for f in fichiers], axis=-1)
            ok = np.isfinite(pixels).all(axis=-1) & (pixels > 0).any(axis=-1)
            coords = np.full(pixels.shape[:2] + (len(fichiers) - 1,), np.nan)
            coords[ok] = _ilr(np.clip(pixels[ok], 0.0, None))
            petits = np.stack([grille.sample(ouvrir(f), b) for f in fichiers], axis=-1) if len(grille.small) else np.empty((0, len(fichiers)))
            coords_petits = np.full((len(petits), len(fichiers) - 1), np.nan)
            okp = np.isfinite(petits).all(axis=-1) & (petits > 0).any(axis=-1) if len(petits) else np.array([], bool)
            if okp.any():
                coords_petits[okp] = _ilr(np.clip(petits[okp], 0.0, None))
            for k in range(len(fichiers) - 1):
                cle = f"{comp['nom']}_ilr_{k + 1}_{etiquette}"
                num[cle], den[cle] = grille.zonal(coords[..., k], coords_petits[:, k] if len(petits) else np.array([]))
        for var in ing.get("variable", []):
            ds = ouvrir(var["fichier"])
            v = _transform(grille.read(ds, b), var.get("transformation"), var.get("plancher", 1e-6))
            vp = _transform(grille.sample(ds, b), var.get("transformation"), var.get("plancher", 1e-6)) if len(grille.small) else np.array([])
            cle = f"{var['nom']}_{etiquette}"
            num[cle], den[cle] = grille.zonal(v, vp)
    for ds in ouverts.values():
        ds.close()
    table = to_reaches(region, pd.DataFrame(num), pd.DataFrame(den), area)
    table.attrs["source"] = source.name
    table.attrs["nature"] = source.nature
    return table


INGESTEURS = {"raster_bandes": ingest_raster_bands}


def ingest(source: Source, region: str) -> pd.DataFrame:
    genre = source.ingestion["type"]
    if genre not in INGESTEURS:
        raise ValueError(f"type d'ingestion inconnu : {genre} (connus : {sorted(INGESTEURS)})")
    return INGESTEURS[genre](source, region)
