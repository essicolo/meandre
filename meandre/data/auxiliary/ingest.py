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


def ingest_raster_tiles(source: Source, region: str) -> pd.DataFrame:
    """Rasters découpés en feuillets, lus à distance à résolution réduite.

    Chaque unité hydrologique accumule l'histogramme des valeurs entières du raster ; les
    statistiques par tronçon se calculent sur la somme des histogrammes de ses unités. Chaque
    feuillet traité est mis en cache, ce qui permet de reprendre une ingestion interrompue.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import geopandas as gpd
    import rasterio
    from rasterio import features
    from rasterio.transform import Affine

    from meandre.data.physitel_loader import _parse_troncon
    from meandre.data.auxiliary.units import project_dir
    from meandre.utils import paths as _paths

    ing = source.ingestion
    units = hydro_units(region)
    index = gpd.read_file("zip://" + source.path(ing["index"]).as_posix())
    zone = units.to_crs(index.crs).union_all()
    feuillets = index[index.intersects(zone)]
    n_classes = int(ing["classes"])
    facteur = int(ing["facteur_reduction"])
    cache = f"{_paths.DERIVED_ROOT}/auxiliaires/cache/{source.name}/{region}"
    os.makedirs(cache, exist_ok=True)
    ids_max = int(units.uhrh.max()) + 1

    def un_feuillet(ligne):
        nom = ligne[ing["champ_feuillet"]]
        f_cache = f"{cache}/{nom}.npz"
        if os.path.exists(f_cache):
            z = np.load(f_cache)
            return z["hist"], float(z["pixel_area"])
        url = "/vsicurl/" + ligne[ing["champ_url"]].rstrip("/") + "/" + ing["motif_fichier"].format(feuillet=nom)
        for essai in range(3):
            try:
                with rasterio.open(url) as ds:
                    forme = (max(1, ds.height // facteur), max(1, ds.width // facteur))
                    data = ds.read(1, out_shape=forme)
                    t = ds.transform * Affine.scale(ds.width / forme[1], ds.height / forme[0])
                    locales = units.to_crs(ds.crs)
                    gauche, bas, droite, haut = rasterio.transform.array_bounds(forme[0], forme[1], t)
                    locales = locales.cx[gauche:droite, bas:haut]
                    hist = np.zeros((ids_max, n_classes), dtype=np.int64)
                    if len(locales):
                        ids = features.rasterize(zip(locales.geometry, locales.uhrh), out_shape=forme, transform=t, fill=0, dtype="int32")
                        ok = (ids > 0) & (data != ds.nodata) & (data < n_classes)
                        cle = ids[ok].astype(np.int64) * n_classes + data[ok].astype(np.int64)
                        hist = np.bincount(cle, minlength=ids_max * n_classes).reshape(ids_max, n_classes)
                    aire = abs(t.a * t.e)
                np.savez_compressed(f_cache, hist=hist, pixel_area=aire)
                return hist, aire
            except rasterio.errors.RasterioIOError as e:
                erreur = e
        print(f"  {nom} : lecture impossible ({erreur})", flush=True)
        return None, None

    total = np.zeros((ids_max, n_classes), dtype=np.float64)
    with ThreadPoolExecutor(max_workers=int(ing.get("fils", 8))) as pool:
        for hist, aire in pool.map(un_feuillet, [r for _, r in feuillets.iterrows()]):
            if hist is not None:
                total += hist * aire
    valeurs = np.arange(n_classes, dtype=float)
    seuils = ing.get("seuils", [])
    rows = []
    area = units.set_index("uhrh").area_m2
    for tr in _parse_troncon(project_dir(region) / "physitel" / "troncon.trl"):
        ids = [u for u in tr["uhrh_ids"] if u in area.index]
        if not ids:
            continue
        a = float(area.loc[ids].sum())
        h = total[ids].sum(axis=0)
        s = h.sum()
        row = {"region": region, "troncon": int(tr["id"]), "area_m2": a}
        nom = ing["nom"]
        stats = {}
        if s > 0:
            cumul = np.cumsum(h) / s
            stats["moyenne"] = float((h * valeurs).sum() / s)
            for q in (10, 50, 90):
                stats[f"q{q}"] = float(np.searchsorted(cumul, q / 100))
            for seuil in seuils:
                stats[f"part_sup_{seuil}"] = float(h[seuil:].sum() / s)
        for k in ["moyenne", "q10", "q50", "q90"] + [f"part_sup_{x}" for x in seuils]:
            row[f"{nom}_{k}"] = stats.get(k, np.nan)
            row[f"couv_{nom}_{k}"] = s / a if a > 0 else 0.0
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["source"] = source.name
    table.attrs["nature"] = source.nature
    print(f"  {len(feuillets)} feuillets", flush=True)
    return table


INGESTEURS["raster_tuiles"] = ingest_raster_tiles
