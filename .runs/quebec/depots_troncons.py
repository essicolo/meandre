"""Dépôts quaternaires du SIGEOM par tronçon : familles de perméabilité, roc, part non
cartographiée et densité d'eskers, pour les attributs du champ spatial.

Motif (registre R314, R315) : le modèle plafonne l'écoulement de base vers 0,60 là où
l'observation atteint 0,70 ; le champ ne reçoit aucune information sur les dépôts meubles
(épaisseur des dépôts nulle partout). Les familles sont celles de la clé validée par un
géologue du Quaternaire (`croiser_drainage.py`). La part non cartographiée est gardée telle
quelle : une absence de carte n'est pas un dépôt imperméable.

Méthode : grille de 100 m dans la projection du projet PHYSITEL, une cellule par famille,
sommes par unité hydrologique puis par tronçon ; eskers par longueur dans chaque unité.

    python .runs/quebec/depots_troncons.py abit cnda ... vaud
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from croiser_drainage import CODE_VERS_GROUPE, RESOLUTION, SIGEOM, SIGEOM_GROUPE, rasteriser
from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

SORTIE = f"{_paths.DERIVED_ROOT}/auxiliaires/depots-sigeom-troncons.parquet"
FAMILLES = list(SIGEOM_GROUPE) + ["roc"]


def une_region(reg):
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    t0 = time.time()
    proj = Path(_paths.PLATFORMS_ROOT) / "LN24HA" / f"{reg.upper()}_LN24HA_2020" / "physitel"
    uh = gpd.read_file(proj / "uhrh.shp")
    col = "ident" if "ident" in uh.columns else uh.columns[0]
    uh = uh.rename(columns={col: "uhrh"})[["uhrh", "geometry"]]
    uh["uhrh"] = uh.uhrh.astype(int)
    crs = uh.crs
    x0, y0, x1, y1 = uh.total_bounds
    emprise = tuple(uh.to_crs(4326).total_bounds)
    forme = (int(np.ceil((y1 - y0) / RESOLUTION)), int(np.ceil((x1 - x0) / RESOLUTION)))
    transform = from_origin(x0, y1, RESOLUTION, RESOLUTION)
    ids = features.rasterize(((g, int(u)) for g, u in zip(uh.geometry, uh.uhrh)), out_shape=forme,
                             transform=transform, fill=0, dtype=np.int32)
    z = gpd.read_file(SIGEOM, layer="F10E15_ZONE_MORPH_SEDIM", columns=["CODE_DEPOT_MORP_SEDM"],
                      engine="pyogrio", bbox=emprise).to_crs(crs)
    code = z.CODE_DEPOT_MORP_SEDM.fillna("")
    z["groupe"] = code.map(CODE_VERS_GROUPE)
    z.loc[code.str.startswith("R"), "groupe"] = "roc"
    z["carte"] = 1.0
    dans = ids > 0
    n = np.bincount(ids[dans])
    par = pd.DataFrame({"n": n})
    carte = rasteriser(z, "carte", transform, forme)
    par["carte"] = np.bincount(ids[dans & np.isfinite(carte)], minlength=len(n))
    for f in FAMILLES:
        z[f"est_{f}"] = np.where(z.groupe == f, 1.0, np.nan)
        r = rasteriser(z, f"est_{f}", transform, forme)
        par[f] = np.bincount(ids[dans & np.isfinite(r)], minlength=len(n))
    # Eskers : longueur dans chaque unité hydrologique, en km.
    esk = gpd.read_file(SIGEOM, layer="F10E36_ESKER_QC_LO", engine="pyogrio", bbox=emprise).to_crs(crs)
    par["esker_km"] = 0.0
    if len(esk):
        inter = gpd.overlay(esk[["geometry"]], uh, how="intersection", keep_geom_type=True)
        lon = inter.geometry.length.groupby(inter.uhrh).sum() / 1000.0
        par.loc[par.index.intersection(lon.index), "esker_km"] = lon.reindex(par.index).fillna(0.0)
    par = par[par.n > 0]
    out = []
    for t in _parse_troncon(proj / "troncon.trl"):
        u = [x for x in t["uhrh_ids"] if x in par.index]
        if not u:
            continue
        s = par.loc[u].sum()
        aire_km2 = float(s.n) * RESOLUTION ** 2 / 1e6
        row = {"region": reg, "troncon": int(t["id"]), "aire_km2": aire_km2,
               "depot_non_carto": 1.0 - float(s.carte / s.n),
               "esker_km_100km2": 100.0 * float(s.esker_km) / max(aire_km2, 1e-6)}
        for f in FAMILLES:
            row[f"depot_{f}"] = float(s[f] / s.n)
        out.append(row)
    d = pd.DataFrame(out)
    print(f"{reg}: {len(d)} tronçons | non cartographié {d.depot_non_carto.mean():.2f} | perméables "
          f"{(d.depot_tres_permeable + d.depot_permeable).mean():.2f} | tills {d.depot_moyennement_permeable.mean():.2f} | "
          f"eskers {d.esker_km_100km2.mean():.2f} km/100 km² | {time.time() - t0:.0f} s", flush=True)
    return d


def main(regions):
    old = pd.read_parquet(SORTIE) if os.path.exists(SORTIE) else None
    parts = [une_region(r) for r in regions]
    d = pd.concat(parts, ignore_index=True)
    if old is not None:
        d = pd.concat([old[~old.region.isin(regions)], d], ignore_index=True)
    d.to_parquet(SORTIE, index=False)
    print(f"-> {SORTIE} ({len(d)} tronçons, {d.region.nunique()} territoires)")


if __name__ == "__main__":
    main([r.lower() for r in sys.argv[1:]])
