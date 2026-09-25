"""Textures SoilGrids 2.0 (ISRIC, 250 m, mondiales, predites) par troncon, pour le banc.

Lecture a distance des VRT d'ISRIC, reprojection sur la grille de 100 m de chaque territoire
(grilles de la reconstruction), moyenne par unite hydrologique puis par troncon en coordonnees
ilr, comme la texture SIIGSOL. Motif : Song et coauteurs (2024) donnent SoilGrids et ses
derives par pedotransfert (HiHydroSoil) a leur reseau ; ici on mesure si cette texture predite
mondialement porte l'information que la texture SIIGSOL porte (+42 % au banc).

    .venv/Scripts/python.exe .runs/quebec/croiser_soilgrids.py mont gasp outv sagu slso
"""
import os
import sys

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.physitel_loader import _parse_troncon
from meandre.utils import paths as _paths

GRILLES = Path(f"{_paths.DATA_ROOT}/derives/reconstruction")
ISRIC = "https://files.isric.org/soilgrids/latest/data"
PROFONDEURS = ("0-5cm", "5-15cm", "15-30cm")


def lire(prop, prof, crs, transform, forme):
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    url = f"/vsicurl/{ISRIC}/{prop}/{prop}_{prof}_mean.vrt"
    with rasterio.open(url) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=forme[1], height=forme[0], resampling=Resampling.average, nodata=src.nodata) as v:
            a = v.read(1, masked=True).astype(np.float32).filled(np.nan)
    # SoilGrids code les textures en g/kg ; on rend des fractions.
    return a / 1000.0


def une_region(reg):
    from rasterio.transform import Affine
    z = np.load(GRILLES / f"grille-{reg}.npz", allow_pickle=True)
    ids, crs = z["ids"], str(z["crs"])
    t = Affine.from_gdal(*z["transform"])
    forme = ids.shape
    dans = ids > 0
    n = np.bincount(ids[dans])
    par = {}
    for prof in PROFONDEURS:
        tex = {p: lire(p, prof, crs, t, forme) for p in ("sand", "silt", "clay")}
        s = np.stack([tex["sand"], tex["silt"], tex["clay"]], axis=-1)
        ok = np.isfinite(s).all(axis=-1) & (s > 0).all(axis=-1) & dans
        # Coordonnees ilr de la composition (sable, limon, argile), meme base que SIIGSOL.
        l = np.log(s[ok])
        ilr1 = (l[:, 0] - l[:, 1]) / np.sqrt(2.0)
        ilr2 = (l[:, 0] + l[:, 1] - 2.0 * l[:, 2]) / np.sqrt(6.0)
        for nom, v in (("ilr_1", ilr1), ("ilr_2", ilr2)):
            par[f"num_{nom}_{prof}"] = np.bincount(ids[ok], weights=v, minlength=len(n))
            par[f"den_{nom}_{prof}"] = np.bincount(ids[ok], minlength=len(n))
        print(f"  {reg} {prof} : {ok[dans].mean():.2f} des cellules", flush=True)
    proj = f"{_paths.PLATFORMS_ROOT}/LN24HA/{reg.upper()}_LN24HA_2020"
    out = []
    for tr in _parse_troncon(Path(f"{proj}/physitel/troncon.trl")):
        u = [x for x in tr["uhrh_ids"] if x < len(n) and n[x] > 0]
        if not u:
            continue
        row = {"region": reg, "troncon": int(tr["id"]), "aire_m2": float(n[u].sum()) * 1e4}
        for prof in PROFONDEURS:
            for nom in ("ilr_1", "ilr_2"):
                d = par[f"den_{nom}_{prof}"][u].sum()
                row[f"texture_{nom}_{prof}"] = float(par[f"num_{nom}_{prof}"][u].sum() / d) if d > 0 else np.nan
                row[f"couv_texture_{nom}_{prof}"] = float(d / n[u].sum())
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    regs = [a.lower() for a in sys.argv[1:]] or ["mont"]
    d = pd.concat([une_region(r) for r in regs], ignore_index=True)
    f = f"{_paths.DATA_ROOT}/derives/auxiliaires/soilgrids-troncons.parquet"
    if os.path.exists(f):
        a = pd.read_parquet(f)
        d = pd.concat([a[~a.region.isin(d.region.unique())], d], ignore_index=True)
    d.to_parquet(f, index=False)
    print(f"{f} : {len(d)} tronçons")
