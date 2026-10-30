"""Fractions d'occupation du sol ESA WorldCover 2021 (10 m) par UHRH d'un projet PHYSITEL.

Pourquoi. La classe « imperméable » de PHYSITEL est une classe d'usage urbain (5,4 % de la
Montérégie), dont meandre envoie toute la pluie à la rivière le jour même ; cette voie aplatit la
réponse aux averses (R454). WorldCover sépare le bâti (bâtiments, routes, surfaces artificielles)
des pelouses et des arbres à 10 m : c'est une mesure indépendante à comparer.

Méthode. Chaque tuile est lue par bandes de lignes ; les classes sont moyennées par blocs de 10 x 10
pixels (environ 100 m), puis chaque bloc est rattaché à l'UHRH qui contient son centre (raster
uhrh.tif du projet). Sortie : un parquet, une ligne par UHRH, fraction de chaque classe.

Usage : python worldcover_uhrh.py <projet PHYSITEL> <dossier des tuiles> <sortie.parquet>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
import rasterio.warp
import rasterio.windows
from pyproj import Transformer

CLASSES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 70: "snow", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss"}
BLOCK = 10


def main(project, tiles_dir, output):
    uh = rasterio.open(Path(project) / "physitel" / "uhrh.tif")
    ids = uh.read(1)
    to_utm = Transformer.from_crs("EPSG:4326", uh.crs, always_xy=True)
    west, south, east, north = rasterio.warp.transform_bounds(uh.crs, "EPSG:4326", *uh.bounds)
    sums = {}
    counts = {}
    for tif in sorted(Path(tiles_dir).glob("ESA_WorldCover_*_Map.tif")):
        with rasterio.open(tif) as src:
            win = rasterio.windows.from_bounds(west, south, east, north, src.transform).round_offsets().round_lengths()
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            row0, col0 = int(win.row_off), int(win.col_off)
            n_rows = (int(win.height) // BLOCK) * BLOCK
            n_cols = (int(win.width) // BLOCK) * BLOCK
            for r in range(0, n_rows, 1000):
                h = min(1000, n_rows - r)
                a = src.read(1, window=rasterio.windows.Window(col0, row0 + r, n_cols, h))
                hb, wb = h // BLOCK, n_cols // BLOCK
                blocks = a[:hb * BLOCK].reshape(hb, BLOCK, wb, BLOCK)
                # centres des blocs en coordonnées géographiques puis dans la grille des UHRH
                rows = row0 + r + np.arange(hb) * BLOCK + BLOCK / 2
                cols = col0 + np.arange(wb) * BLOCK + BLOCK / 2
                lon = src.transform.c + cols * src.transform.a
                lat = src.transform.f + rows * src.transform.e
                LON, LAT = np.meshgrid(lon, lat)
                x, y = to_utm.transform(LON.ravel(), LAT.ravel())
                rr, cc = rasterio.transform.rowcol(uh.transform, x, y)
                rr, cc = np.asarray(rr), np.asarray(cc)
                inside = (rr >= 0) & (rr < ids.shape[0]) & (cc >= 0) & (cc < ids.shape[1])
                uid = np.zeros(rr.shape, dtype=np.int64)
                uid[inside] = ids[rr[inside], cc[inside]]
                keep = uid > 0
                if not keep.any():
                    continue
                valid = (blocks > 0).sum(axis=(1, 3)).ravel()
                for code, name in CLASSES.items():
                    frac_count = (blocks == code).sum(axis=(1, 3)).ravel()
                    s = np.bincount(uid[keep], weights=frac_count[keep], minlength=1)
                    sums.setdefault(name, np.zeros(0))
                    if len(sums[name]) < len(s):
                        sums[name] = np.pad(sums[name], (0, len(s) - len(sums[name])))
                    sums[name][:len(s)] += s
                c = np.bincount(uid[keep], weights=valid[keep], minlength=1)
                counts.setdefault("n", np.zeros(0))
                if len(counts["n"]) < len(c):
                    counts["n"] = np.pad(counts["n"], (0, len(c) - len(counts["n"])))
                counts["n"][:len(c)] += c
    n = counts["n"]
    size = max(len(n), *(len(v) for v in sums.values()))
    n = np.pad(n, (0, size - len(n)))
    out = pd.DataFrame({"uhrh": np.arange(size)})
    for name, v in sums.items():
        v = np.pad(v, (0, size - len(v)))
        out[f"wc_{name}"] = np.where(n > 0, v / np.maximum(n, 1), np.nan)
    out["wc_pixels"] = n
    out = out[out.wc_pixels > 0]
    out.to_parquet(output)
    print(f"{len(out)} UHRH, bâti moyen {out.wc_built.mean():.3%}, pondéré {np.average(out.wc_built, weights=out.wc_pixels):.3%}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
