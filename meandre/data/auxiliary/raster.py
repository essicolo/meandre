"""Sommes zonales d'un raster par unité hydrologique.

Les unités sont rasterisées sur la grille du raster, dans la fenêtre qui couvre la région. Une
unité plus petite qu'un pixel, qui ne reçoit aucun centre de pixel, prend la valeur du pixel
sous son point représentatif, pondérée par sa propre aire.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class UnitGrid:
    """Identifiants des unités hydrologiques sur la grille d'un raster, pour une région."""

    def __init__(self, units, dataset):
        from rasterio import features, windows

        self.units = units.to_crs(dataset.crs)
        gauche, bas, droite, haut = self.units.total_bounds
        # Colonnes et lignes des quatre coins : certains rasters ont l'axe des lignes vers le nord.
        inv = ~dataset.transform
        cols, rows = zip(*[inv * (x, y) for x in (gauche, droite) for y in (bas, haut)])
        c0, r0 = int(np.floor(min(cols))), int(np.floor(min(rows)))
        fen = windows.Window(c0, r0, int(np.ceil(max(cols))) - c0 + 1, int(np.ceil(max(rows))) - r0 + 1)
        self.window = fen.intersection(windows.Window(0, 0, dataset.width, dataset.height))
        self.transform = dataset.window_transform(self.window)
        shape = (int(self.window.height), int(self.window.width))
        self.ids = features.rasterize(zip(self.units.geometry, self.units.uhrh), out_shape=shape, transform=self.transform, fill=0, dtype="int32")
        self.pixel_area = abs(dataset.transform.a * dataset.transform.e)
        self.index = self.units.uhrh.to_numpy()
        presents = np.unique(self.ids)
        self.small = self.units[~self.units.uhrh.isin(presents)]
        self.small_points = [(p.x, p.y) for p in self.small.geometry.representative_point()]
        self.small_area = self.small.geometry.area.to_numpy()

    def read(self, dataset, band: int) -> np.ndarray:
        a = dataset.read(band, window=self.window, masked=True).astype("float64")
        return np.ma.filled(a, np.nan)

    def sample(self, dataset, band: int) -> np.ndarray:
        if not self.small_points:
            return np.array([])
        v = np.array([s[0] for s in dataset.sample(self.small_points, indexes=band, masked=True)], dtype=object)
        return np.array([np.nan if (x is np.ma.masked or x is None) else float(x) for x in v])

    def zonal(self, values: np.ndarray, small_values: np.ndarray) -> tuple[pd.Series, pd.Series]:
        """Somme de valeur × aire et aire valide, par unité."""
        ok = np.isfinite(values) & (self.ids > 0)
        n = int(self.index.max()) + 1
        num = np.bincount(self.ids[ok], weights=values[ok], minlength=n) * self.pixel_area
        den = np.bincount(self.ids[ok], minlength=n).astype(float) * self.pixel_area
        num_s = pd.Series(num[self.index], index=self.index)
        den_s = pd.Series(den[self.index], index=self.index)
        for uid, v, a in zip(self.small.uhrh.to_numpy(), small_values, self.small_area):
            if np.isfinite(v):
                num_s.loc[uid] += v * a
                den_s.loc[uid] += a
        return num_s, den_s
