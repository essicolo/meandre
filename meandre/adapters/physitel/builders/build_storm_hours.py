"""Durée effective d'averse par jour et par tronçon, tirée du CaSR horaire, pour un territoire.

Durée = pluie du jour divisée par sa plus forte heure, bornée entre 1 et 24 h ; heures sous
0,3 mm écartées, jour local (UTC − 5). Une averse convective qui tombe en une heure vaut 1 h,
une pluie frontale régulière 24 h. C'est la définition du canal DT_eff du constructeur de
forçage (`QC_INTENS`), calculée seule, sans refaire le forçage : elle sert de septième canal
au ruissellement hortonien (`column.hortonian`). Jours sans pluie : 24 h.

    python -m meandre.adapters.physitel.builders.build_storm_hours MONT [--test]

Sortie : <MEANDRE_DATA>/quebec/storm-hours-<territoire>.npy, (jours 2000-2024, tronçons), float32.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from meandre.data.basin_cache import BasinCache

DATA_ROOT = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]
DRIZZLE_MM_H = 0.3
SHIFT_H = -5
REF_TILE = "CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc"


def tile(name):
    path = f"{DATA_ROOT}/casr/{name}"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def node_grid(node_coords):
    """Coordonnées des tronçons dans la grille tournée du CaSR, et blocs de tuiles à lire."""
    ref = xr.open_dataset(tile(REF_TILE))
    g = ref["rotated_pole"].attrs
    rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude", "grid_north_pole_latitude": float(g["grid_north_pole_latitude"]), "grid_north_pole_longitude": float(g["grid_north_pole_longitude"]), "north_pole_grid_longitude": float(g.get("north_pole_grid_longitude", 0.0))})
    geo = CRS.from_proj4(f"+proj=longlat +R={float(g['earth_radius'])} +no_defs")
    rlon, rlat = Transformer.from_crs(geo, rp, always_xy=True).transform(node_coords[:, 0], node_coords[:, 1])
    step_lon, step_lat = float(np.diff(ref.rlon.values).mean()), float(np.diff(ref.rlat.values).mean())
    lon0, lat0 = ref.rlon.values[0] - 526 * step_lon, ref.rlat.values[0] - 351 * step_lat
    ref.close()

    def blocks(v, origin, step):
        k = sorted(set((np.round((v - origin) / step).astype(int) - 1) // 35))
        return [f"{35 * b + 1}-{35 * b + 35}" for b in k]
    return rlon, rlat, [f"rlon{b}" for b in blocks(rlon, lon0, step_lon)], [f"rlat{b}" for b in blocks(rlat, lat0, step_lat)]


def hourly_precip(rlon, rlat, lon_blocks, lat_blocks, chunk):
    """Précipitation horaire (mm/h) interpolée aux tronçons, indexée en heure locale."""
    rows, times = [], None
    for lat_b in lat_blocks:
        cols = []
        for lon_b in lon_blocks:
            ds = xr.open_dataset(tile(f"CaSR_v3.2_A_PR0_SFC_{lon_b}_{lat_b}_{chunk}.nc"))
            var = [x for x in ds.data_vars if "CaSR" in x][0]
            cols.append(ds[var])
            times = pd.to_datetime(ds.time.values)
        rows.append(xr.concat(cols, dim="rlon"))
    grid = xr.concat(rows, dim="rlat")
    values = grid.interp(rlon=xr.DataArray(rlon, dims="node"), rlat=xr.DataArray(rlat, dims="node"), method="linear").values
    return pd.DataFrame(values * 1000.0, index=times + pd.Timedelta(hours=SHIFT_H))


def main():
    reg = sys.argv[1].lower()
    test = "--test" in sys.argv
    h = BasinCache(f"{DATA_ROOT}/quebec/{reg}.duckdb").load(device="cpu")
    rlon, rlat, lon_blocks, lat_blocks = node_grid(h["node_coords"].numpy())
    print(f"[{reg}] {h['n_nodes']} tronçons, tuiles {lon_blocks} x {lat_blocks}")
    totals, peaks = [], []
    for chunk in CHUNKS[:1] if test else CHUNKS:
        t0 = time.perf_counter()
        kept = hourly_precip(rlon, rlat, lon_blocks, lat_blocks, chunk)
        kept = kept.where(kept >= DRIZZLE_MM_H, 0.0).resample("1D")
        totals.append(kept.sum())
        peaks.append(kept.max())
        print(f"[{reg}] {chunk} : {time.perf_counter() - t0:.0f} s", flush=True)
    total = pd.concat(totals).groupby(level=0).sum()
    peak = pd.concat(peaks).groupby(level=0).max()
    hours = (total / peak.where(peak > 0.0)).clip(lower=1.0, upper=24.0).fillna(24.0)
    days = pd.date_range("2000-01-01", "2003-12-31" if test else "2024-12-31", freq="D")
    out = hours.reindex(days).fillna(24.0).to_numpy(dtype=np.float32)
    wet = total.reindex(days).to_numpy() > 1.0
    print(f"[{reg}] jours de plus de 1 mm : durée médiane {np.median(out[wet]):.1f} h, part sous 3 h {100 * np.mean(out[wet] < 3.0):.0f} %")
    if test:
        return
    path = f"{DATA_ROOT}/quebec/storm-hours-{reg}.npy"
    np.save(path, out)
    print(f"[{reg}] {path} {out.shape}")


if __name__ == "__main__":
    main()
