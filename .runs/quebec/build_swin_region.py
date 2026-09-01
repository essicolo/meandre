"""Canal sw_in (FB, courte longueur d'onde incidente, W/m²) par région.

Pourquoi (R42, accord d'Essi 2026-08-23 : « démarre dès la fin de la flotte »). Le mode
de fonte ETI (radiation réelle, implémenté et validé en juin) dissout dans une variable
physique du forçage les deux derniers scalaires nivaux de la recette : l'amplitude
saisonnière (substitut du cycle radiatif, fausse en climat maritime — suspect n° 1 du
manteau gaspésien à 2x) et une part du degré-jour lui-même. Même mouvement que le bulbe
humide pour le seuil pluie-neige.

Les forçages québécois ont 6 canaux (P, Tmin, Tmax, R_n, u2, e_a) : R_n est DÉRIVÉ de
FB mais FB seul n'y est pas, et la colonne ETI lit le canal 6. Ce script extrait FB des
mêmes tuiles CaSR, par la même mosaïque et la même interpolation bilinéaire en pôle
tourné que build_forcing_region.py, et écrit forcing-<reg>-swin.nc (time, node). Le
pilote l'injecte dans le canal 6 (libre sous ETL_ETP=linacre, où la demande apprise
est ignorée) via ETL_ETI=1.

    .venv/Scripts/python.exe .runs/quebec/build_swin_region.py OUTV
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as _paths

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

REG = (sys.argv[1] if len(sys.argv) > 1 else "OUTV").upper()
DB = _paths.data_path("quebec", f"{REG.lower()}.duckdb")
OUT = _paths.data_path("quebec", f"forcing-{REG.lower()}-swin.nc")
CASR_DIRS = [f"{_DATA_ROOT}/casr", ".runs/slso/data/casr"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]


def casr_path(fn):
    for d in CASR_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(fn)


h = BasinCache(DB).load(device="cpu")
nc_ = h["node_coords"].numpy()
n_nodes = h["n_nodes"]

# CRS pôle tourné + mosaïque : à l'identique de build_forcing_region.py
_g = xr.open_dataset(casr_path("CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc"))["rotated_pole"].attrs
rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0))})
geo = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
nrlon, nrlat = Transformer.from_crs(geo, rp, always_xy=True).transform(nc_[:, 0], nc_[:, 1])
_ref = xr.open_dataset(casr_path("CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc"))
dr_lon = float(np.diff(_ref.rlon.values).mean()); dr_lat = float(np.diff(_ref.rlat.values).mean())
rlon0 = _ref.rlon.values[0] - 526 * dr_lon; rlat0 = _ref.rlat.values[0] - 351 * dr_lat
_ref.close()

def blocks(vals, orig, step):
    k = sorted(set((np.round((vals - orig) / step).astype(int) - 1) // 35))
    return [f"{35*b+1}-{35*b+35}" for b in k]

RLON_BLOCKS = [f"rlon{b}" for b in blocks(nrlon, rlon0, dr_lon)]
RLAT_BLOCKS = [f"rlat{b}" for b in blocks(nrlat, rlat0, dr_lat)]
print(f"[{REG}] {n_nodes} noeuds | mosaique {RLON_BLOCKS} x {RLAT_BLOCKS}")
nrlon_da = xr.DataArray(nrlon, dims="node"); nrlat_da = xr.DataArray(nrlat, dims="node")

def _axes():
    rlons, rlats = [], []
    for rb in RLON_BLOCKS:
        d = xr.open_dataset(casr_path(f"CaSR_v3.2_A_TT_1.5m_{rb}_{RLAT_BLOCKS[0]}_2000-2003.nc")); rlons.append(d.rlon.values); d.close()
    for ab in RLAT_BLOCKS:
        d = xr.open_dataset(casr_path(f"CaSR_v3.2_A_TT_1.5m_{RLON_BLOCKS[0]}_{ab}_2000-2003.nc")); rlats.append(d.rlat.values); d.close()
    return np.concatenate(rlons), np.concatenate(rlats)

RLON, RLAT = _axes()

def hourly_chunk(var, ch):
    times = None; rows = []
    for ab in RLAT_BLOCKS:
        cols = []
        for rb in RLON_BLOCKS:
            ds = xr.open_dataset(casr_path(f"CaSR_v3.2_{var}_{rb}_{ab}_{ch}.nc"))
            v = [x for x in ds.data_vars if "CaSR" in x][0]
            cols.append(ds[v].values); times = pd.to_datetime(ds.time.values); ds.close()
        rows.append(np.concatenate(cols, axis=2))
    da = xr.DataArray(np.concatenate(rows, axis=1), dims=("time", "rlat", "rlon"),
                      coords={"time": times, "rlat": RLAT, "rlon": RLON})
    return da.interp(rlon=nrlon_da, rlat=nrlat_da, method="linear"), times

parts = []
for ch in CHUNKS:
    print(f"  FB {ch}...")
    samp, times = hourly_chunk("P_FB_SFC", ch)
    parts.append(pd.DataFrame(samp.values, index=times).resample("1D").mean())
FB = pd.concat(parts)
FB = FB.groupby(FB.index).mean()
FB = FB.loc["2000-01-01":"2024-12-31"]
idx = pd.date_range("2000-01-01", "2024-12-31", freq="D")
FB = FB.reindex(idx).interpolate(limit=3)
arr = FB.to_numpy(dtype=np.float32)
assert arr.shape == (len(idx), n_nodes), arr.shape
print(f"[{REG}] FB journalier : moy {np.nanmean(arr):.1f} W/m2 | "
      f"jan {np.nanmean(arr[idx.month == 1]):.0f} | jul {np.nanmean(arr[idx.month == 7]):.0f}")
xr.DataArray(arr[:, :, None], dims=("time", "node", "var"),
             coords={"time": idx, "node": np.arange(n_nodes), "var": ["sw_in"]},
             name="forcing").to_netcdf(OUT)
print(f"[{REG}] ecrit : {OUT}")
