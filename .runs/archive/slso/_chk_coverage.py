import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, xarray as xr, duckdb
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache
h = BasinCache(".runs/slso/data/slso.duckdb").load(device="cpu"); nc = h["node_coords"].numpy()
g = xr.open_dataset(".runs/slso/data/casr/CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc")["rotated_pole"].attrs
rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude", "grid_north_pole_latitude": float(g["grid_north_pole_latitude"]), "grid_north_pole_longitude": float(g["grid_north_pole_longitude"]), "north_pole_grid_longitude": 0.0})
geo = CRS.from_proj4(f"+proj=longlat +R={float(g['earth_radius'])} +no_defs")
tr = Transformer.from_crs(geo, rp, always_xy=True)
rl, ra = tr.transform(nc[:, 0], nc[:, 1])
RLON0, RLON1, RLAT0, RLAT1 = 11.853, 14.913, -12.600, -6.390
oe = int((rl > RLON1).sum()); ow = int((rl < RLON0).sum()); on = int((ra > RLAT1).sum()); os_ = int((ra < RLAT0).sum())
print(f"noeuds={len(nc)}")
print(f"hors EST (rlon>{RLON1}): {oe} ({100*oe/len(nc):.1f}%), max rlon {rl.max():.2f}, deficit {rl.max()-RLON1:.2f}deg = {(rl.max()-RLON1)/0.09:.0f} mailles")
print(f"hors OUEST={ow} NORD={on} SUD={os_}")
c = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True); ni = c.execute("SELECT node_idx FROM stations").fetchdf()["node_idx"].astype(int).values; c.close()
print(f"stations en zone non couverte (est): {int((rl[ni]>RLON1).sum())}/{len(ni)}")
