"""QUÉBEC étape 1 : construit le cache DuckDB de chaque région PHYSITEL + importe les
observations (stations_concatenees.nc, préfixe par région) + calcule les tuiles CaSR
nécessaires (transform pôle tourné, blocs de 35 indices, origine 35k+1).
Sorties : D:/meandre-data/quebec/<reg>.duckdb + tiles_needed.txt (union pour fetch).
  python .runs/quebec/build_regions.py [REG ...]   (défaut : les 15)
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, xarray as xr
from pathlib import Path
from pyproj import CRS, Transformer
from meandre.data.basin_cache import BasinCache

PLATEFORMES = Path("C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA")
STATIONS = Path(r"C:\Users\parse01\documents-locaux\rqh-local\rqh_2026-04\data\07_stations\stations_concatenees.nc")
OUT = Path("D:/meandre-data/quebec"); OUT.mkdir(parents=True, exist_ok=True)
REGIONS = sys.argv[1:] or ["GASP", "VAUD", "MONT", "SLSO", "SLNO", "SAGU", "LABI", "ABIT",
                           "OUTM", "OUTV", "CNDA", "CNDB", "CNDC", "CNDD", "CNDE"]

# grille pôle tourné CaSR (attrs depuis une tuile SLSO existante)
_ref = xr.open_dataset(".runs/slso/data/casr/CaSR_v3.2_A_TT_1.5m_rlon526-560_rlat351-385_2000-2003.nc")
_g = _ref["rotated_pole"].attrs
rlon_vals = _ref.rlon.values; rlat_vals = _ref.rlat.values; _ref.close()
# pas et origine : index i (1-based) -> valeur ; tuile rlon526-560 = indices 526..560
dr_lon = float(np.diff(rlon_vals).mean()); dr_lat = float(np.diff(rlat_vals).mean())
rlon0 = rlon_vals[0] - 526 * dr_lon   # valeur à l'index 0 (virtuel)
rlat0 = rlat_vals[0] - 351 * dr_lat
rp = CRS.from_cf({"grid_mapping_name": "rotated_latitude_longitude",
    "grid_north_pole_latitude": float(_g["grid_north_pole_latitude"]),
    "grid_north_pole_longitude": float(_g["grid_north_pole_longitude"]),
    "north_pole_grid_longitude": float(_g.get("north_pole_grid_longitude", 0.0))})
geo = CRS.from_proj4(f"+proj=longlat +R={float(_g['earth_radius'])} +no_defs")
tr = Transformer.from_crs(geo, rp, always_xy=True)
def blocks(idx):
    k = (np.asarray(idx) - 1) // 35
    return [f"{35*int(b)+1}-{35*int(b)+35}" for b in sorted(set(k))]

all_tiles = set()
for reg in REGIONS:
    proj = PLATEFORMES / f"{reg}_LN24HA_2020"
    db = OUT / f"{reg.lower()}.duckdb"
    print(f"\n=== {reg} ===")
    if not proj.exists():
        print(f"  ABSENT : {proj}"); continue
    if db.exists():
        print(f"  cache déjà présent : {db}")
        cache = BasinCache(db)
    else:
        cache = BasinCache.from_hydrotel(project_dir=proj, path=db)
        try:
            cache.import_observations(STATIONS, basin_prefix=reg)
        except Exception as e:
            print(f"  [obs] échec import ({e}) — région sans jauge ?")
    h = cache.load(device="cpu")
    nc_ = h["node_coords"].numpy()
    rlon_n, rlat_n = tr.transform(nc_[:, 0], nc_[:, 1])
    i_lon = np.round((rlon_n - rlon0) / dr_lon).astype(int)
    i_lat = np.round((rlat_n - rlat0) / dr_lat).astype(int)
    tiles = [f"rlon{a}_rlat{b}" for a in blocks(i_lon) for b in blocks(i_lat)]
    # ATTENTION : produit cartésien = enveloppe ; suffisant pour le fetch
    all_tiles.update(tiles)
    print(f"  nœuds {h['n_nodes']} | tuiles : {tiles}")

with open(OUT / "tiles_needed.txt", "w") as f:
    f.write("\n".join(sorted(all_tiles)))
print(f"\nUNION : {len(all_tiles)} tuiles -> {OUT / 'tiles_needed.txt'}")
print(f"volume estimé : ~{len(all_tiles) * 2.5:.0f} Go (6 vars x 7 chunks x ~60 Mo)")
