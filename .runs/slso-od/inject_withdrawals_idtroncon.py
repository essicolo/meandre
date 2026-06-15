"""Réinjecte les prélèvements/rejets en HONORANT l'assignation io-eau (IDTRONCON).

Le snap par défaut de meandre rattache chaque site au nœud le plus proche de sa
PRISE D'EAU (lon/lat du site), ce qui ignore le travail d'io-eau : reloc lac,
exclusions de masses non modélisées, corrections par toponymes — tout encodé
dans la colonne IDTRONCON du parquet (tronçon PHYSITEL soigneusement choisi).

Ici on transfère par géométrie de tronçon : pour chaque site, on prend la
position (X, Y) de SON IDTRONCON dans troncons.parquet, et on la snappe au nœud
open data le plus proche. Le prélèvement atterrit donc sur le nœud correspondant
au tronçon assigné par io-eau, pas à la prise d'eau brute.

  python .runs/slso-od/inject_withdrawals_idtroncon.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import duckdb

from meandre.data.basin_cache import BasinCache

BASIN_DB = Path(".runs/slso-od/data/basin.duckdb")
WD_PARQUET = Path(".runs/slso/data/io-eau-meandre.parquet")
TRONCONS = Path("C:/Users/parse01/documents-locaux/GitHub/io-eau/data/source/troncons.parquet")
MAX_SNAP_KM = 5.0   # un IDTRONCON hors réseau open data au-delà = site exclu

wd = pd.read_parquet(WD_PARQUET)
# ATTENTION : les colonnes X,Y de troncons.parquet sont arrondies au degré
# entier (inutilisables). On prend le point représentatif de la géométrie.
trc = gpd.read_parquet(TRONCONS)
trc = trc.dropna(subset=["IDTRONCON"]).drop_duplicates("IDTRONCON")
rp = trc.geometry.representative_point()
trc_xy = pd.DataFrame({"X": rp.x.values, "Y": rp.y.values}, index=trc["IDTRONCON"].values)

# Nœuds open data
con = duckdb.connect(str(BASIN_DB), read_only=True)
nodes = con.execute("SELECT node_idx, lon, lat, is_lake FROM nodes ORDER BY node_idx").fetchdf()
con.close()
nlon = nodes["lon"].to_numpy(); nlat = nodes["lat"].to_numpy()
nidx = nodes["node_idx"].to_numpy()
lat0 = float(np.median(nlat)); kx = 111.320 * np.cos(np.radians(lat0)); ky = 110.574

# IDTRONCON présents dans les withdrawals → position du tronçon → nœud le plus proche
wd_ids = wd["IDTRONCON"].dropna().unique()
id2node = {}
n_far = 0
for tid in wd_ids:
    if tid not in trc_xy.index:
        continue
    x, y = trc_xy.loc[tid, "X"], trc_xy.loc[tid, "Y"]
    d = np.hypot((nlon - x) * kx, (nlat - y) * ky)
    j = int(np.argmin(d))
    if d[j] > MAX_SNAP_KM:
        n_far += 1
        continue
    id2node[tid] = int(nidx[j])

print(f"IDTRONCON withdrawals : {len(wd_ids)} ; appariés à un nœud : {len(id2node)} "
      f"; hors réseau (>{MAX_SNAP_KM} km) : {n_far}")

# Assignation : chaque ligne reçoit le node_idx de SON IDTRONCON
wd2 = wd.dropna(subset=["IDTRONCON"]).copy()
wd2["node_idx"] = wd2["IDTRONCON"].map(id2node)
wd2 = wd2.dropna(subset=["node_idx"])
wd2["node_idx"] = wd2["node_idx"].astype(int)
n_sites = wd2["site_id"].nunique()
n_nodes_hit = wd2["node_idx"].nunique()
print(f"lignes retenues : {len(wd2):,} ; sites : {n_sites} ; nœuds touchés : {n_nodes_hit}")

# Format attendu par import_withdrawals avec node_idx pré-assigné
out = wd2[["date", "node_idx", "net_withdrawal", "source"]].copy()
tmp = Path(".runs/slso-od/data/_withdrawals_idtroncon.parquet")
out.to_parquet(tmp, index=False)

# Vider les prélèvements coordonnée-snappés du build (sinon double comptage :
# import_withdrawals fait INSERT OR REPLACE par (date,node_idx), il ne purge pas
# les lignes sur d'autres nœuds).
con = duckdb.connect(str(BASIN_DB))
con.execute("DROP TABLE IF EXISTS withdrawals")
con.close()

cache = BasinCache(str(BASIN_DB))
n = cache.import_withdrawals(tmp, net_col="net_withdrawal", node_col="node_idx",
                            source_col="source")
print(f"INJECTÉ : {n} lignes (par node_idx, honore l'assignation io-eau)")
tmp.unlink()
print("INJECT_DONE", flush=True)
