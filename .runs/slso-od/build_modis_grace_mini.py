"""Peuple les tables modis_et (ETR MOD16A2 8-jours) et grace_tws (TWS GRACE)
dans le DB du mini-banc HydroSHEDS, à partir de l'extraction du DB open-data
complet (6166 nœuds).

- grace_tws : observation basin-wide (date, tws_mm, uncertainty, quality_ok) →
  copie directe (identique pour tout le domaine SLSO).
- modis_et  : observation par-nœud. Le mini (384 nœuds) est un réseau distinct
  du complet (distance au plus proche ~1.4 km médian ; MODIS = 500 m). On
  transfère chaque série de nœud par PLUS-PROCHE-VOISIN spatial. Approximation
  défendable pour un bilan ET agrégé domaine ; une ré-extraction exacte par nœud
  depuis les granules (WSL/pyhdf) reste possible si le terme de loss l'exige.

  python .runs/slso-od/build_modis_grace_mini.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import duckdb
import numpy as np

FULL = ".runs/slso-od/data/basin.duckdb"
MINI = ".runs/slso-od/data/basin_hs_mini.duckdb"

full = duckdb.connect(FULL, read_only=True)
fc = full.execute("SELECT node_idx, lon, lat FROM nodes ORDER BY node_idx").fetchnumpy()
f_idx = fc["node_idx"]; f_lon = fc["lon"].astype(np.float64); f_lat = fc["lat"].astype(np.float64)
modis = full.execute(
    "SELECT node_idx, date, etr_mm_day, quality_ok FROM modis_et").df()
grace = full.execute(
    "SELECT date, tws_mm, uncertainty, quality_ok FROM grace_tws ORDER BY date").df()
full.close()

mini = duckdb.connect(MINI)
mc = mini.execute("SELECT node_idx, lon, lat FROM nodes ORDER BY node_idx").fetchnumpy()
m_idx = mc["node_idx"]; m_lon = mc["lon"].astype(np.float64); m_lat = mc["lat"].astype(np.float64)

# ── Mapping mini → full par plus-proche-voisin (lon/lat) ──
nearest_full = np.empty(len(m_idx), dtype=np.int64)
dist = np.empty(len(m_idx), dtype=np.float64)
for i in range(len(m_idx)):
    d = np.sqrt((f_lon - m_lon[i]) ** 2 + (f_lat - m_lat[i]) ** 2)
    j = int(np.argmin(d)); nearest_full[i] = f_idx[j]; dist[i] = d[j]
print(f"mapping {len(m_idx)} nœuds mini → full | dist deg med {np.median(dist):.4f} "
      f"max {dist.max():.4f} (~{dist.max()*111:.1f} km)", flush=True)

# Table de correspondance : full_node_idx → mini_node_idx (peut être 1-à-plusieurs)
map_df = __import__("pandas").DataFrame(
    {"mini_idx": m_idx, "full_idx": nearest_full})

# ── modis_et du mini = série du full voisin, ré-étiquetée mini_idx ──
mini.execute("DROP TABLE IF EXISTS modis_et")
mini.execute(
    "CREATE TABLE modis_et (date DATE, node_idx INTEGER, etr_mm_day FLOAT, "
    "quality_ok BOOLEAN)")
mini.register("modis_full", modis)
mini.register("nn_map", map_df)
mini.execute(
    "INSERT INTO modis_et "
    "SELECT m.date, n.mini_idx AS node_idx, m.etr_mm_day, m.quality_ok "
    "FROM modis_full m JOIN nn_map n ON m.node_idx = n.full_idx")
n_modis = mini.execute("SELECT count(*) FROM modis_et").fetchone()[0]
rng = mini.execute("SELECT min(date), max(date) FROM modis_et").fetchone()
print(f"modis_et mini : {n_modis} lignes, {rng[0]}..{rng[1]}", flush=True)

# ── grace_tws : copie directe (basin-wide) ──
mini.execute("DROP TABLE IF EXISTS grace_tws")
mini.execute(
    "CREATE TABLE grace_tws (date DATE, tws_mm FLOAT, uncertainty FLOAT, "
    "quality_ok BOOLEAN)")
mini.register("grace_full", grace)
mini.execute("INSERT INTO grace_tws SELECT date, tws_mm, uncertainty, quality_ok FROM grace_full")
n_grace = mini.execute("SELECT count(*) FROM grace_tws").fetchone()[0]
rng = mini.execute("SELECT min(date), max(date) FROM grace_tws").fetchone()
print(f"grace_tws mini : {n_grace} lignes, {rng[0]}..{rng[1]}", flush=True)
mini.close()
print("DONE", flush=True)
