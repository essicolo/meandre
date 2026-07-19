"""Ingestion ETR MOD16A2GF 8-jours pour UNE région Québec (généralise ingest_modis_et8).
Cache granules PARTAGÉ sur D: (les régions se recouvrent en tuiles MODIS).
Purge l'annuel A3GF avant d'importer le 8-jours. Resumable par année.
  python .runs/quebec/ingest_et8_region.py GASP
"""
import os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
import torch, duckdb, numpy as np
from meandre.data.basin_cache import BasinCache
from meandre.data.modis_loader import fetch_modis_et_8day

REG = sys.argv[1].lower()
BASIN_DB = f"D:/meandre-data/quebec/{REG}.duckdb"
CACHE_DIR = "D:/meandre-data/modis8"
YEAR_START, YEAR_END = 2000, 2024

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
nc = hydro["node_coords"].cpu().numpy(); n_nodes = len(nc)
bbox = (float(nc[:,0].min())-0.1, float(nc[:,1].min())-0.1, float(nc[:,0].max())+0.1, float(nc[:,1].max())+0.1)
print(f"[{REG}] {n_nodes} nœuds | bbox {tuple(round(b,2) for b in bbox)}", flush=True)
con = duckdb.connect(BASIN_DB)
tabs = [t[0] for t in con.execute("show tables").fetchall()]
if "modis_et" in tabs:
    has8 = con.execute("select count(*) from modis_et where not (month(date)=1 and day(date)=1)").fetchone()[0]
    if has8 == 0:
        n = con.execute("select count(*) from modis_et").fetchone()[0]
        con.execute("DELETE FROM modis_et")
        print(f"[{REG}] table annuelle purgée ({n} lignes)", flush=True)
con.close()
for year in range(YEAR_START, YEAR_END + 1):
    df = fetch_modis_et_8day(bbox, f"{year}-01-01", f"{year}-12-31", nc, np.arange(n_nodes), cache_dir=CACHE_DIR)
    if df.empty:
        print(f"[{REG}] {year}: AUCUN granule", flush=True); continue
    n = cache.import_modis_et(df)
    print(f"[{REG}] {year}: {len(df):,} lignes ({df['date'].nunique()} composites, {df['quality_ok'].mean():.0%} qualité)", flush=True)
# vérification bruyante de complétude
con = duckdb.connect(BASIN_DB, read_only=True)
tot, val, nn, ncomp = con.execute("""SELECT COUNT(*), COUNT(*) FILTER (etr_mm_day IS NOT NULL AND etr_mm_day > 0),
    COUNT(DISTINCT node_idx) FILTER (etr_mm_day IS NOT NULL), COUNT(DISTINCT date) FROM modis_et""").fetchone()
con.close()
ok = ncomp >= 1000 and nn >= 0.9 * n_nodes
print(f"[{REG}] BILAN : {tot:,} lignes | {val:,} valides | {nn}/{n_nodes} nœuds | {ncomp} composites -> {'COMPLET' if ok else 'INCOMPLET !!'}", flush=True)
sys.exit(0 if ok else 1)
