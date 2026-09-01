"""Ingestion couvert nival MOD10A1 QUOTIDIEN (fenêtre de fonte 1er mars - 30 juin)
pour UNE région Québec — phase 2 du design_modules_appris.md (banc fonte).
Cache granules PARTAGÉ D:/meandre-data/modis10 (tuiles communes entre régions).
WSL requis (pyhdf). Resumable par année (upsert INSERT OR REPLACE).
  python .runs/quebec/ingest_snow_region.py GASP
"""
import os, sys
from pathlib import Path
os.chdir(Path(__file__).resolve().parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception: pass
import torch, duckdb, numpy as np
from meandre.data.basin_cache import BasinCache
from meandre.data.modis_loader import fetch_modis_snow_daily

REG = sys.argv[1].lower()
import platform

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")
_D = "/mnt/d" if platform.system() == "Linux" else "D:"
BASIN_DB = f"{_D}/meandre-data/quebec/{REG}.duckdb" if REG != "slso" else ".runs/slso/data/slso.duckdb"
CACHE_DIR = f"{_D}/meandre-data/modis10"
YEAR_START, YEAR_END = 2000, 2024

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
nc = hydro["node_coords"].cpu().numpy(); n_nodes = len(nc)
bbox = (float(nc[:,0].min())-0.1, float(nc[:,1].min())-0.1, float(nc[:,0].max())+0.1, float(nc[:,1].max())+0.1)
print(f"[{REG}] {n_nodes} nœuds | bbox {tuple(round(b,2) for b in bbox)}", flush=True)

# reprise : années déjà complètes (>= 100 dates dans la fenêtre) sautées
con = duckdb.connect(BASIN_DB, read_only=True)
done = {}
if "modis_snow" in [t[0] for t in con.execute("show tables").fetchall()]:
    for y, n in con.execute("select year(date), count(distinct date) from modis_snow "
                            "where month(date) between 3 and 6 group by 1").fetchall():
        done[int(y)] = int(n)
con.close()

for year in range(YEAR_START, YEAR_END + 1):
    if done.get(year, 0) >= 100:
        print(f"[{REG}] {year}: déjà complet ({done[year]} jours), sauté", flush=True); continue
    df = fetch_modis_snow_daily(bbox, f"{year}-03-01", f"{year}-06-30", nc, np.arange(n_nodes), cache_dir=CACHE_DIR)
    if df.empty:
        print(f"[{REG}] {year}: AUCUN granule", flush=True); continue
    n = cache.import_modis_snow(df)
    print(f"[{REG}] {year}: {len(df):,} lignes ({df['date'].nunique()} jours, {df['quality_ok'].mean():.0%} sans nuage)", flush=True)

# vérification bruyante de complétude
con = duckdb.connect(BASIN_DB, read_only=True)
tot, val, nn, nd = con.execute("""SELECT COUNT(*), COUNT(*) FILTER (snow_frac IS NOT NULL),
    COUNT(DISTINCT node_idx) FILTER (snow_frac IS NOT NULL), COUNT(DISTINCT date)
    FROM modis_snow WHERE month(date) BETWEEN 3 AND 6""").fetchone()
con.close()
ok = nd >= 2500 and nn >= 0.9 * n_nodes
print(f"[{REG}] BILAN fonte : {tot:,} lignes | {val:,} sans nuage | {nn}/{n_nodes} nœuds | {nd} jours -> {'COMPLET' if ok else 'INCOMPLET !!'}", flush=True)
sys.exit(0 if ok else 1)
