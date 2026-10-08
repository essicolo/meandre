"""Ingestion GRACE/GRACE-FO (stock d'eau total, mensuel) pour UN territoire quebecois.

Meme chargeur que le banc SLSO (`meandre.data.grace_loader.fetch_grace_tws`), sur l'emprise des
noeuds de la base, table `grace_tws` de la base DuckDB du territoire. Identifiants Earthdata
par ~/.netrc ou EARTHDATA_TOKEN. Ecrit le 2026-10-08 parce que la base du Saint-Laurent
sud-ouest n'avait ni MOD16 ni GRACE et a ete entrainee sans eux (registre R331).

    python .runs/quebec/ingest_grace_region.py SLSO
"""
import os
import platform
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch

from meandre.data.basin_cache import BasinCache
from meandre.data.grace_loader import fetch_grace_tws

REG = sys.argv[1].lower()
_D = "/mnt/d" if platform.system() == "Linux" else "D:"
BASIN_DB = f"{_D}/meandre-data/quebec/{REG}.duckdb"
DATE_START, DATE_END = "2002-04-01", "2026-06-01"

cache = BasinCache(BASIN_DB)
hydro = cache.load(device=torch.device("cpu"))
nc = hydro["node_coords"].cpu().numpy()
bbox = (float(nc[:, 0].min()) - 0.1, float(nc[:, 1].min()) - 0.1, float(nc[:, 0].max()) + 0.1, float(nc[:, 1].max()) + 0.1)
print(f"[{REG}] {len(nc)} noeuds | emprise {tuple(round(b, 2) for b in bbox)}", flush=True)
if cache.has_grace_tws() and os.environ.get("GRACE_ECRASER", "0") != "1":
    print(f"[{REG}] table grace_tws deja presente ; GRACE_ECRASER=1 pour la refaire")
    sys.exit(0)
df = fetch_grace_tws(bbox, DATE_START, DATE_END)
if df.empty:
    print(f"[{REG}] aucune donnee GRACE recuperee")
    sys.exit(1)
n = cache.import_grace_tws(df)
print(f"[{REG}] {n} mois importes, {df['date'].min()} a {df['date'].max()}, TWS {df['tws_mm'].min():.0f} a {df['tws_mm'].max():.0f} mm")
