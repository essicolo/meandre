#!/usr/bin/env python
"""Enrich SLSO territorial table with SIGEOM geology fractions.

Adds bedrock and quaternary deposit one-hot columns to the DuckDB so that
the NeRF spatial encoder can differentiate geological regions (e.g. Canadian
Shield vs Appalachians).

Usage
-----
    python notebooks/slso/enrich_geology.py
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Ensure we run from the repo root
os.chdir(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
)

from meandre.data.geology_enrichment import enrich_territorial_with_geology

DB = Path("notebooks/slso/data/slso.duckdb")
GPKG_DIR = Path(".gpkg")

# -----------------------------------------------------------------------
# 1. Bedrock geology (F3E04_ZONE_GEOLOGIQUE)
#    NOM_ABRG_ETQT_LITH has detailed lithology; coarsen to first character
#    (I=intrusive, M=metamorphic, V=volcanic, S=sedimentary)
# -----------------------------------------------------------------------
bedrock_cols = enrich_territorial_with_geology(
    db_path=DB,
    gpkg_path=GPKG_DIR / "SIGEOM_QC_Bedrock_geology_GPKG" / "sigeom.gpkg",
    layer="F3E04_ZONE_GEOLOGIQUE",
    class_col="NOM_ABRG_ETQT_LITH",
    prefix="bedrock",
    top_k=4,
    coarsen="x[:1] if x else 'unknown'",
)
print(f"\nBedrock columns added: {bedrock_cols}")

# -----------------------------------------------------------------------
# 2. Quaternary deposits (F10E15_ZONE_MORPH_SEDIM)
#    CODE_DEPOT_MORP_SEDM has 59 codes; keep top 8 as-is.
#    O=organic, L=lacustrine, Tm=marine, Tc=colluvial, R=rock,
#    Gx=glaciofluvial, Ri=alluvial, Tr=fluvial, Go=glaciolacustrine
# -----------------------------------------------------------------------
quat_cols = enrich_territorial_with_geology(
    db_path=DB,
    gpkg_path=GPKG_DIR / "SIGEOM_QC_Quaternary_geology_GPKG" / "sigeom.gpkg",
    layer="F10E15_ZONE_MORPH_SEDIM",
    class_col="CODE_DEPOT_MORP_SEDM",
    prefix="quat",
    top_k=8,
)
print(f"\nQuaternary columns added: {quat_cols}")

# -----------------------------------------------------------------------
# 3. Summary
# -----------------------------------------------------------------------
import duckdb

con = duckdb.connect(str(DB), read_only=True)
cols = con.execute("PRAGMA table_info('territorial')").df()["name"].tolist()
geo_cols = [c for c in cols if c.startswith("bedrock_") or c.startswith("quat_")]
print(f"\nAll geology columns in territorial ({len(geo_cols)}):")
for c in geo_cols:
    stats = con.execute(
        f'SELECT SUM("{c}") AS n_nodes, AVG("{c}") AS frac FROM territorial'
    ).df()
    print(f"  {c:30s}  n={int(stats['n_nodes'].iloc[0]):>5d}  frac={stats['frac'].iloc[0]:.3f}")
con.close()
