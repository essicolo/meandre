"""Ingere CanSWE dans la base d'une region (ou de toutes).

  python .runs/quebec/ingest_canswe.py outv
  python .runs/quebec/ingest_canswe.py toutes
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import duckdb, numpy as np
from meandre import chemins as _ch
from meandre.data.basin_cache import BasinCache
from meandre.data.canswe_loader import charger_canswe

REGIONS = (["abit", "cnda", "cndb", "cndc", "cndd", "cnde", "gasp", "labi", "mont",
            "outm", "outv", "sagu", "slno", "slso", "vaud"]
           if (len(sys.argv) > 1 and sys.argv[1].lower() == "toutes")
           else [(sys.argv[1] if len(sys.argv) > 1 else "outv").lower()])

for reg in REGIONS:
    db = f"{_ch.DATA}/quebec/{reg}.duckdb"
    if not os.path.exists(db):
        print(f"[canswe] {reg}: base absente, ignore")
        continue
    con = duckdb.connect(db, read_only=True)
    noeuds = con.execute("SELECT node_idx, lat, lon FROM nodes ORDER BY node_idx").df()
    con.close()
    # ALTITUDE PHYSIQUE, en metres. La colonne mean_elevation_m de la table territorial
    # est NORMALISEE (elle vaut -2 a +3 sur OUTV) : s'en servir telle quelle donnait un
    # ecart site-noeud de +186 a +412 m dans toutes les regions, un biais entierement
    # fabrique par le code (erreur du 2026-08-20). Les valeurs brutes vivent dans le
    # parquet territorial provincial.
    elev_n = None
    try:
        import pandas as pd
        rw = pd.read_parquet(f"{_ch.DATA}/quebec/territorial-raw-QC.parquet")
        rw = rw[rw.region == reg]
        if len(rw) == len(noeuds):
            elev_n = rw["mean_elevation_m"].values
        else:
            print(f"[canswe] {reg}: altitudes brutes ignorees ({len(rw)} vs {len(noeuds)} noeuds)")
    except Exception as e:
        print(f"[canswe] {reg}: altitudes brutes indisponibles ({type(e).__name__})")

    sites, mesures = charger_canswe(noeuds["lat"].values, noeuds["lon"].values, elev_n)
    if sites.empty:
        print(f"[canswe] {reg}: aucun site dans l'emprise")
        continue
    n = BasinCache(db).import_canswe(sites, mesures)
    ok = int(mesures["quality_ok"].sum()) if len(mesures) else 0
    print(f"[canswe] {reg}: {len(sites)} sites | {n:,} mesures dont {ok:,} de qualite "
          f"| dist mediane {float(sites.dist_km.median()):.1f} km "
          f"| ecart d'altitude median {float(np.nanmedian(sites.elev_diff_m)):+.0f} m")
