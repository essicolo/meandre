"""Ingère le drapeau de qualité du CEHQ à côté de chaque débit observé.

Motif (registre R19, 2026-08-21). Nos tables `observations` ne gardaient que la VALEUR.
Le CEHQ publie à côté une remarque, et deux de ses codes disent que la valeur n'est pas
une lecture de courbe de tarage : `E` (estimée) et `R` (corrigée pour l'effet de
refoulement, c'est-à-dire sous glace ou embâcle). Mesuré sur les 16 stations d'OUTV,
tenue de côté 2022-2024 : janvier 85.4 %, février 87.3 %, mars 60.8 %, décembre 47.0 %,
avril 9.5 %, et zéro de mai à octobre. Un cinquième de toute la tenue de côté.

Conséquence directe sur la lecture des scores : l'écart d'hiver du modèle se compare
majoritairement à une reconstruction d'hydrologue, alors que le déficit d'avril (notre
pire mois, 0.753) porte sur des jours MESURÉS et reste donc entier.

Ajoute une colonne `remark` et un booléen `reconstructed` à `observations`, sans toucher
aux valeurs. Idempotent : relancer met à jour au lieu de dupliquer.

    python -m meandre.adapters.physitel.builders.ingest_cehq_flags [region ...]
"""
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import duckdb
import pandas as pd

from meandre.utils import paths as _paths
from meandre.adapters.physitel.cehq_loader import fetch_station, parse_cehq_text, is_reconstructed

REGIONS = sys.argv[1:] or ["outv", "gasp", "sagu", "slno", "mont", "abit", "slso",
                           "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "outm", "vaud", "otsh"]


def ingest(region: str) -> None:
    file_path = _paths.data_path("quebec", f"{region}.duckdb")
    if not os.path.exists(file_path):
        print(f"[{region}] base absente, ignoree")
        return
    con = duckdb.connect(file_path)
    stations = con.execute("SELECT DISTINCT station_id FROM observations ORDER BY 1") \
                  .df().station_id.tolist()

    pieces, failures = [], []
    for s in stations:
        try:
            d = parse_cehq_text(fetch_station(str(s)))
            d = d[d.discharge.notna()][["station_id", "date", "remark"]]
            if len(d):
                pieces.append(d)
        except Exception as e:
            failures.append(f"{s}({type(e).__name__})")
        time.sleep(0.4)   # courtoisie envers le serveur du CEHQ

    if not pieces:
        print(f"[{region}] aucune station recuperee ({len(failures)} echecs)")
        con.close()
        return
    flags = pd.concat(pieces, ignore_index=True)
    flags["station_id"] = flags.station_id.astype(str)
    flags["date"] = pd.to_datetime(flags.date).dt.date
    flags["reconstructed"] = flags.remark.map(is_reconstructed)

    cols = [r[0] for r in con.execute("DESCRIBE observations").fetchall()]
    if "remark" not in cols:
        con.execute("ALTER TABLE observations ADD COLUMN remark VARCHAR")
    if "reconstructed" not in cols:
        con.execute("ALTER TABLE observations ADD COLUMN reconstructed BOOLEAN")
    con.register("f", flags)
    con.execute("""
        UPDATE observations o SET remark = f.remark, reconstructed = f.reconstructed
        FROM f WHERE o.station_id = f.station_id AND o.date = f.date
    """)
    n_tot, n_rec = con.execute(
        "SELECT count(*), count(*) FILTER (WHERE reconstructed) FROM observations "
        "WHERE discharge IS NOT NULL AND reconstructed IS NOT NULL").fetchone()
    n_without = con.execute(
        "SELECT count(*) FROM observations WHERE discharge IS NOT NULL "
        "AND reconstructed IS NULL").fetchone()[0]
    con.close()
    print(f"[{region}] {len(pieces)}/{len(stations)} stations | {n_tot} valeurs "
          f"appariees dont {100*n_rec/max(n_tot,1):.1f} % reconstruites | "
          f"{n_without} sans drapeau"
          + (f" | echecs : {', '.join(failures)}" if failures else ""))


if __name__ == "__main__":
    for r in REGIONS:
        ingest(r)
