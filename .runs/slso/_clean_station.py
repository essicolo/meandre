"""Supprime une station (+ ses observations) d'une base, avec backup.
  python _clean_station.py <db_path> <station_id>
"""
import sys, shutil, duckdb
DB, SID = sys.argv[1], sys.argv[2]
shutil.copy(DB, DB + ".bak-preclean")
c = duckdb.connect(DB)
for tab in ("stations", "observations"):
    b = c.execute(f"SELECT COUNT(*) FROM {tab}").fetchone()[0]
    c.execute(f"DELETE FROM {tab} WHERE station_id = ?", [SID])
    a = c.execute(f"SELECT COUNT(*) FROM {tab}").fetchone()[0]
    print(f"{tab}: {b} -> {a} (retire {b - a})")
print("stations restantes:", c.execute("SELECT COUNT(*) FROM stations").fetchone()[0])
c.close()
print(f"[ok] backup -> {DB}.bak-preclean")
