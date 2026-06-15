import duckdb
con = duckdb.connect(".runs/slso/data/slso.duckdb", read_only=True)
tables = [t[0] for t in con.execute(
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
).fetchall()]
print("tables:", tables)
for t in tables:
    if "ndvi" in t.lower() or "modis" in t.lower():
        n = con.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {t}").fetchone()
        print(f"{t}: {n}")
