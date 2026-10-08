"""Injecte les jauges provinciales québécoises (CEHQ/MELCC) dans la BD open-data.

Source PoC : les 41 stations CEHQ déjà curées dans la BD PHYSITEL slso.duckdb
(métadonnée station + débit journalier 2000-2026). Elles sont RE-SNAPPÉES sur
les nœuds D8 open-data par aire drainée + proximité (même coût que
populate_basin_observations) puis APPEND aux tables stations/observations de la
BD open-data. Les stations HYDAT existantes sont conservées ; une jauge CEHQ et
une jauge HYDAT sur le même nœud sont gardées toutes les deux (décision
2026-06-12). Pas de rebuild : on snappe sur les nœuds existants.

Scale-up TODO : remplacer la source slso.duckdb par un download_observations_cehq()
reproductible (Données Québec / portail CEHQ ; stub dans meandre/data/open_data.py).

  python .runs/slso-od/inject_cehq_obs.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import duckdb
import numpy as np
import pandas as pd

SRC = ".runs/slso/data/slso.duckdb"      # BD PHYSITEL = source CEHQ (PoC)
DST = ".runs/slso-od/data/basin_hydrosheds.duckdb"  # bassin open-data (destination)
MAX_SNAP_KM = 5.0
MAX_RATIO = 2.0
W_KM = 50.0  # poids du mismatch d'aire (|log ratio|) en km, cf. populate_basin_observations

# 1. Stations CEHQ + obs depuis la BD PHYSITEL
src = duckdb.connect(SRC, read_only=True)
cehq = src.execute(
    "SELECT station_id, lon, lat, drainage_area_km2 FROM stations"
).df()
obs = src.execute("SELECT station_id, date, discharge FROM observations").df()
src.close()
print(f"Source CEHQ : {len(cehq)} stations, {len(obs):,} obs ({SRC})")

# 2. Nœuds open-data (lon/lat + aire cumulée physique pour le snapping)
dst = duckdb.connect(DST)
nodes = dst.execute(
    "SELECT n.node_idx, n.lon, n.lat, t.area_km2_physical AS sim_area "
    "FROM nodes n LEFT JOIN territorial t ON n.node_idx = t.node_idx"
).df()
nlon = nodes["lon"].to_numpy()
nlat = nodes["lat"].to_numpy()
narea = np.nan_to_num(nodes["sim_area"].to_numpy(), nan=1e-3)
nidx = nodes["node_idx"].to_numpy()
cos_lat = float(np.cos(np.radians(cehq["lat"].mean())))

# 3. Snapping aire+proximité (mêmes filtres durs que populate_basin_observations)
acc = []
rejected = []
for _, s in cehq.iterrows():
    dx = (nlon - s["lon"]) * 111.0 * cos_lat
    dy = (nlat - s["lat"]) * 111.0
    dist = np.hypot(dx, dy)
    a = s["drainage_area_km2"]
    if a is None or pd.isna(a) or a <= 0:
        i = int(dist.argmin())
        if dist[i] > MAX_SNAP_KM:
            rejected.append((s["station_id"], f"{dist[i]:.1f} km, pas d'aire"))
            continue
        acc.append((s["station_id"], int(nidx[i]), float(s["lon"]), float(s["lat"]),
                    a, float(dist[i]), np.nan))
        continue
    ratio = narea / max(float(a), 1e-3)
    lr = np.abs(np.log(np.clip(ratio, 1e-6, 1e6)))
    cost = dist + W_KM * lr
    valid = (ratio < MAX_RATIO) & (ratio > 1.0 / MAX_RATIO) & (dist <= MAX_SNAP_KM)
    if not valid.any():
        i_c = int(dist.argmin())
        rejected.append((s["station_id"],
                         f"cehq={float(a):.0f} km², sim proche={narea[i_c]:.0f} ({dist[i_c]:.1f} km)"))
        continue
    i = int(np.where(valid, cost, np.inf).argmin())
    acc.append((s["station_id"], int(nidx[i]), float(s["lon"]), float(s["lat"]),
                float(a), float(dist[i]), float(ratio[i])))

acc_df = pd.DataFrame(acc, columns=["station_id", "node_idx", "lon", "lat",
                                    "drainage_area_km2", "snap_dist_km", "snap_ratio"])
print(f"CEHQ snappées : {len(acc_df)}/{len(cehq)}")
if rejected:
    print(f"Rejetées ({len(rejected)}) :")
    for sid, why in rejected:
        print(f"  {sid}: {why}")

# 4. Injection idempotente : purge des CEHQ déjà injectées, puis append
ids = acc_df["station_id"].tolist()
s_db = acc_df[["station_id", "node_idx", "lon", "lat", "drainage_area_km2"]].copy()
o_db = obs[obs["station_id"].isin(ids)][["station_id", "date", "discharge"]].copy()

dst.register("cehq_ids", pd.DataFrame({"station_id": ids}))
dst.execute("DELETE FROM stations WHERE station_id IN (SELECT station_id FROM cehq_ids)")
dst.execute("DELETE FROM observations WHERE station_id IN (SELECT station_id FROM cehq_ids)")
dst.register("s_df", s_db)
dst.register("o_df", o_db)
dst.execute("INSERT INTO stations SELECT * FROM s_df")
dst.execute("INSERT INTO observations SELECT * FROM o_df")

# 5. Bilan
n_sta = dst.execute("SELECT count(*) FROM stations").fetchone()[0]
n_obs_sta = dst.execute("SELECT count(DISTINCT station_id) FROM observations").fetchone()[0]
print(f"Injecté : {len(s_db)} stations CEHQ, {len(o_db):,} obs")
print(f"BD open-data : {n_sta} stations au total, {n_obs_sta} avec obs")
dst.close()
print("INJECT_CEHQ_DONE")
