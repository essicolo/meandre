# Basin Database Schema (DuckDB)

Each basin is stored in a single `.duckdb` file created by `BasinCache` from a PHYSITEL project.

## Tables

### metadata

Key-value configuration store.

| Column | Type | Description |
|--------|------|-------------|
| `key` | TEXT (PK) | Configuration key |
| `value` | TEXT | Configuration value |

Keys: `n_nodes`, `source` (PHYSITEL path), `created_at` (ISO timestamp).

### nodes

River network nodes (one per troncon or lake).

| Column | Type | Description |
|--------|------|-------------|
| `node_idx` | INTEGER (PK) | Sequential index (0 to N−1) |
| `node_id` | INTEGER | Original PHYSITEL troncon ID |
| `lon` | FLOAT | Centroid longitude |
| `lat` | FLOAT | Centroid latitude |
| `is_lake` | BOOLEAN | True if node represents a lake |
| `topo_order` | INTEGER | Topological rank (0 = headwater) |

### edges

Directed river connectivity (upstream → downstream).

| Column | Type | Description |
|--------|------|-------------|
| `src` | INTEGER | Upstream node index |
| `dst` | INTEGER | Downstream node index |
| `edge_attr_0` | FLOAT | Edge attribute (reserved) |
| `edge_attr_1` | FLOAT | Edge attribute (reserved) |
| `edge_attr_2` | FLOAT | Edge attribute (reserved) |
| `travel_time_days` | INTEGER | Flow travel time between nodes (days) |

Edge rule: `A.from_junct == B.to_junct` (junction-based connectivity).

### territorial

Catchment characteristics per node (19 fields).

| Column | Type | Description |
|--------|------|-------------|
| `node_idx` | INTEGER (PK) | Node index |
| `drainage_area_km2` | FLOAT | Contributing drainage area |
| `strahler_order` | FLOAT | Stream order |
| `dist_to_outlet_km` | FLOAT | Distance to basin outlet |
| `mean_slope_pct` | FLOAT | Mean slope (%) |
| `mean_elevation_m` | FLOAT | Mean elevation (m) |
| `sin_aspect` | FLOAT | Aspect sine component |
| `cos_aspect` | FLOAT | Aspect cosine component |
| `f_forest` | FLOAT | Forest fraction (0–1) |
| `f_agriculture` | FLOAT | Agriculture fraction |
| `f_urban` | FLOAT | Urban fraction |
| `f_wetland` | FLOAT | Wetland fraction |
| `f_water` | FLOAT | Open water fraction |
| `f_sand` | FLOAT | Sand fraction |
| `f_silt` | FLOAT | Silt fraction |
| `f_clay` | FLOAT | Clay fraction |
| `depth_to_bedrock_m` | FLOAT | Depth to bedrock (m) |
| `lake_fraction` | FLOAT | Lake coverage fraction |
| `area_km2_physical` | FLOAT | Physical catchment area (km²) |
| `area_km2_local` | FLOAT | Local contributing area (km²) |

### cold_state

Default hydrological initial conditions (cold start).

| Column | Type | Description |
|--------|------|-------------|
| `node_idx` | INTEGER (PK) | Node index |
| `theta1` | FLOAT | Soil moisture layer 1 (m³/m³) |
| `theta2` | FLOAT | Soil moisture layer 2 |
| `theta3` | FLOAT | Soil moisture layer 3 |
| `swe` | FLOAT | Snow water equivalent (mm) |
| `t_soil` | FLOAT | Soil temperature (°C) |
| `canopy_storage` | FLOAT | Canopy water storage (mm) |
| `wetland_storage` | FLOAT | Wetland water storage (mm) |

Note: `S_gw` (groundwater storage) and `T_water` (stream temperature) are added at load time with default values (0.0 and 10.0°C) if not present in the table.

### gauging_stations

Hydrometric gauging station metadata.

| Column | Type | Description |
|--------|------|-------------|
| `station_id` | TEXT (PK) | Station identifier (e.g., "023402") |
| `node_idx` | INTEGER | Nearest node index (NULL if unlocated) |
| `lon` | DOUBLE | Station longitude |
| `lat` | DOUBLE | Station latitude |
| `drainage_area_km2` | DOUBLE | Official drainage area (km²) |

### streamflow_obs

Daily discharge measurements at gauging stations.

| Column | Type | Description |
|--------|------|-------------|
| `station_id` | TEXT | Station identifier |
| `date` | DATE | Measurement date |
| `discharge` | FLOAT | Discharge (m³/s), NaN if missing |

Primary key: `(station_id, date)`.

### warm_states

Saved hydrological state snapshots for warm-starting simulations.

| Column | Type | Description |
|--------|------|-------------|
| `state_date` | TEXT | ISO date of snapshot |
| `node_idx` | INTEGER | Node index |
| `theta1`…`wetland_storage` | FLOAT | Same fields as `cold_state` |
| `lake_storage` | FLOAT | Lake water storage (mm) |
| `q_out_prev` | FLOAT | Previous timestep outflow (m³/s) |

Primary key: `(state_date, node_idx)`.

### encoder_states

Saved hidden states for the temporal encoder.

| Column | Type | Description |
|--------|------|-------------|
| `state_date` | TEXT | Snapshot date |
| `run_id` | TEXT | Run identifier |
| `data` | BLOB | Pickled numpy array `(1, N, d_hidden)` |

Primary key: `(state_date, run_id)`.

### withdrawals

Net surface and groundwater pumping / return flow per node and day.
Imported via `BasinCache.import_withdrawals(...)` which snaps each site
(by lon/lat) to the nearest model node, then aggregates by (date, node).

| Column | Type | Description |
|--------|------|-------------|
| `date` | DATE | Day (monthly source data is disaggregated to daily) |
| `node_idx` | INTEGER | Snapped reach index |
| `net_surface` | FLOAT | Net **surface** withdrawal (m³/s) — applied to stream Q |
| `net_gw` | FLOAT | Net **groundwater** withdrawal (m³/s) — applied to S_gw aquifer reservoir |

Primary key: `(date, node_idx)`.

**Sign convention** (matches `routing/withdrawals.py:WithdrawalData`):

* **Positive** = water *added* (effluent, return flow, artificial recharge)
* **Negative** = water *removed* (pumping, irrigation)

Surface intakes/rejects are routed directly into the stream discharge at
the snapped reach.  Groundwater pumping depletes `S_gw` and reduces
baseflow naturally through the aquifer recession `k_gw` — there is no
instantaneous effect on river Q.

Source: `io-eau-meandre.parquet` (site-level monthly records,
positive = withdrawal *as recorded by ETL* — sign is preserved on import,
so positive in the parquet must mean "water added" for correct physics).
