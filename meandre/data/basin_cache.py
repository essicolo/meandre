"""DuckDB-backed cache for hydrological basin data.

Source-agnostic: any loader (PHYSITEL, Planetary Computer, synthetic, …)
that produces the standard ``hydro`` dict can write it here.

Standard hydro dict keys
------------------------
graph           RiverGraph
territorial     TerritorialFeatures
node_coords     Tensor (n_nodes, 2)
initial_state   HydroState  (table: initial_state)
node_ids        list[int]
n_nodes         int

Schema
------
metadata        key/value store (n_nodes, source, created_at, …)
nodes           node_idx, node_id, lon, lat, is_lake, topo_order
edges           src, dst, edge_attr_0..2, travel_time_days
territorial     node_idx + 17 normalised fields + area_km2_physical + area_km2_local
initial_state      node_idx + 7 HydroState fields
stations  station_id, node_idx, lon, lat, drainage_area_km2
observations  station_id, date, discharge
warm_states     state_date, node_idx + HydroState + lake_storage + q_out_prev
encoder_states  state_date, run_id, data BLOB (pickled numpy h_context)

Typical usage
-------------
# One-time: build from a PHYSITEL project
cache = BasinCache.from_hydrotel("path/to/SLSO", "data/slso.duckdb")

# Load in milliseconds (drop-in for load_hydrotel)
hydro = BasinCache("data/slso.duckdb").load(device=device)

# Import station observations (idempotent — upserts)
cache.import_observations("stations.nc", basin_prefix="SLSO")

# Load observations for training
obs = cache.load_observations("2000-01-01", "2001-12-31", min_valid_days=365)

# Persist a warm-start state after spinup
cache.save_state("2002-12-31", state, lake_storage, Q_out_prev, h_context)

# Resume next run
ws = cache.load_state("2002-12-31", device=device)
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from meandre.routing.graph import RiverGraph
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures, DEFAULT_PHYSICAL_COLUMNS
from meandre.utils.state import HydroState

_HYDROSTATE_FIELDS = [
    "theta1", "theta2", "theta3",
    "swe", "t_soil", "canopy_storage", "wetland_storage",
]


class BasinCache:
    """Read/write basin data to/from a DuckDB file.

    Parameters
    ----------
    path : str | Path
        Path to the DuckDB file.  Created on first write if absent.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_hydrotel(
        cls,
        project_dir: str | Path,
        path: str | Path,
        normalise: bool = True,
        device: torch.device | None = None,
    ) -> "BasinCache":
        """Parse a PHYSITEL/HYDROTEL project and write to DuckDB.

        Parameters
        ----------
        project_dir :
            Root of the HYDROTEL project (contains physitel/ and etat/).
        path :
            Destination DuckDB file (overwritten if it exists).
        normalise :
            Same flag as ``load_hydrotel``.
        device :
            Ignored at write time; tensors stored as CPU numpy arrays.
        """
        from meandre.data.physitel_loader import load_hydrotel

        hydro = load_hydrotel(project_dir, normalise=normalise, device=None)
        cache = cls(path)
        cache.write(hydro, source=str(project_dir))
        return cache

    @classmethod
    def from_dict(
        cls,
        hydro: dict,
        path: str | Path,
        source: str = "synthetic",
    ) -> "BasinCache":
        """Write any standard hydro dict directly to DuckDB."""
        cache = cls(path)
        cache.write(hydro, source=source)
        return cache

    # ------------------------------------------------------------------
    # Load static data
    # ------------------------------------------------------------------

    def load(self, device: torch.device | None = None) -> dict:
        """Load static basin data — same dict as ``load_hydrotel``."""
        import duckdb

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            graph = self._load_graph(con, device)
            territorial = self._load_territorial(con, device)
            node_coords, node_ids = self._load_nodes(con, device)
            initial_state = self._load_initial_state(con, device)
        finally:
            con.close()

        return {
            "graph": graph,
            "territorial": territorial,
            "node_coords": node_coords,
            "initial_state": initial_state,
            "node_ids": node_ids,
            "n_nodes": len(node_ids),
        }

    # ------------------------------------------------------------------
    # Warm-start state persistence
    # ------------------------------------------------------------------

    def save_state(
        self,
        date: str,
        state: HydroState,
        lake_storage: Tensor | None = None,
        q_out_prev: Tensor | None = None,
        h_context: Tensor | None = None,
        run_id: str = "default",
    ) -> None:
        """Persist a simulation state snapshot for later warm-starting.

        Parameters
        ----------
        date :
            ISO 8601 string, e.g. ``"2002-12-31"``.  Used as primary key;
            an existing entry for the same (date, run_id) is replaced.
        state :
            HydroState at the end of the simulation period.
        lake_storage :
            (n_nodes,) tensor, or None (stored as zeros).
        q_out_prev :
            (n_nodes,) tensor, or None (stored as zeros).
        h_context :
            GRU hidden state (1, n_nodes, d_hidden), or None.
        run_id :
            Tag to distinguish multiple runs at the same date.
        """
        import duckdb
        import pandas as pd

        n = state.theta1.shape[0]
        df = pd.DataFrame({"state_date": date, "node_idx": np.arange(n)})
        for f in _HYDROSTATE_FIELDS:
            df[f] = getattr(state, f).detach().cpu().numpy()
        df["lake_storage"] = (
            lake_storage.detach().cpu().numpy() if lake_storage is not None
            else np.zeros(n, dtype=np.float32)
        )
        df["q_out_prev"] = (
            q_out_prev.detach().cpu().numpy() if q_out_prev is not None
            else np.zeros(n, dtype=np.float32)
        )
        # Enforce column order to match schema
        df = df[["state_date", "node_idx"] + _HYDROSTATE_FIELDS
                + ["lake_storage", "q_out_prev"]]

        con = duckdb.connect(str(self.path))
        try:
            con.execute(
                "DELETE FROM warm_states WHERE state_date = ?", [date]
            )
            con.execute("INSERT INTO warm_states SELECT * FROM df")

            con.execute(
                "DELETE FROM encoder_states WHERE state_date = ? AND run_id = ?",
                [date, run_id],
            )
            if h_context is not None:
                blob = pickle.dumps(h_context.detach().cpu().numpy())
                con.execute(
                    "INSERT INTO encoder_states VALUES (?, ?, ?)",
                    [date, run_id, blob],
                )
        finally:
            con.close()

    def load_state(
        self,
        date: str,
        run_id: str = "default",
        device: torch.device | None = None,
    ) -> dict:
        """Load a warm-start snapshot.

        Returns
        -------
        dict with keys: ``state``, ``lake_storage``, ``q_out_prev``, ``h_context``
        """
        import duckdb

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            df = con.execute(
                "SELECT * FROM warm_states WHERE state_date = ? ORDER BY node_idx",
                [date],
            ).df()
            if df.empty:
                raise KeyError(f"No warm state found for date '{date}'")

            def _t(col: str) -> Tensor:
                return torch.tensor(df[col].to_numpy(), dtype=torch.float32,
                                    device=device)

            n = len(df)
            state = HydroState(
                theta1=_t("theta1"), theta2=_t("theta2"), theta3=_t("theta3"),
                swe=_t("swe"), t_soil=_t("t_soil"),
                canopy_storage=_t("canopy_storage"),
                wetland_storage=_t("wetland_storage"),
                S_gw=_t("S_gw") if "S_gw" in df.columns else torch.zeros(n, dtype=torch.float32, device=device),
                T_water=_t("T_water") if "T_water" in df.columns else torch.full((n,), 10.0, dtype=torch.float32, device=device),
            )

            h_context: Tensor | None = None
            row = con.execute(
                "SELECT data FROM encoder_states WHERE state_date = ? AND run_id = ?",
                [date, run_id],
            ).fetchone()
            if row is not None:
                h_context = torch.tensor(
                    pickle.loads(row[0]), dtype=torch.float32, device=device
                )
        finally:
            con.close()

        return {
            "state": state,
            "lake_storage": _t("lake_storage"),
            "q_out_prev": _t("q_out_prev"),
            "h_context": h_context,
        }

    def list_states(self) -> list[str]:
        """Sorted list of available warm-start dates."""
        import duckdb

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            rows = con.execute(
                "SELECT DISTINCT state_date FROM warm_states ORDER BY state_date"
            ).fetchall()
        finally:
            con.close()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Observations import / load
    # ------------------------------------------------------------------

    def import_observations(
        self,
        source: str | Path,
        basin_prefix: str | None = None,
    ) -> int:
        """Import station observations into the DuckDB.

        Idempotent: new data is upserted (INSERT OR REPLACE).  Existing
        observations for the same (station_id, date) are overwritten.

        Parameters
        ----------
        source :
            Path to one of:
            - NetCDF with ``discharge``, ``troncon_id``, ``station_id``,
              ``lon``, ``lat``, ``drainage_area`` (the current format).
            - CSV with columns ``station_id, date, discharge`` (long format).
            - Parquet with the same columns as CSV.
        basin_prefix :
            If given, filter stations by ``troncon_id`` prefix (e.g. "SLSO").
            Only relevant for NetCDF sources that contain multiple basins.
            If None, import all stations found in the source.

        Returns
        -------
        int : number of observation rows inserted.
        """
        import duckdb

        source = Path(source)
        suffix = source.suffix.lower()

        con = duckdb.connect(str(self.path))
        try:
            self._ensure_obs_tables(con)

            if suffix in (".nc", ".nc4", ".netcdf"):
                n = self._import_from_nc(con, source, basin_prefix)
            elif suffix == ".csv":
                n = self._import_from_tabular(con, source, "csv")
            elif suffix == ".parquet":
                n = self._import_from_tabular(con, source, "parquet")
            else:
                raise ValueError(f"Unsupported format: {suffix}")
        finally:
            con.close()

        print(f"[import_observations] {n:,} rows imported from {source.name}")
        return n

    def import_observations_df(
        self,
        stations_df,
        obs_df,
    ) -> int:
        """Import from DataFrames directly.

        Parameters
        ----------
        stations_df :
            DataFrame with columns: station_id, node_idx, lon, lat,
            drainage_area_km2.
        obs_df :
            DataFrame with columns: station_id, date, discharge.

        Returns
        -------
        int : number of observation rows inserted.
        """
        import duckdb

        con = duckdb.connect(str(self.path))
        try:
            self._ensure_obs_tables(con)
            con.execute(
                "INSERT OR REPLACE INTO stations SELECT * FROM stations_df"
            )
            con.execute(
                "INSERT OR REPLACE INTO observations SELECT * FROM obs_df"
            )
            n = len(obs_df)
        finally:
            con.close()
        return n

    def load_observations(
        self,
        date_start: str,
        date_end: str,
        min_valid_days: int = 365,
    ) -> dict:
        """Load observations from DuckDB for a date window.

        Parameters
        ----------
        date_start, date_end :
            ISO date strings (inclusive).
        min_valid_days :
            Minimum non-NULL discharge days for a station to be retained.

        Returns
        -------
        dict with keys:
            ``discharge``        : (T, N) float32 array — NaN at ungauged nodes
            ``station_node_map`` : {station_id: node_idx}
            ``dates``            : (T,) datetime64 array
            ``n_stations``       : int
        """
        import duckdb
        import pandas as pd

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            # Get node count
            n_nodes = int(con.execute(
                "SELECT value FROM metadata WHERE key = 'n_nodes'"
            ).fetchone()[0])

            # Build date grid
            dates_df = con.execute(
                "SELECT CAST(date AS DATE) as date "
                "FROM generate_series(CAST(? AS DATE), CAST(? AS DATE), "
                "INTERVAL 1 DAY) AS t(date)",
                [date_start, date_end],
            ).df()
            dates = dates_df["date"].values.astype("datetime64[ns]")
            n_time = len(dates)

            # Get stations with enough valid data
            stations_df = con.execute("""
                SELECT s.station_id, s.node_idx, s.drainage_area_km2
                FROM stations s
                WHERE s.node_idx IS NOT NULL
                  AND (SELECT COUNT(*) FROM observations o
                       WHERE o.station_id = s.station_id
                         AND o.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                         AND o.discharge IS NOT NULL) >= ?
                ORDER BY s.station_id
            """, [date_start, date_end, min_valid_days]).df()

            if stations_df.empty:
                return {
                    "discharge": np.full((n_time, n_nodes), np.nan, dtype=np.float32),
                    "station_node_map": {},
                    "dates": dates,
                    "n_stations": 0,
                }

            # Pivot observations into (time, station) matrix
            sids = stations_df["station_id"].tolist()
            sid_list = ", ".join(f"'{s}'" for s in sids)

            obs_df = con.execute(f"""
                SELECT o.date, o.station_id, o.discharge
                FROM observations o
                WHERE o.station_id IN ({sid_list})
                  AND o.date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
                ORDER BY o.date, o.station_id
            """, [date_start, date_end]).df()
        finally:
            con.close()

        # Build the (T, N_all) discharge array
        discharge_full = np.full((n_time, n_nodes), np.nan, dtype=np.float32)
        station_node_map: dict[str, int] = {}

        # Build station→node map (one station per node, first wins)
        seen_nodes: set[int] = set()
        for sid, ni in zip(stations_df["station_id"], stations_df["node_idx"].astype(int)):
            if ni not in seen_nodes:
                station_node_map[sid] = ni
                seen_nodes.add(ni)

        # Vectorized fill: merge obs with date index and node index
        if not obs_df.empty and station_node_map:
            obs_df = obs_df[obs_df["station_id"].isin(station_node_map)]
            obs_df = obs_df.dropna(subset=["discharge"])

            # Map station_id → node_idx
            sid_to_node = pd.Series(station_node_map)
            ni = sid_to_node.reindex(obs_df["station_id"]).values.astype(int)

            # Map date → time index
            date_idx = pd.Series(np.arange(n_time), index=dates)
            obs_dates = pd.to_datetime(obs_df["date"]).values.astype("datetime64[ns]")
            ti = date_idx.reindex(obs_dates).values

            vals = obs_df["discharge"].values.astype(np.float32)

            # Filter valid (date within range)
            valid = ~np.isnan(ti)
            ti = ti[valid].astype(int)
            ni = ni[valid]
            vals = vals[valid]

            discharge_full[ti, ni] = vals

        n_kept = len(station_node_map)
        print(f"[load_observations] {n_kept} stations, "
              f"{date_start} to {date_end}")

        return {
            "discharge": discharge_full,
            "station_node_map": station_node_map,
            "dates": dates,
            "n_stations": n_kept,
        }

    # ------------------------------------------------------------------
    # Withdrawals import / load
    # ------------------------------------------------------------------

    def import_withdrawals(
        self,
        source,
        date_col: str = "date",
        net_col: str = "net_withdrawal",
        lon_col: str = "lon",
        lat_col: str = "lat",
        site_col: str | None = None,
        node_col: str | None = None,
        source_col: str | None = "source",
        gw_values: tuple[str, ...] = ("Souterrain", "SOUTERRAIN", "GW", "groundwater"),
        max_snap_km: float = 10.0,
    ) -> int:
        """Import net withdrawal data into the DuckDB.

        Two modes:

        * **With coordinates** (``lon``, ``lat`` columns present): each site
          is snapped to the nearest model node automatically.  If multiple
          sites snap to the same node on the same date, values are summed.
        * **With node_idx** (``node_idx`` column present): inserted directly,
          no snapping.

        Parameters
        ----------
        source :
            Path to CSV/Parquet, or a pandas DataFrame.
            Required columns: ``date``, ``net_withdrawal`` (m³/s).
            Positive = water added (effluent, return flow).
            Negative = water removed (pumping, irrigation).
            Plus either ``lon``/``lat`` or ``node_idx``.
        site_col :
            Optional column identifying each site (for reporting snap
            distances).  If absent, each row is treated independently.
        node_col :
            Column with pre-assigned node indices.  If present, snapping
            is skipped.
        source_col :
            Optional column identifying the physical source of the
            withdrawal.  Rows whose value matches ``gw_values`` are
            stored in ``net_gw`` (applied to the aquifer reservoir);
            all other rows go to ``net_surface`` (applied to stream Q).
            If the column is absent, every row is treated as surface.
        gw_values :
            Tuple of strings flagging a groundwater source.
        max_snap_km :
            Warn for sites farther than this from any node.

        Returns
        -------
        int : number of rows inserted.
        """
        import duckdb
        import pandas as pd

        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            source = Path(source)
            if source.suffix == ".csv":
                df = pd.read_csv(source, parse_dates=[date_col])
            elif source.suffix == ".parquet":
                # Use DuckDB to read parquet (no pyarrow dependency)
                _con = duckdb.connect(":memory:")
                df = _con.execute(f"SELECT * FROM '{source}'").fetchdf()
                _con.close()
            else:
                raise ValueError(f"Unsupported format: {source.suffix}")

        # Rename to canonical columns
        rename = {date_col: "date", net_col: "net_withdrawal"}
        if lon_col in df.columns:
            rename[lon_col] = "lon"
        if lat_col in df.columns:
            rename[lat_col] = "lat"
        if site_col and site_col in df.columns:
            rename[site_col] = "site_id"
        if node_col and node_col in df.columns:
            rename[node_col] = "node_idx"
        if source_col and source_col in df.columns and source_col != "source":
            rename[source_col] = "source"
        df = df.rename(columns=rename)

        # ── Classify rows as surface or groundwater ──────────────────────
        if "source" in df.columns:
            gw_set = {v for v in gw_values}
            is_gw = df["source"].astype(str).isin(gw_set)
        else:
            is_gw = pd.Series(False, index=df.index)

        df["net_surface"] = df["net_withdrawal"].where(~is_gw, 0.0)
        df["net_gw"] = df["net_withdrawal"].where(is_gw, 0.0)

        n_gw_rows = int(is_gw.sum())
        n_surf_rows = int((~is_gw).sum())

        # ── Snap to nearest node if lon/lat present ──────────────────────
        if "node_idx" not in df.columns:
            if "lon" not in df.columns or "lat" not in df.columns:
                raise ValueError(
                    "Need either 'node_idx' or 'lon'+'lat' columns. "
                    f"Got: {list(df.columns)}"
                )
            df = self._snap_withdrawals(df, max_snap_km)

        # ── Aggregate by (date, node_idx) — sum if multiple sites ────────
        df = (
            df.groupby(["date", "node_idx"])[["net_surface", "net_gw"]]
            .sum()
            .reset_index()
        )

        con = duckdb.connect(str(self.path))
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS withdrawals (
                    date         DATE,
                    node_idx     INTEGER,
                    net_surface  FLOAT,
                    net_gw       FLOAT,
                    PRIMARY KEY (date, node_idx)
                )
            """)
            # Migrate legacy schema (net_withdrawal → net_surface + net_gw)
            cols = [r[0] for r in con.execute(
                "PRAGMA table_info('withdrawals')"
            ).fetchall()]
            # DuckDB PRAGMA returns (cid, name, type, ...); grab name column
            col_names = [r[1] for r in con.execute(
                "PRAGMA table_info('withdrawals')"
            ).fetchall()]
            if "net_withdrawal" in col_names and "net_surface" not in col_names:
                con.execute(
                    "ALTER TABLE withdrawals RENAME COLUMN net_withdrawal TO net_surface"
                )
                con.execute(
                    "ALTER TABLE withdrawals ADD COLUMN net_gw FLOAT DEFAULT 0.0"
                )
                print("[import_withdrawals] migrated legacy schema "
                      "(net_withdrawal -> net_surface + net_gw)")

            con.execute(
                "INSERT OR REPLACE INTO withdrawals "
                "SELECT date, node_idx, net_surface, net_gw FROM df"
            )
        finally:
            con.close()

        n_nodes = df["node_idx"].nunique()
        n_dates = df["date"].nunique()
        print(f"[import_withdrawals] {len(df):,} rows imported "
              f"({n_nodes} nodes, {n_dates} dates; "
              f"{n_surf_rows} surface, {n_gw_rows} groundwater)")
        return len(df)

    def _snap_withdrawals(self, df, max_snap_km: float):
        """Snap lon/lat to nearest model node, return df with node_idx."""
        import duckdb

        con = duckdb.connect(str(self.path), read_only=True)
        nodes = con.execute(
            "SELECT node_idx, lon, lat FROM nodes ORDER BY node_idx"
        ).df()
        con.close()

        node_lons = nodes["lon"].values
        node_lats = nodes["lat"].values

        # Build site→node_idx map
        if "site_id" in df.columns:
            sites = df.drop_duplicates("site_id")[["site_id", "lon", "lat"]]
        else:
            sites = df[["lon", "lat"]].drop_duplicates()
            sites["site_id"] = range(len(sites))
            df["site_id"] = df.apply(
                lambda r: sites.loc[
                    (sites["lon"] == r["lon"]) & (sites["lat"] == r["lat"]),
                    "site_id",
                ].iloc[0],
                axis=1,
            )

        snap_map = {}
        n_dropped = 0
        for _, row in sites.iterrows():
            R = 6371.0
            dlon = np.radians(node_lons - row["lon"])
            dlat = np.radians(node_lats - row["lat"])
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(np.radians(row["lat"]))
                 * np.cos(np.radians(node_lats))
                 * np.sin(dlon / 2) ** 2)
            dists = R * 2 * np.arcsin(np.sqrt(a))
            best_idx = int(np.argmin(dists))
            best_dist = dists[best_idx]
            if best_dist > max_snap_km:
                n_dropped += 1
                continue  # skip sites outside the basin
            snap_map[row["site_id"]] = best_idx
            if best_dist > 1.0:
                print(f"[import_withdrawals] site '{row['site_id']}' -> "
                      f"node {best_idx} ({best_dist:.1f} km)")

        if n_dropped:
            print(f"[import_withdrawals] {n_dropped} sites dropped "
                  f"(> {max_snap_km} km from any node)")

        df["node_idx"] = df["site_id"].map(snap_map)
        df = df.dropna(subset=["node_idx"])
        df["node_idx"] = df["node_idx"].astype(int)
        return df

    def load_withdrawals(
        self,
        date_start: str,
        date_end: str,
        device: torch.device | None = None,
    ) -> WithdrawalData:
        """Load withdrawals from DuckDB for a date window.

        Returns ``WithdrawalData.zeros()`` if the table does not exist.
        """
        import duckdb

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            n_nodes = int(con.execute(
                "SELECT value FROM metadata WHERE key = 'n_nodes'"
            ).fetchone()[0])

            if "withdrawals" not in tables:
                con.close()
                import pandas as pd
                dates = pd.date_range(date_start, date_end, freq="D")
                return WithdrawalData.zeros(len(dates), n_nodes, device=device)

            # Detect schema version (legacy vs split)
            col_names = [r[1] for r in con.execute(
                "PRAGMA table_info('withdrawals')"
            ).fetchall()]
            has_split = "net_surface" in col_names and "net_gw" in col_names

            # Build date grid
            import pandas as pd
            dates = pd.date_range(date_start, date_end, freq="D")
            n_time = len(dates)

            if has_split:
                df = con.execute(
                    "SELECT date, node_idx, net_surface, net_gw "
                    "FROM withdrawals "
                    "WHERE date >= CAST(? AS DATE) AND date <= CAST(? AS DATE) "
                    "ORDER BY date, node_idx",
                    [date_start, date_end],
                ).df()
            else:
                # Legacy schema: all values treated as surface withdrawals.
                df = con.execute(
                    "SELECT date, node_idx, net_withdrawal AS net_surface, "
                    "CAST(0.0 AS FLOAT) AS net_gw "
                    "FROM withdrawals "
                    "WHERE date >= CAST(? AS DATE) AND date <= CAST(? AS DATE) "
                    "ORDER BY date, node_idx",
                    [date_start, date_end],
                ).df()
        finally:
            con.close()

        net_surface = np.zeros((n_time, n_nodes), dtype=np.float32)
        net_gw = np.zeros((n_time, n_nodes), dtype=np.float32)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["node_idx"] < n_nodes]

            # Detect monthly vs daily data
            unique_days = df["date"].dt.day.unique()
            is_monthly = len(unique_days) == 1 and unique_days[0] == 1

            if is_monthly:
                # Expand monthly rows to daily using vectorized merge
                month_starts = df["date"].unique()
                daily_rows = []
                for ms in month_starts:
                    me = ms + pd.offsets.MonthEnd(0)
                    month_days = pd.date_range(ms, me, freq="D")
                    month_days = month_days[(month_days >= dates[0]) & (month_days <= dates[-1])]
                    if len(month_days) > 0:
                        month_df = df[df["date"] == ms][
                            ["node_idx", "net_surface", "net_gw"]
                        ]
                        for d in month_days:
                            daily_rows.append(month_df.assign(date=d))
                if daily_rows:
                    df = pd.concat(daily_rows, ignore_index=True)
                else:
                    df = df.iloc[:0]

            # Vectorized assignment via date index lookup
            date_idx_series = pd.Series(
                np.arange(n_time), index=dates,
            )
            ti = date_idx_series.reindex(df["date"]).values
            ni = df["node_idx"].values.astype(int)
            vals_s = df["net_surface"].values.astype(np.float32)
            vals_gw = df["net_gw"].values.astype(np.float32)

            # Filter valid indices
            valid = ~np.isnan(ti)
            ti = ti[valid].astype(int)
            ni = ni[valid]
            vals_s = vals_s[valid]
            vals_gw = vals_gw[valid]

            # Accumulate (handles duplicates via np.add.at)
            np.add.at(net_surface, (ti, ni), vals_s)
            np.add.at(net_gw, (ti, ni), vals_gw)

        return WithdrawalData(
            net=torch.tensor(net_surface, device=device),
            net_gw=torch.tensor(net_gw, device=device),
        )

    def import_modis_et(self, df: "pd.DataFrame") -> int:
        """Import MODIS MOD16A2 ETR rows into the ``modis_et`` DuckDB table.

        Parameters
        ----------
        df : DataFrame with columns (date, node_idx, etr_mm_day, quality_ok).
             Produced by :func:`meandre.data.modis_loader.fetch_modis_et`.

        Returns
        -------
        int : number of rows inserted.
        """
        import duckdb
        import pandas as pd

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        con = duckdb.connect(str(self.path))
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS modis_et (
                    date        DATE    NOT NULL,
                    node_idx    INTEGER NOT NULL,
                    etr_mm_day  FLOAT,
                    quality_ok  BOOLEAN,
                    PRIMARY KEY (date, node_idx)
                )
            """)
            con.execute(
                "INSERT OR REPLACE INTO modis_et SELECT * FROM df"
            )
            n = len(df)
        finally:
            con.close()

        print(f"[import_modis_et] {n:,} rows upserted into modis_et")
        return n

    def load_modis_et(
        self,
        date_start: str,
        date_end: str,
        device: "torch.device | None" = None,
    ) -> "torch.Tensor | None":
        """Load MODIS MOD16A2 ETR from DuckDB as a dense (T, n_nodes) tensor.

        Returns ``None`` if the ``modis_et`` table does not exist (enables
        graceful fallback to w_nll_et=0 with no code change in slso.py).

        NaN encodes:
          - Days between 8-day composites (no observation).
          - Pixels flagged as cloudy or fill.
        The Gaussian NLL loss skips NaN entries automatically.
        """
        import duckdb
        import pandas as pd

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if "modis_et" not in tables:
                return None

            n_nodes = int(con.execute(
                "SELECT value FROM metadata WHERE key = 'n_nodes'"
            ).fetchone()[0])

            dates = pd.date_range(date_start, date_end, freq="D")
            n_time = len(dates)
            date_idx = {d: i for i, d in enumerate(dates.normalize())}

            df = con.execute(
                "SELECT date, node_idx, etr_mm_day, quality_ok "
                "FROM modis_et "
                "WHERE date >= CAST(? AS DATE) AND date <= CAST(? AS DATE) "
                "  AND quality_ok = TRUE "
                "ORDER BY date, node_idx",
                [date_start, date_end],
            ).df()
        finally:
            con.close()

        et_arr = np.full((n_time, n_nodes), np.nan, dtype=np.float32)

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            for _, row in df.iterrows():
                t = date_idx.get(row["date"])
                n = int(row["node_idx"])
                if t is not None and n < n_nodes:
                    et_arr[t, n] = float(row["etr_mm_day"])

        import torch
        return torch.from_numpy(et_arr).to(device)

    def has_modis_et(self) -> bool:
        """Return True if the modis_et table exists and has at least one row."""
        import duckdb
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if "modis_et" not in tables:
                return False
            count = con.execute("SELECT COUNT(*) FROM modis_et").fetchone()[0]
            return int(count) > 0
        finally:
            con.close()

    def import_modis_snow(self, df: "pd.DataFrame") -> int:
        """Import MOD10A1 snow cover fraction into ``modis_snow`` table."""
        import duckdb, pandas as pd
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        con = duckdb.connect(str(self.path))
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS modis_snow (
                    date        DATE    NOT NULL,
                    node_idx    INTEGER NOT NULL,
                    snow_frac   FLOAT,
                    quality_ok  BOOLEAN,
                    PRIMARY KEY (date, node_idx)
                )
            """)
            con.execute("INSERT OR REPLACE INTO modis_snow SELECT * FROM df")
            n = len(df)
        finally:
            con.close()
        print(f"[import_modis_snow] {n:,} rows upserted")
        return n

    def load_modis_snow(
        self, date_start: str, date_end: str,
        device: "torch.device | None" = None,
    ) -> "torch.Tensor | None":
        """Load MOD10A1 snow cover as (T, n_nodes) tensor. None if absent."""
        import duckdb, pandas as pd
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if "modis_snow" not in tables:
                return None
            n_nodes = int(con.execute(
                "SELECT value FROM metadata WHERE key = 'n_nodes'"
            ).fetchone()[0])
            dates = pd.date_range(date_start, date_end, freq="D")
            date_idx = {d: i for i, d in enumerate(dates.normalize())}
            df = con.execute(
                "SELECT date, node_idx, snow_frac FROM modis_snow "
                "WHERE date >= CAST(? AS DATE) AND date <= CAST(? AS DATE) "
                "  AND quality_ok = TRUE ORDER BY date, node_idx",
                [date_start, date_end],
            ).df()
        finally:
            con.close()
        arr = np.full((len(dates), n_nodes), np.nan, dtype=np.float32)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            for _, row in df.iterrows():
                t = date_idx.get(row["date"])
                n = int(row["node_idx"])
                if t is not None and n < n_nodes:
                    arr[t, n] = float(row["snow_frac"])
        import torch
        return torch.from_numpy(arr).to(device)

    def has_modis_snow(self) -> bool:
        import duckdb
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            return "modis_snow" in tables and \
                int(con.execute("SELECT COUNT(*) FROM modis_snow").fetchone()[0]) > 0
        finally:
            con.close()

    def import_modis_ndvi(self, df: "pd.DataFrame") -> int:
        """Import MOD13A2 NDVI into ``modis_ndvi`` table."""
        import duckdb, pandas as pd
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        con = duckdb.connect(str(self.path))
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS modis_ndvi (
                    date        DATE    NOT NULL,
                    node_idx    INTEGER NOT NULL,
                    ndvi        FLOAT,
                    quality_ok  BOOLEAN,
                    PRIMARY KEY (date, node_idx)
                )
            """)
            con.execute("INSERT OR REPLACE INTO modis_ndvi SELECT * FROM df")
            n = len(df)
        finally:
            con.close()
        print(f"[import_modis_ndvi] {n:,} rows upserted")
        return n

    def load_modis_ndvi(
        self, date_start: str, date_end: str,
        device: "torch.device | None" = None,
    ) -> "torch.Tensor | None":
        """Load MOD13A2 NDVI as (T, n_nodes) tensor. None if absent."""
        import duckdb, pandas as pd
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if "modis_ndvi" not in tables:
                return None
            n_nodes = int(con.execute(
                "SELECT value FROM metadata WHERE key = 'n_nodes'"
            ).fetchone()[0])
            dates = pd.date_range(date_start, date_end, freq="D")
            date_idx = {d: i for i, d in enumerate(dates.normalize())}
            df = con.execute(
                "SELECT date, node_idx, ndvi FROM modis_ndvi "
                "WHERE date >= CAST(? AS DATE) AND date <= CAST(? AS DATE) "
                "  AND quality_ok = TRUE ORDER BY date, node_idx",
                [date_start, date_end],
            ).df()
        finally:
            con.close()
        arr = np.full((len(dates), n_nodes), np.nan, dtype=np.float32)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            for _, row in df.iterrows():
                t = date_idx.get(row["date"])
                n = int(row["node_idx"])
                if t is not None and n < n_nodes:
                    arr[t, n] = float(row["ndvi"])
        import torch
        return torch.from_numpy(arr).to(device)

    def has_modis_ndvi(self) -> bool:
        import duckdb
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            return "modis_ndvi" in tables and \
                int(con.execute("SELECT COUNT(*) FROM modis_ndvi").fetchone()[0]) > 0
        finally:
            con.close()

    def import_grace_tws(self, df: "pd.DataFrame") -> int:
        """Import GRACE/GRACE-FO TWS anomaly into ``grace_tws`` table.

        df columns: (date, tws_mm, uncertainty, quality_ok).
        """
        import duckdb, pandas as pd
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        con = duckdb.connect(str(self.path))
        try:
            con.execute("""
                CREATE TABLE IF NOT EXISTS grace_tws (
                    date        DATE  NOT NULL PRIMARY KEY,
                    tws_mm      FLOAT,
                    uncertainty FLOAT,
                    quality_ok  BOOLEAN
                )
            """)
            con.execute("INSERT OR REPLACE INTO grace_tws SELECT * FROM df")
            n = len(df)
        finally:
            con.close()
        print(f"[import_grace_tws] {n:,} rows upserted")
        return n

    def load_grace_tws(
        self, date_start: str, date_end: str,
    ) -> "pd.DataFrame | None":
        """Load GRACE TWS as DataFrame(date, tws_mm, uncertainty). None if absent."""
        import duckdb, pandas as pd
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if "grace_tws" not in tables:
                return None
            df = con.execute(
                "SELECT date, tws_mm, uncertainty FROM grace_tws "
                "WHERE date >= CAST(? AS DATE) AND date <= CAST(? AS DATE) "
                "  AND quality_ok = TRUE ORDER BY date",
                [date_start, date_end],
            ).df()
        finally:
            con.close()
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        return df

    def has_grace_tws(self) -> bool:
        import duckdb
        con = duckdb.connect(str(self.path), read_only=True)
        try:
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            return "grace_tws" in tables and \
                int(con.execute("SELECT COUNT(*) FROM grace_tws").fetchone()[0]) > 0
        finally:
            con.close()

    def list_stations(self) -> list[dict]:
        """List stations with their metadata and observation date ranges."""
        import duckdb

        con = duckdb.connect(str(self.path), read_only=True)
        try:
            rows = con.execute("""
                SELECT s.station_id, s.node_idx, s.drainage_area_km2,
                       MIN(o.date) AS first_date,
                       MAX(o.date) AS last_date,
                       COUNT(o.discharge) AS n_obs
                FROM stations s
                LEFT JOIN observations o ON o.station_id = s.station_id
                GROUP BY s.station_id, s.node_idx, s.drainage_area_km2
                ORDER BY s.station_id
            """).fetchall()
        finally:
            con.close()
        return [
            {"station_id": r[0], "node_idx": r[1], "area_km2": r[2],
             "first_date": r[3], "last_date": r[4], "n_obs": r[5]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Observations — private helpers
    # ------------------------------------------------------------------

    def _ensure_obs_tables(self, con) -> None:
        """Create stations/observations tables if they don't exist yet."""
        con.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                station_id          TEXT PRIMARY KEY,
                node_idx            INTEGER,
                lon                 DOUBLE,
                lat                 DOUBLE,
                drainage_area_km2   DOUBLE
            );
            CREATE TABLE IF NOT EXISTS observations (
                station_id  TEXT,
                date        DATE,
                discharge   FLOAT,
                PRIMARY KEY (station_id, date)
            );
        """)

    def _import_from_nc(self, con, path: Path, basin_prefix: str | None) -> int:
        """Import from the stations NetCDF format."""
        import xarray as xr

        ds = xr.open_dataset(path)
        try:
            troncon_ids_raw = ds.troncon_id.values
            station_ids_raw = ds.station_id.values
            lons = ds.lon.values
            lats = ds.lat.values
            areas = ds.drainage_area.values
            times = ds.time.values
            discharge = ds.discharge.values  # (n_stations, n_time)
        finally:
            ds.close()

        # Load node_ids from the DB to resolve troncon→node_idx
        node_ids_rows = con.execute(
            "SELECT node_idx, node_id FROM nodes ORDER BY node_idx"
        ).fetchall()
        troncon_to_node = {int(r[1]): int(r[0]) for r in node_ids_rows}

        import pandas as pd

        stations_rows = []
        obs_rows = []

        for i, (sid, tid_str) in enumerate(zip(station_ids_raw, troncon_ids_raw)):
            tid_str = str(tid_str).strip()
            sid = str(sid).strip()

            # Filter by basin prefix if given
            if basin_prefix:
                prefix = basin_prefix.upper()
                if not tid_str.upper().startswith(prefix):
                    continue
                numeric_part = tid_str[len(prefix):]
            else:
                # Try to extract numeric suffix from any prefix
                numeric_part = ""
                for ch in reversed(tid_str):
                    if ch.isdigit():
                        numeric_part = ch + numeric_part
                    else:
                        break

            try:
                tid_int = int(numeric_part)
            except ValueError:
                continue

            node_idx = troncon_to_node.get(tid_int)

            stations_rows.append({
                "station_id": sid,
                "node_idx": node_idx,
                "lon": float(lons[i]),
                "lat": float(lats[i]),
                "drainage_area_km2": float(areas[i]),
            })

            # Extract discharge time-series for this station
            q = discharge[i]
            valid_mask = ~np.isnan(q)
            valid_times = times[valid_mask]
            valid_q = q[valid_mask].astype(np.float32)

            for t, qval in zip(valid_times, valid_q):
                obs_rows.append({
                    "station_id": sid,
                    "date": pd.Timestamp(t).date(),
                    "discharge": float(qval),
                })

        if not stations_rows:
            return 0

        stations_df = pd.DataFrame(stations_rows)
        con.execute(
            "INSERT OR REPLACE INTO stations SELECT * FROM stations_df"
        )

        if obs_rows:
            obs_df = pd.DataFrame(obs_rows)
            con.execute(
                "INSERT OR REPLACE INTO observations SELECT * FROM obs_df"
            )

        return len(obs_rows)

    def _import_from_tabular(self, con, path: Path, fmt: str) -> int:
        """Import from CSV or Parquet (long format: station_id, date, discharge).

        Expects a ``stations`` section (station_id, node_idx, lon, lat,
        drainage_area_km2) already in the DB, or a companion
        ``<name>_stations.csv/.parquet`` file alongside.
        """
        import pandas as pd

        if fmt == "csv":
            df = pd.read_csv(path, parse_dates=["date"])
        else:
            df = pd.read_parquet(path)

        required = {"station_id", "date", "discharge"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing columns: {required - set(df.columns)}"
            )

        # Check for companion stations file
        stations_path = path.with_name(
            path.stem.replace("_observations", "") + "_stations" + path.suffix
        )
        if stations_path.exists():
            if fmt == "csv":
                sdf = pd.read_csv(stations_path)
            else:
                sdf = pd.read_parquet(stations_path)
            con.execute(
                "INSERT OR REPLACE INTO stations SELECT * FROM sdf"
            )

        obs_df = df[["station_id", "date", "discharge"]].dropna(
            subset=["discharge"]
        )
        con.execute(
            "INSERT OR REPLACE INTO observations SELECT * FROM obs_df"
        )
        return len(obs_df)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, hydro: dict, source: str = "") -> None:
        """Write a standard hydro dict to DuckDB (overwrites if exists)."""
        import duckdb

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

        con = duckdb.connect(str(self.path))
        try:
            self._create_schema(con)
            self._write_metadata(con, hydro, source)
            self._write_nodes(con, hydro)
            self._write_edges(con, hydro)
            self._write_territorial(con, hydro)
            self._write_initial_state(con, hydro)
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self, con) -> None:
        con.execute("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);

            CREATE TABLE nodes (
                node_idx        INTEGER PRIMARY KEY,
                node_id         INTEGER,
                lon             FLOAT,
                lat             FLOAT,
                is_lake         BOOLEAN,
                topo_order      INTEGER
            );

            CREATE TABLE edges (
                src             INTEGER,
                dst             INTEGER,
                edge_attr_0     FLOAT,
                edge_attr_1     FLOAT,
                edge_attr_2     FLOAT,
                travel_time_days INTEGER
            );

            CREATE TABLE initial_state (
                node_idx        INTEGER PRIMARY KEY,
                theta1          FLOAT,
                theta2          FLOAT,
                theta3          FLOAT,
                swe             FLOAT,
                t_soil          FLOAT,
                canopy_storage  FLOAT,
                wetland_storage FLOAT
            );

            CREATE TABLE stations (
                station_id          TEXT PRIMARY KEY,
                node_idx            INTEGER,
                lon                 DOUBLE,
                lat                 DOUBLE,
                drainage_area_km2   DOUBLE
            );

            CREATE TABLE observations (
                station_id  TEXT,
                date        DATE,
                discharge   FLOAT,
                PRIMARY KEY (station_id, date)
            );

            CREATE TABLE warm_states (
                state_date      TEXT,
                node_idx        INTEGER,
                theta1          FLOAT,
                theta2          FLOAT,
                theta3          FLOAT,
                swe             FLOAT,
                t_soil          FLOAT,
                canopy_storage  FLOAT,
                wetland_storage FLOAT,
                lake_storage    FLOAT,
                q_out_prev      FLOAT,
                PRIMARY KEY (state_date, node_idx)
            );

            CREATE TABLE encoder_states (
                state_date  TEXT,
                run_id      TEXT,
                data        BLOB,
                PRIMARY KEY (state_date, run_id)
            );
        """)

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def _write_metadata(self, con, hydro: dict, source: str) -> None:
        import pandas as pd

        df = pd.DataFrame([
            ("n_nodes", str(hydro["n_nodes"])),
            ("source", source),
            ("created_at", datetime.now(timezone.utc).isoformat()),
        ], columns=["key", "value"])
        con.execute("INSERT INTO metadata SELECT * FROM df")

    def _write_nodes(self, con, hydro: dict) -> None:
        import pandas as pd

        graph: RiverGraph = hydro["graph"]
        coords: Tensor = hydro["node_coords"]
        n = hydro["n_nodes"]
        topo = graph.topo_order.cpu().numpy()
        rank = np.empty(n, dtype=np.int64)
        rank[topo] = np.arange(n)

        df = pd.DataFrame({
            "node_idx": np.arange(n),
            "node_id": np.array(hydro["node_ids"], dtype=np.int64),
            "lon": coords[:, 0].cpu().numpy(),
            "lat": coords[:, 1].cpu().numpy(),
            "is_lake": graph.is_lake.cpu().numpy(),
            "topo_order": rank,
        })
        con.execute("INSERT INTO nodes SELECT * FROM df")

    def _write_edges(self, con, hydro: dict) -> None:
        import pandas as pd

        graph: RiverGraph = hydro["graph"]
        if graph.n_edges == 0:
            return
        ei = graph.edge_index.cpu().numpy()
        ea = graph.edge_attr.cpu().numpy()
        if ea.ndim == 1:
            ea = ea[:, None]
        if ea.shape[1] < 3:
            ea = np.pad(ea, ((0, 0), (0, 3 - ea.shape[1])))
        tt = graph.travel_time_days.cpu().numpy()

        df = pd.DataFrame({
            "src": ei[0], "dst": ei[1],
            "edge_attr_0": ea[:, 0],
            "edge_attr_1": ea[:, 1],
            "edge_attr_2": ea[:, 2],
            "travel_time_days": tt,
        })
        con.execute("INSERT INTO edges SELECT * FROM df")

    def _write_territorial(self, con, hydro: dict) -> None:
        import pandas as pd

        t: TerritorialFeatures = hydro["territorial"]
        n = hydro["n_nodes"]
        df = pd.DataFrame({"node_idx": np.arange(n)})
        # Write feature columns (normalised)
        for i, col in enumerate(t.columns):
            df[col] = t.data[:, i].cpu().numpy()
        # Write physical columns (un-normalised)
        for name, tensor in t.physical.items():
            df[name] = tensor.cpu().numpy()
        # Create table dynamically from DataFrame schema
        con.execute("CREATE TABLE territorial AS SELECT * FROM df")
        con.execute(
            "CREATE UNIQUE INDEX idx_territorial_node ON territorial(node_idx)"
        )

    def _write_initial_state(self, con, hydro: dict) -> None:
        import pandas as pd

        s: HydroState = hydro["initial_state"]
        n = hydro["n_nodes"]
        df = pd.DataFrame({"node_idx": np.arange(n)})
        for f in _HYDROSTATE_FIELDS:
            df[f] = getattr(s, f).cpu().numpy()
        con.execute("INSERT INTO initial_state SELECT * FROM df")

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    def _load_graph(self, con, device) -> RiverGraph:
        nodes_df = con.execute(
            "SELECT node_idx, is_lake, topo_order FROM nodes ORDER BY node_idx"
        ).df()
        edges_df = con.execute(
            "SELECT src, dst, edge_attr_0, edge_attr_1, edge_attr_2, "
            "travel_time_days FROM edges"
        ).df()

        n = len(nodes_df)
        rank = nodes_df["topo_order"].to_numpy()
        topo_order = np.argsort(rank).astype(np.int64)

        is_lake = torch.tensor(nodes_df["is_lake"].to_numpy(), dtype=torch.bool,
                               device=device)
        topo_order_t = torch.tensor(topo_order, dtype=torch.long, device=device)

        if edges_df.empty:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.zeros((0, 3), dtype=torch.float32, device=device)
            travel_time_days = torch.zeros(0, dtype=torch.long, device=device)
        else:
            ei = edges_df[["src", "dst"]].to_numpy().T.astype(np.int64)
            ea = edges_df[["edge_attr_0", "edge_attr_1", "edge_attr_2"]].to_numpy()
            tt = edges_df["travel_time_days"].to_numpy().astype(np.int64)
            edge_index = torch.tensor(ei, dtype=torch.long, device=device)
            edge_attr = torch.tensor(ea, dtype=torch.float32, device=device)
            travel_time_days = torch.tensor(tt, dtype=torch.long, device=device)

        return RiverGraph(
            edge_index=edge_index,
            edge_attr=edge_attr,
            topo_order=topo_order_t,
            is_lake=is_lake,
            travel_time_days=travel_time_days,
        )

    def _load_territorial(self, con, device) -> TerritorialFeatures:
        df = con.execute(
            "SELECT * FROM territorial ORDER BY node_idx"
        ).df()

        # Separate feature columns from physical columns
        all_cols = [c for c in df.columns if c != "node_idx"]
        # Colonnes physiques : DEFAULT_PHYSICAL_COLUMNS + convention suffixe _raw
        # (fractions brutes occupation/texture pour le split BV3C2). Jamais au NeRF.
        physical_cols = [c for c in all_cols
                         if c in DEFAULT_PHYSICAL_COLUMNS or c.endswith("_raw")]
        feature_cols = [c for c in all_cols if c not in physical_cols]

        # Build normalised feature tensor
        feature_data = torch.tensor(
            df[feature_cols].to_numpy(), dtype=torch.float32, device=device,
        )

        # Build physical dict
        physical = {}
        for col in physical_cols:
            physical[col] = torch.tensor(
                df[col].to_numpy(), dtype=torch.float32, device=device,
            )

        return TerritorialFeatures(
            data=feature_data, columns=feature_cols, physical=physical,
        )

    def _load_nodes(self, con, device) -> tuple[Tensor, list[int]]:
        df = con.execute(
            "SELECT node_id, lon, lat FROM nodes ORDER BY node_idx"
        ).df()
        coords = torch.tensor(df[["lon", "lat"]].to_numpy(), dtype=torch.float32,
                              device=device)
        return coords, df["node_id"].tolist()

    def _load_initial_state(self, con, device) -> HydroState:
        df = con.execute(
            "SELECT * FROM initial_state ORDER BY node_idx"
        ).df()

        def _t(col: str) -> Tensor:
            return torch.tensor(df[col].to_numpy(), dtype=torch.float32,
                                device=device)

        n = len(df)
        return HydroState(
            theta1=_t("theta1"), theta2=_t("theta2"), theta3=_t("theta3"),
            swe=_t("swe"), t_soil=_t("t_soil"),
            canopy_storage=_t("canopy_storage"),
            wetland_storage=_t("wetland_storage"),
            S_gw=torch.zeros(n, dtype=torch.float32, device=device),
            T_water=torch.full((n,), 10.0, dtype=torch.float32, device=device),
        )
