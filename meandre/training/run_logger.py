"""DuckDB-backed experiment run logger.

Lightweight replacement for MLflow: stores runs, hyperparameters, and
per-epoch metrics in a single DuckDB file with no server required.

Schema
------
runs     run_id, name, created_at, ended_at, status
params   run_id, key, value (TEXT)
metrics  run_id, step, key, value (FLOAT)

Typical usage
-------------
logger = RunLogger("notebooks/runs.duckdb")
logger.start_run("slso_phase1")
logger.log_params({"lr": 5e-4, "n_epochs": 300})

for epoch in range(n_epochs):
    ...
    logger.log_metrics({"train_loss": loss, "val_nse": nse}, step=epoch)

logger.end_run()

# Analysis
df = logger.metrics_df(logger.last_run_id, key="val_nse")
best = logger.best_run("val_nse", mode="max")
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    """DuckDB-backed experiment tracker.

    Uses a single persistent connection to avoid Windows file-locking issues
    when opening/closing a connection per call.

    Parameters
    ----------
    path : str | Path
        Path to the DuckDB file.  Created if absent.
    """

    def __init__(self, path: str | Path) -> None:
        import duckdb

        self.path = Path(path)
        self._run_id: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._ensure_schema()

    def close(self) -> None:
        """Close the underlying DuckDB connection and release the file lock."""
        if self._run_id is not None:
            self.end_run(status="KILLED")
        try:
            self._con.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, name: str = "") -> str:
        """Open a new run and return its run_id.

        Any previously active run is closed automatically.
        """
        if self._run_id is not None:
            self.end_run(status="KILLED")

        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        self._con.execute(
            "INSERT INTO runs (run_id, name, created_at, status) VALUES (?,?,?,?)",
            [run_id, name, created_at, "RUNNING"],
        )
        self._run_id = run_id
        return run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """Close the active run."""
        if self._run_id is None:
            return
        ended_at = datetime.now(timezone.utc).isoformat()
        self._con.execute(
            "UPDATE runs SET ended_at = ?, status = ? WHERE run_id = ?",
            [ended_at, status, self._run_id],
        )
        self._run_id = None

    @property
    def last_run_id(self) -> str | None:
        """The currently active run_id, or None."""
        return self._run_id

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_params(self, params: dict) -> None:
        """Log hyperparameters for the active run."""
        import pandas as pd

        if self._run_id is None:
            raise RuntimeError("No active run — call start_run() first.")

        df = pd.DataFrame([
            {"run_id": self._run_id, "key": str(k), "value": str(v)}
            for k, v in params.items()
        ])
        self._con.execute("INSERT INTO params SELECT * FROM df")

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a single metric value."""
        if self._run_id is None:
            raise RuntimeError("No active run — call start_run() first.")

        self._con.execute(
            "INSERT INTO metrics VALUES (?, ?, ?, ?)",
            [self._run_id, step, key, float(value)],
        )

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metric values at the same step."""
        import pandas as pd

        if self._run_id is None:
            raise RuntimeError("No active run — call start_run() first.")

        df = pd.DataFrame([
            {"run_id": self._run_id, "step": step, "key": k, "value": float(v)}
            for k, v in metrics.items()
        ])
        self._con.execute("INSERT INTO metrics SELECT * FROM df")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def runs_df(self):
        """Return a DataFrame with all runs (id, name, status, duration)."""
        return self._con.execute("SELECT * FROM runs ORDER BY created_at").df()

    def metrics_df(self, run_id: str, key: str | None = None):
        """Return metrics DataFrame for a run, optionally filtered by key."""
        if key is not None:
            return self._con.execute(
                "SELECT step, key, value FROM metrics "
                "WHERE run_id = ? AND key = ? ORDER BY step",
                [run_id, key],
            ).df()
        return self._con.execute(
            "SELECT step, key, value FROM metrics "
            "WHERE run_id = ? ORDER BY step, key",
            [run_id],
        ).df()

    def params_df(self, run_id: str):
        """Return params DataFrame for a run."""
        return self._con.execute(
            "SELECT key, value FROM params WHERE run_id = ?", [run_id]
        ).df()

    def best_run(self, metric: str, mode: str = "max") -> dict:
        """Return the run with the best value for a metric.

        Parameters
        ----------
        metric : str
            Metric key, e.g. ``"val_nse"``.
        mode : ``"max"`` | ``"min"``
        """
        agg = "MAX" if mode == "max" else "MIN"
        row = self._con.execute(f"""
            SELECT r.run_id, r.name, {agg}(m.value) AS best_value
            FROM metrics m JOIN runs r USING (run_id)
            WHERE m.key = ?
            GROUP BY r.run_id, r.name
            ORDER BY best_value {'DESC' if mode == 'max' else 'ASC'}
            LIMIT 1
        """, [metric]).fetchone()

        if row is None:
            raise KeyError(f"No runs with metric '{metric}' found.")
        return {"run_id": row[0], "name": row[1], "best_value": row[2]}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id      TEXT PRIMARY KEY,
                name        TEXT,
                created_at  TEXT,
                ended_at    TEXT,
                status      TEXT
            );
            CREATE TABLE IF NOT EXISTS params (
                run_id  TEXT,
                key     TEXT,
                value   TEXT
            );
            CREATE TABLE IF NOT EXISTS metrics (
                run_id  TEXT,
                step    INTEGER,
                key     TEXT,
                value   DOUBLE
            );
        """)
