"""Load net withdrawal data from DuckDB or CSV.

DuckDB table schema (``withdrawals``)::

    date            DATE
    node_idx        INTEGER
    net_withdrawal  FLOAT   -- m³/s, positive = addition, negative = removal

If the table does not exist, returns ``WithdrawalData.zeros()``.

Legacy CSV support: if a directory of HYDROTEL-style CSVs is provided
(GPE.csv, PR.csv, …), they are aggregated into a net withdrawal using
consumption coefficients.  EFFLUENT is added (return flow).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from meandre.routing.withdrawals import WithdrawalData


def load_withdrawals_duckdb(
    db_path: str | Path,
    n_timesteps: int,
    n_reaches: int,
    date_start: str | None = None,
    date_end: str | None = None,
    device: torch.device | None = None,
) -> WithdrawalData:
    """Load withdrawals from a DuckDB ``withdrawals`` table.

    Returns ``WithdrawalData.zeros()`` if the table does not exist.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)

    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "withdrawals" not in tables:
        con.close()
        return WithdrawalData.zeros(n_timesteps, n_reaches, device=device)

    where = ""
    if date_start and date_end:
        where = f"WHERE date >= '{date_start}' AND date <= '{date_end}'"

    df = con.execute(
        f"SELECT date, node_idx, net_withdrawal "
        f"FROM withdrawals {where} "
        f"ORDER BY date, node_idx"
    ).df()
    con.close()

    if df.empty:
        return WithdrawalData.zeros(n_timesteps, n_reaches, device=device)

    dates = sorted(df["date"].unique())
    n_t = min(len(dates), n_timesteps)

    net = np.zeros((n_timesteps, n_reaches), dtype=np.float32)
    for i, d in enumerate(dates[:n_t]):
        day = df[df["date"] == d]
        for _, row in day.iterrows():
            idx = int(row["node_idx"])
            if idx < n_reaches:
                net[i, idx] = float(row["net_withdrawal"])

    return WithdrawalData(net=torch.tensor(net, device=device))


def load_withdrawals_csv(
    data_dir: str | Path,
    n_timesteps: int,
    n_reaches: int,
    consumption_coeff: dict[str, float] | None = None,
    device: torch.device | None = None,
) -> WithdrawalData:
    """Load HYDROTEL-style CSVs and aggregate into net withdrawal.

    Consumptive types (GPE, PR, ELEVAGE, CULTURE) are multiplied by their
    consumption coefficient and summed.  EFFLUENT is subtracted.
    """
    import pandas as pd

    data_dir = Path(data_dir)

    defaults = {
        "GPE": 0.8, "PR": 0.8, "ELEVAGE": 0.9, "CULTURE": 0.7,
    }
    if consumption_coeff:
        defaults.update(consumption_coeff)

    def _load_or_zeros(fname: str) -> np.ndarray:
        p = data_dir / fname
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df.values[:n_timesteps, :n_reaches].astype(np.float32)
        return np.zeros((n_timesteps, n_reaches), dtype=np.float32)

    net = np.zeros((n_timesteps, n_reaches), dtype=np.float32)
    for name, alpha in defaults.items():
        net -= _load_or_zeros(f"{name}.csv") * alpha  # consumptive removal → negative
    net += _load_or_zeros("EFFLUENT.csv")  # return flow → positive

    return WithdrawalData(net=torch.from_numpy(net).to(device))
