"""Meteorological forcing loader — NetCDF -> per-node tensors.

Variables expected in the NetCDF:
    P        Precipitation (mm/day)
    T_min    Minimum temperature (C)
    T_max    Maximum temperature (C)
    R_n      Net radiation (MJ/m2/day)
    u2       Wind speed at 2m (m/s)
    e_a      Actual vapour pressure (kPa)

Nodes correspond to subbasins/reaches in the river graph. Spatial
interpolation/aggregation from gridded products to nodes is done as
a preprocessing step (not here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


FORCING_VARS = ["P", "T_min", "T_max", "R_n", "u2", "e_a"]
N_FORCING = len(FORCING_VARS)


class ForcingDataset:
    """Loads and serves per-node meteorological forcing as PyTorch tensors.

    Parameters
    ----------
    path : str or Path
        Path to NetCDF file or directory of per-node CSVs.
    node_ids : list of str, optional
        Subset of nodes to load. If None, load all.
    device : torch.device, optional
    """

    def __init__(
        self,
        path: str | Path,
        node_ids: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.path = Path(path)
        self.node_ids = node_ids
        self.device = device
        self._data: Tensor | None = None      # (n_timesteps, n_nodes, N_FORCING)
        self._dates: list | None = None

    def load(self) -> "ForcingDataset":
        """Load data from disk into memory."""
        try:
            import xarray as xr
        except ImportError as e:
            raise ImportError("xarray is required for forcing loading") from e

        ds = xr.open_dataset(self.path)
        arrays = [ds[v].values for v in FORCING_VARS]  # each (T, N)
        data_np = np.stack(arrays, axis=-1)  # (T, N, n_forcing)
        self._data = torch.from_numpy(data_np).float().to(self.device)
        self._dates = ds["time"].values.tolist() if "time" in ds else None
        ds.close()
        return self

    @property
    def data(self) -> Tensor:
        if self._data is None:
            raise RuntimeError("Call .load() before accessing data")
        return self._data

    @property
    def n_timesteps(self) -> int:
        return self.data.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.data.shape[1]

    def day_of_year(self) -> Tensor:
        """Return (n_timesteps,) integer day-of-year 1-366."""
        if self._dates is None:
            return torch.arange(1, self.n_timesteps + 1) % 366 + 1
        import pandas as pd
        doys = [pd.Timestamp(d).day_of_year for d in self._dates]
        return torch.tensor(doys, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.n_timesteps

    def __getitem__(self, idx: int | slice) -> Tensor:
        return self.data[idx]
