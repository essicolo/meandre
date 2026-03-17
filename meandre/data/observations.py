"""Gauging station observations loader.

Maps observed streamflow time-series to node indices in the river graph.
Missing values are represented as NaN and masked in the loss function.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class ObservationDataset:
    """Daily streamflow observations at gauging stations.

    Parameters
    ----------
    path : str or Path
        CSV or NetCDF with columns/variables per station.
    station_node_map : dict[str, int]
        Maps station ID to node index in the river graph.
    device : torch.device, optional
    """

    def __init__(
        self,
        path: str | Path,
        station_node_map: dict[str, int],
        n_nodes: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.path = Path(path)
        self.station_node_map = station_node_map
        self.n_nodes = n_nodes
        self.device = device
        self._data: Tensor | None = None   # (n_timesteps, n_stations)
        self._station_ids: list[str] | None = None

    def load(self) -> "ObservationDataset":
        """Load observations from disk."""
        import pandas as pd

        df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        station_ids = [s for s in self.station_node_map if s in df.columns]
        self._station_ids = station_ids
        data_np = df[station_ids].values.astype(np.float32)
        self._data = torch.from_numpy(data_np).to(self.device)
        return self

    @property
    def data(self) -> Tensor:
        if self._data is None:
            raise RuntimeError("Call .load() before accessing data")
        return self._data

    @property
    def station_mask(self) -> Tensor:
        """Boolean mask of shape (n_nodes,) — True at nodes with observations.

        Requires ``n_nodes`` to be set at construction time and ``.load()``
        to have been called so that ``_station_ids`` is populated.
        """
        if self._station_ids is None:
            raise RuntimeError("Call .load() before accessing station_mask")
        if self.n_nodes is None:
            raise RuntimeError(
                "Pass n_nodes= to ObservationDataset to build station_mask"
            )
        mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        for sid in self._station_ids:
            node_idx = self.station_node_map.get(sid)
            if node_idx is not None and 0 <= node_idx < self.n_nodes:
                mask[node_idx] = True
        return mask

    @property
    def station_indices(self) -> Tensor:
        """Integer indices (n_stations,) of nodes with observations."""
        return self.station_mask.nonzero(as_tuple=False).squeeze(-1)

    @property
    def n_stations(self) -> int:
        return self.data.shape[1]

    def __len__(self) -> int:
        return self.data.shape[0]
