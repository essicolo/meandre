"""Territorial features — per-node static GIS descriptors for the spatial network.

Dynamic: any number of feature columns. The DuckDB ``territorial`` table can
have arbitrary columns; the loader reads all numeric columns and separates
them into *network features* (normalised, fed to the spatial field network)
and *physical columns* (un-normalised, used by routing/vertical).

Physical columns are accessed by name via ``get_physical(name)`` or attribute
access (``territorial.area_km2_local``).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Columns that are stored raw and NOT passed to the spatial network.
# They are used directly by the physics modules (routing, vertical).
DEFAULT_PHYSICAL_COLUMNS = frozenset({
    "node_idx",
    "area_km2_physical",
    "area_km2_local",
    "slope_fraction",
    "depth_to_bedrock_m",
    "reach_length_m",
})


class TerritorialFeatures:
    """Per-node static features — dynamic number of columns.

    Parameters
    ----------
    data : Tensor
        (n_nodes, n_features) normalised feature tensor.
    columns : list[str]
        Feature names matching ``data`` columns, in order.
    physical : dict[str, Tensor]
        Un-normalised physical fields (area, slope, etc.), keyed by name.
        Not included in ``to_tensor()``.
    """

    def __init__(
        self,
        data: Tensor,
        columns: list[str],
        physical: dict[str, Tensor] | None = None,
    ) -> None:
        assert data.ndim == 2, f"Expected (n_nodes, n_features), got {data.shape}"
        assert data.shape[1] == len(columns), (
            f"data has {data.shape[1]} columns but {len(columns)} names given"
        )
        self.data = data
        self.columns = list(columns)
        self.physical = dict(physical) if physical else {}

    def to_tensor(self) -> Tensor:
        """(n_nodes, n_features) normalised tensor for the spatial network."""
        return self.data

    @property
    def n_features(self) -> int:
        return self.data.shape[1]

    @property
    def n_nodes(self) -> int:
        return self.data.shape[0]

    def get_physical(self, name: str) -> Tensor | None:
        """Get an un-normalised physical column by name, or None."""
        return self.physical.get(name)

    def to(self, device: torch.device) -> "TerritorialFeatures":
        """Move all tensors to device."""
        return TerritorialFeatures(
            data=self.data.to(device),
            columns=self.columns,
            physical={k: v.to(device) for k, v in self.physical.items()},
        )

    def __getattr__(self, name: str):
        """Fallback attribute access for physical columns (backward compat)."""
        if name in ("data", "columns", "physical"):
            raise AttributeError(name)
        phys = object.__getattribute__(self, "physical")
        if name in phys:
            return phys[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available physical columns: {list(phys.keys())}"
        )

    @classmethod
    def zeros(
        cls,
        n_nodes: int,
        n_features: int = 17,
        device: torch.device | None = None,
    ) -> "TerritorialFeatures":
        """Create zero-filled features (for tests / synthetic basins)."""
        data = torch.zeros(n_nodes, n_features, device=device)
        columns = [f"feature_{i}" for i in range(n_features)]
        return cls(data=data, columns=columns)


# Backward-compatible alias
TerritorialIndicators = TerritorialFeatures
