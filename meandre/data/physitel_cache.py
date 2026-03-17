"""Backward-compatibility shim — use basin_cache.BasinCache directly."""
from meandre.data.basin_cache import BasinCache

# Legacy alias
PhysitelCache = BasinCache

__all__ = ["BasinCache", "PhysitelCache"]
