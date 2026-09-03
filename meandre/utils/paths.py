"""Helpers for resolving config-relative file paths.

The TOML configs under `.runs/<case>/config/*.toml` declare paths like
`data/slso.duckdb` or `checkpoints/best.pt`. These are resolved relative
to the *run directory* (`.runs/<case>/`), which is the parent of the
config file's directory. This lets a `.runs/<case>/` tree be moved or
duplicated without editing every TOML.

Absolute paths in the TOML are returned unchanged.
"""
from __future__ import annotations

from pathlib import Path


def run_dir_from_config(config_path: str | Path) -> Path:
    """Return the run directory for a given config file.

    Convention: `.runs/<case>/config/<name>.toml` → `.runs/<case>/`.
    """
    return Path(config_path).resolve().parent.parent


def resolve_run_path(p: str | Path, run_dir: str | Path) -> Path:
    """Resolve a path declared in a TOML, relative to the run directory.

    Absolute paths are returned unchanged. Forward and backslashes both work.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return Path(run_dir) / pp


# ── Data roots, overridable by environment ──────────────────────────────────
# The repo carried 24 absolute paths hard-coded in source, making it impossible to
# run anywhere else: another workstation, a rented compute node, a container. The
# defaults are the values of the original machine, so nothing changes without an
# explicit action.
#
#   MEANDRE_DATA         derived data (DuckDB caches, forcings, field parquets)
#   MEANDRE_PLATEFORMES  Hydrotel platforms (PHYSITEL projects, calibrations)
#   MEANDRE_RQH          RQH reference outputs (post-processing zarrs)
#
# Governance note: PLATFORMS_ROOT and RQH_ROOT point at INTERNAL data. If training is
# offloaded to an external machine, those two must NOT follow -- the comparison to the
# Hydrotel ensemble stays local.
import os as _os

DATA_ROOT = _os.environ.get("MEANDRE_DATA", "D:/meandre-data")
# Two spellings accepted: the original French name and the English one used by the
# cluster scripts. A silent fallback to the workstation default cost a debugging
# round on Narval (2026-09-01): the job ran twenty minutes before failing on a
# Windows path it could not possibly reach.
PLATFORMS_ROOT = (_os.environ.get("MEANDRE_PLATEFORMES")
                  or _os.environ.get("MEANDRE_PLATFORMS")
                  or "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel")
RQH_ROOT = _os.environ.get(
    "MEANDRE_RQH",
    "C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data")


def data_path(*parts: str) -> str:
    """Path under the derived-data root."""
    return str(Path(DATA_ROOT).joinpath(*parts)).replace("\\", "/")
