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
