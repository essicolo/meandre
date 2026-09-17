"""Lecture du `source.toml` d'une source auxiliaire."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from meandre.utils import paths as _paths

NATURES = ("observe", "interpole", "predit")


@dataclass
class Source:
    name: str
    root: Path
    meta: dict
    ingestion: dict
    files: list[dict] = field(default_factory=list)

    @property
    def nature(self) -> str:
        return self.meta["nature"]

    def path(self, relative: str) -> Path:
        return self.root / relative


def load_source(name: str, root: str | Path | None = None) -> Source:
    """Charge `<sources>/<name>/source.toml` et vérifie les champs obligatoires."""
    base = Path(root or _paths.SOURCES_ROOT) / name
    with open(base / "source.toml", "rb") as f:
        doc = tomllib.load(f)
    meta = doc.get("source", {})
    for cle in ("nom", "producteur", "url", "licence", "version", "nature"):
        if cle not in meta:
            raise ValueError(f"{base / 'source.toml'} : champ [source].{cle} manquant")
    if meta["nature"] not in NATURES:
        raise ValueError(f"{base / 'source.toml'} : nature « {meta['nature']} » hors de {NATURES}")
    if "ingestion" not in doc:
        raise ValueError(f"{base / 'source.toml'} : section [ingestion] manquante")
    return Source(name=name, root=base, meta=meta, ingestion=doc["ingestion"], files=doc.get("fichier", []))
