"""Manifeste d'un domaine : d'où vient chaque pièce, et son empreinte.

Le 2026-10-26, la table des attributs du champ différait entre deux machines depuis six semaines
(distance à l'exutoire fausse sur l'une) sans qu'aucun contrôle le signale, et le modèle
provincial avait été entraîné sur la mauvaise. Le manifeste rend cet écart visible : chaque
fichier qui entre dans un domaine y est inscrit avec son empreinte, le manifeste est capturé
dans la recette du point de reprise, et un chargement sur d'autres fichiers le dit.

L'empreinte est un SHA-1 des premiers et derniers mégaoctets plus la taille : assez pour
distinguer deux versions d'un fichier, sans relire des gigaoctets de forçage à chaque lancement.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Source:
    path: str
    fingerprint: str
    size: int
    modified: str
    origin: str = ""     # producteur ou produit (PHYSITEL, CaSR, MOD16...), en clair
    role: str = ""       # ce que la pièce fournit au domaine (network, attributes, forcing...)


@dataclass
class Manifest:
    name: str
    sources: dict[str, Source] = field(default_factory=dict)

    def add(self, key: str, path, origin: str = "", role: str = "") -> Source:
        s = source_of(path, origin, role)
        self.sources[key] = s
        return s

    def as_dict(self) -> dict:
        return {k: {"path": s.path, "fingerprint": s.fingerprint, "size": s.size, "modified": s.modified, "origin": s.origin, "role": s.role} for k, s in sorted(self.sources.items())}

    def differences(self, saved: dict | None) -> list[str]:
        """Écarts entre ce manifeste et un dictionnaire sauvegardé (`as_dict`), en phrases."""
        if not saved:
            return []
        out = []
        for k in sorted(set(saved) | set(self.sources)):
            a, c = saved.get(k), self.sources.get(k)
            if a is None:
                out.append(f"domaine : {k} présent maintenant, absent à l'entraînement")
            elif c is None:
                out.append(f"domaine : {k} présent à l'entraînement, absent maintenant")
            elif a.get("fingerprint") != c.fingerprint:
                out.append(f"domaine : {k} n'a pas la même empreinte qu'à l'entraînement ({os.path.basename(c.path)}, modifié {c.modified} contre {a.get('modified')})")
        return out


def fingerprint(path, max_bytes: int = 4 * 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha1()
    size = p.stat().st_size
    with open(p, "rb") as f:
        h.update(f.read(max_bytes))
        if size > 2 * max_bytes:
            f.seek(size - max_bytes)
            h.update(f.read(max_bytes))
    h.update(str(size).encode())
    return h.hexdigest()[:16]


def fingerprint_dir(path, patterns: tuple[str, ...] = ("*",)) -> tuple[str, int, str]:
    """Empreinte, taille et date d'un dossier : agrège celles de ses fichiers, triés."""
    p = Path(path)
    files = sorted({f for pat in patterns for f in p.rglob(pat) if f.is_file()})
    h = hashlib.sha1()
    size, latest = 0, 0.0
    for f in files:
        st = f.stat()
        h.update(f"{f.relative_to(p)}|{st.st_size}|{int(st.st_mtime)}".encode())
        size += st.st_size
        latest = max(latest, st.st_mtime)
    return h.hexdigest()[:16], size, datetime.fromtimestamp(latest).isoformat(timespec="seconds") if files else ""


def source_of(path, origin: str = "", role: str = "") -> Source:
    p = Path(path)
    if p.is_dir():
        fp, size, modified = fingerprint_dir(p)
    else:
        st = p.stat()
        fp, size, modified = fingerprint(p), st.st_size, datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return Source(path=str(p), fingerprint=fp, size=size, modified=modified, origin=origin, role=role)


# Manifeste du domaine en cours : posé par l'adaptateur, lu par la recette à la sauvegarde.
_current: Manifest | None = None


def register(manifest: Manifest) -> None:
    global _current
    _current = manifest


def current() -> Manifest | None:
    return _current
