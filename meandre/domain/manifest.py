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
        # Clé préfixée par le territoire (2026-11-09) : dans un domaine de plusieurs territoires, les
        # bases et les forçages de chacun s'écrasaient sous la même clé, et seul le dernier était vérifié.
        self.sources[f"{self.name}_{key}" if self.name else key] = s
        return s

    def as_dict(self) -> dict:
        return {k: {"path": s.path, "fingerprint": s.fingerprint, "size": s.size, "modified": s.modified, "origin": s.origin, "role": s.role} for k, s in sorted(self.sources.items())}

    def differences(self, saved: dict | None) -> list[str]:
        """Fichiers lus à l'entraînement ET maintenant dont l'empreinte diffère, en phrases.

        La comparaison se fait par clé (territoire et rôle du fichier), qui vaut d'une machine à
        l'autre. Pour un manifeste d'avant les clés préfixées par le territoire, elle se rabat sur le
        nom de fichier, seulement quand ce nom est unique : les quatorze territoires ont chacun leur
        `troncon.trl`, et comparer par nom les confondait (2026-11-09). Un fichier présent d'un seul
        côté n'est pas un écart de données."""
        if not saved:
            return []
        by_name = {}
        for v in saved.values():
            by_name.setdefault(os.path.basename(v.get("path", "")), []).append(v)
        out = []
        for key, c in sorted(self.sources.items()):
            # Les dossiers se comparent par le script entre machines (meandre.run.data_fingerprint) :
            # leur empreinte a changé de méthode le 2026-11-09 et ferait crier les anciens points de reprise.
            if os.path.isdir(c.path):
                continue
            a = saved.get(key)
            if a is None:
                same_name = by_name.get(os.path.basename(c.path), [])
                a = same_name[0] if len(same_name) == 1 else None
            # Une empreinte d'avant le 2026-11-09 a été calculée octet par octet, fichiers texte compris.
            if a is not None and a.get("fingerprint") != c.fingerprint and not (
                    Path(c.path).is_file() and a.get("fingerprint") == fingerprint_bytes(c.path)):
                out.append(f"domaine : {key} ({os.path.basename(c.path)}) n'a pas la même empreinte qu'à l'entraînement (modifié {c.modified} contre {a.get('modified')})")
        return sorted(set(out))


TEXT_SUFFIXES = frozenset({".csv", ".trl", ".txt", ".config", ".toml", ".json"})


def fingerprint(path, max_bytes: int = 4 * 1024 * 1024) -> str:
    """Empreinte d'un fichier. Un fichier texte se lit sans ses retours chariot (2026-11-09) : la même
    plateforme copiée sous Windows et sous Linux ne différait que par ses fins de ligne."""
    p = Path(path)
    if p.suffix.lower() in TEXT_SUFFIXES:
        data = p.read_bytes().replace(b"\r", b"")
        return hashlib.sha1(data + str(len(data)).encode()).hexdigest()[:16]
    return fingerprint_bytes(p, max_bytes)


def fingerprint_bytes(path, max_bytes: int = 4 * 1024 * 1024) -> str:
    """Empreinte octet par octet : SHA-1 des premiers et derniers mégaoctets plus la taille."""
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
        # Contenu et non date (2026-11-09) : une copie sur une autre machine change les dates.
        h.update(f"{f.relative_to(p).as_posix()}|{fingerprint(f)}".encode())
        size += st.st_size
        latest = max(latest, st.st_mtime)
    return h.hexdigest()[:16], size, datetime.fromtimestamp(latest).isoformat(timespec="seconds") if files else ""


# Tables d'une base de territoire que le modèle lit : une table ajoutée par un constructeur (barrages,
# copies datées des prélèvements) ne change pas le modèle et ne doit pas déclencher d'alerte.
TABLES_READ = frozenset({"nodes", "edges", "territorial", "territorial_siigsol", "observations", "stations", "modis_et",
                         "modis_snow", "grace_tws", "snow_obs", "snow_sites", "initial_state", "metadata", "withdrawals"})


def fingerprint_duckdb(path) -> str:
    """Empreinte du CONTENU d'une base DuckDB, table par table (nombre de lignes et somme des
    hachages de lignes) : deux bases au même contenu écrites différemment ont des fichiers
    différents, et une empreinte de fichier les distinguerait à tort (2026-11-09)."""
    import duckdb
    h = hashlib.sha1()
    con = duckdb.connect(str(path), read_only=True)
    try:
        for (table,) in sorted(con.execute("SHOW TABLES").fetchall()):
            if table not in TABLES_READ:
                continue
            n, total = con.execute(f'SELECT count(*), sum(hash(t)) FROM "{table}" t').fetchone()
            h.update(f"{table}|{n}|{total}".encode())
    finally:
        con.close()
    return h.hexdigest()[:16]


def source_of(path, origin: str = "", role: str = "") -> Source:
    p = Path(path)
    if p.is_dir():
        fp, size, modified = fingerprint_dir(p)
    else:
        st = p.stat()
        fp = fingerprint_duckdb(p) if p.suffix == ".duckdb" else fingerprint(p)
        size, modified = st.st_size, datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    return Source(path=str(p), fingerprint=fp, size=size, modified=modified, origin=origin, role=role)


# Manifeste du domaine en cours : posé par l'adaptateur, lu par la recette à la sauvegarde.
_current: Manifest | None = None


def register(manifest: Manifest | None) -> None:
    """Pose le manifeste du domaine en cours ; les territoires d'un même domaine s'y ajoutent,
    un territoire déjà présent est remplacé, et None remet à zéro."""
    global _current
    if manifest is None or _current is None or _current.name == manifest.name:
        _current = manifest
        return
    prefix = f"{manifest.name}_"
    _current.sources = {k: v for k, v in _current.sources.items() if not k.startswith(prefix)}
    _current.sources.update(manifest.sources)


def current() -> Manifest | None:
    return _current
