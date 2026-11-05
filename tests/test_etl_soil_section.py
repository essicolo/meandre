"""Le pilote régional doit démarrer sur une configuration SANS processus de sol déclarés.

Huit passes ont été perdues la nuit du 2026-09-20 parce que le pilote testait la présence de la
SECTION `[soil]` et non celle du PROFIL. Toutes les configurations existantes portent cette
section, pour les épaisseurs de couches et le répertoire de calage, sans y déclarer le moindre
processus ; `from_toml` rend alors None, et le pilote plantait aussitôt sur `_profile.layers`.

Le garde-fou contre un profil vide en avait donc créé un autre. Ce fichier épingle les deux
cas, avec les vraies sections des configurations du dépôt.
"""
import glob
import tomllib

import pytest

from meandre.vertical import soil_processes as sp


def _real_sections():
    for f in sorted(glob.glob(".runs/quebec/*.toml") + glob.glob(".runs/slso/config/*.toml") + glob.glob("experiments/**/config/*.toml", recursive=True)):
        with open(f, "rb") as fh:
            try:
                cfg = tomllib.load(fh)
            except tomllib.TOMLDecodeError:
                continue
        if "soil" in cfg and not _retired_forms(f, cfg["soil"]):
            yield f, cfg["soil"]


def _retired_forms(f, section):
    """Configuration archivee qui declare une forme retiree du catalogue (Hooghoudt, PDM, TOPMODEL...) :
    elle decrit une experience passee et n'est plus lisible, ce qui est voulu."""
    raw = section.get("process") or section.get("processes") or []
    raw = [raw] if isinstance(raw, dict) else raw
    return "archive" in f.replace("\\", "/").split("/") and any(e.get("form") not in sp.FORMS for e in raw)


def test_repository_configurations_do_not_crash():
    """Chaque section `[soil]` réelle donne soit None, soit un profil utilisable."""
    vues = 0
    for f, section in _real_sections():
        vues += 1
        profil = sp.from_toml(section)
        if profil is None:
            continue
        assert profil.layers >= 1, f"{f} : profil sans couche"
        assert profil.processes, f"{f} : profil rendu mais sans processus"
    assert vues > 0, "aucune configuration ne porte de section [soil] : le test ne verifie rien"


def test_at_least_one_configuration_declares_a_profile():
    """Sans cela, le chemin declaratif ne serait exerce par aucune configuration."""
    declarants = [f for f, s in _real_sections() if sp.from_toml(s) is not None]
    assert declarants, "aucune configuration ne declare de processus de sol"


def test_pilot_pattern_follows_profile_not_section():
    """Le test qui plantait : `if section` au lieu de `if profil is not None`."""
    section_without_process = {"z1": 0.1, "z2_min": 0.3, "hydrotel_calib_dir": "/x"}
    assert bool(section_without_process) is True, "la section est vraie, d'ou le piege"
    assert sp.from_toml(section_without_process) is None
