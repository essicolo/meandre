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


def _sections_reelles():
    for f in sorted(glob.glob(".runs/quebec/config/*.toml") + glob.glob(".runs/slso/config/*.toml")):
        with open(f, "rb") as fh:
            try:
                cfg = tomllib.load(fh)
            except tomllib.TOMLDecodeError:
                continue
        if "soil" in cfg:
            yield f, cfg["soil"]


def test_les_configurations_du_depot_ne_plantent_pas():
    """Chaque section `[soil]` réelle donne soit None, soit un profil utilisable."""
    vues = 0
    for f, section in _sections_reelles():
        vues += 1
        profil = sp.from_toml(section)
        if profil is None:
            continue
        assert profil.layers >= 1, f"{f} : profil sans couche"
        assert profil.processes, f"{f} : profil rendu mais sans processus"
    assert vues > 0, "aucune configuration ne porte de section [soil] : le test ne verifie rien"


def test_au_moins_une_configuration_declare_un_profil():
    """Sans cela, le chemin declaratif ne serait exerce par aucune configuration."""
    declarants = [f for f, s in _sections_reelles() if sp.from_toml(s) is not None]
    assert declarants, "aucune configuration ne declare de processus de sol"


def test_le_motif_du_pilote_est_celui_du_profil_et_non_de_la_section():
    """Le test qui plantait : `if section` au lieu de `if profil is not None`."""
    section_sans_processus = {"z1": 0.1, "z2_min": 0.3, "hydrotel_calib_dir": "/x"}
    assert bool(section_sans_processus) is True, "la section est vraie, d'ou le piege"
    assert sp.from_toml(section_sans_processus) is None
