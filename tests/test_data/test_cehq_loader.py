"""Drapeaux de qualite du CEHQ (registre R19).

Ce qui est verrouille ici : la lecture du format publie, et surtout la REGLE qui decide
qu'une valeur est une reconstruction plutot qu'une mesure. C'est cette regle qui donne
les 87.3 % de fevrier en tenue de cote, chiffre sur lequel repose la disqualification du
chantier hivernal comme cible d'ajustement -- et, symetriquement, la confirmation que le
deficit d'avril (9.5 % seulement) porte sur des jours reellement mesures.
"""
import numpy as np
import pandas as pd
import pytest

from meandre.data.cehq_loader import (RECONSTRUCTED_FLAGS, is_reconstructed,
                                      parse_cehq_text)

# Extrait fidele d'un fichier NNNNNN_Q.txt : en-tete, lexique, tableau. On garde les
# colonnes alignees par des espaces comme le CEHQ les publie, y compris les lignes sans
# debit (lacunes de station) qui ne doivent PAS disparaitre a l'analyse.
_FICHIER = """Ministere de l'Environnement
Donnees validees jusqu'au 2025-09-30, preliminaires par la suite
Station:        040830         Gatineau

Lexique:        E:  La donnee est estimee.
(Remarque)      R:  Le debit est corrige pour tenir compte de l'effet de refoulement.

Station        Date                Debit (m3/s)   Remarque
040830         2020/01/01          12.5           R
040830         2020/01/02          13.0           E
040830         2020/01/03          14.0           MJ R
040830         2020/01/04
040830         2020/07/01          45.2           MC
040830         2020/07/02          44.0
040830         2020/07/03          43.1           P*
"""


def test_parse_lit_valeurs_et_remarques():
    d = parse_cehq_text(_FICHIER)
    assert len(d) == 7, "les lignes SANS debit sont conservees : une lacune est une information"
    assert list(d.station_id.unique()) == ["040830"]
    assert d.date.iloc[0] == pd.Timestamp("2020-01-01")
    assert d.discharge.iloc[0] == 12.5
    assert np.isnan(d.discharge.iloc[3]), "la ligne sans debit doit rendre NaN, pas decaler"
    assert d.remark.iloc[3] == "", "et ne doit pas inventer de remarque"


def test_parse_ne_confond_pas_remarque_et_debit():
    """Piege du format : quand le debit manque, le 3e champ EST la remarque.

    Sans ce cas, une remarque se retrouverait lue comme une valeur de debit (ou ferait
    lever), et une lacune deviendrait une observation fantome.
    """
    d = parse_cehq_text(_FICHIER + "040830         2020/02/01                         R\n")
    assert np.isnan(d.discharge.iloc[-1])
    assert d.remark.iloc[-1] == "R"


@pytest.mark.parametrize("remarque, attendu", [
    ("R", True),        # refoulement : courbe de tarage jugee inapplicable
    ("E", True),        # estimee
    ("MJ R", True),     # codes cumules
    ("MC", False),      # debit moyen converti : provenance, pas mise en doute
    ("MJ", False),      # moyenne journaliere
    ("J", False),       # un jaugeage a eu lieu : c'est la MEILLEURE donnee possible
    ("", False),
    ("P", False),       # provisoire, mais mesuree
])
def test_regle_de_reconstruction(remarque, attendu):
    assert is_reconstructed(remarque) is attendu


def test_p_etoile_nest_pas_une_reconstruction():
    """`P*` signifie provisoire ET NON ESTIMEE : l'etoile inverse le sens.

    Le `E` de `P*` ne doit pas etre lu comme le code `E`, et plus generalement aucune
    sous-chaine ne doit compter : les codes sont des jetons separes par des espaces.
    """
    assert is_reconstructed("P*") is False
    assert is_reconstructed("MC") is False, "le C de MC n'est pas un code a lui seul"


def test_regle_robuste_aux_entrees_vides():
    for x in (None, float("nan"), 0, ""):
        assert is_reconstructed(x) is False


def test_les_deux_codes_sont_bien_ceux_documentes():
    """Garde-fou : elargir cette liste change TOUS les chiffres de R19."""
    assert RECONSTRUCTED_FLAGS == ("E", "R")
