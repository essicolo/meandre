"""L'erreur hors bloc évalue le bloc tenu de côté, et lui seul quand on le demande.

Le piège du 2026-09-20 : un découpage en deux blocs, le territoire d'un côté et le reste de
l'autre, moyenne les DEUX sens du partage. La prédiction du reste par le territoire entre alors
dans un chiffre présenté comme celui du territoire.
"""
import os

import numpy as np
import pytest
from importlib.machinery import SourceFileLoader

_F = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  ".runs", "quebec", "identifiabilite_indice_base.py")
pytest.importorskip("sklearn")
_m = SourceFileLoader("idb", _F).load_module()


def _jeu(n_par_bloc=30, graine=0, decalage=5.0):
    """Trois blocs. Les deux premiers portent une relation nette, le bloc 2 est PLAT.

    `decalage` éloigne le bloc plat de la moyenne des autres. À zéro, il tombe dessus : son
    erreur devient minuscule alors que sa part de variance expliquée part très bas, parce que
    le dénominateur de celle-ci est sa propre dispersion.
    """
    rng = np.random.default_rng(graine)
    X, y, blocs = [], [], []
    for b, (pente, bruit) in enumerate([(1.0, 0.1), (1.0, 0.1), (0.0, 0.0)]):
        x = rng.normal(size=(n_par_bloc, 2))
        X.append(x)
        y.append(pente * x[:, 0] + rng.normal(scale=bruit, size=n_par_bloc) + decalage * (b == 2))
        blocs.append(np.full(n_par_bloc, b))
    return np.vstack(X), np.concatenate(y), np.concatenate(blocs)


def test_bloc_evalue_ne_melange_pas_les_deux_sens():
    X, y, blocs = _jeu()
    deux = np.where(blocs == 2, 2, 0)
    melange, _ = _m.erreur_hors_bloc(X, y, deux)
    seul, _ = _m.erreur_hors_bloc(X, y, deux, evalue=2)
    assert seul > melange, "le bloc 2 est le plus dur : l'isoler doit donner une erreur plus grande"


def test_temoin_est_la_moyenne_du_reste():
    X, y, blocs = _jeu()
    _, temoin = _m.erreur_hors_bloc(X, y, blocs, evalue=2)
    attendu = np.abs(y[blocs == 2] - y[blocs != 2].mean()).mean()
    assert temoin == pytest.approx(attendu, rel=1e-9)


def test_covariables_informatives_battent_le_temoin():
    X, y, blocs = _jeu()
    err, temoin = _m.erreur_hors_bloc(X, y, blocs, evalue=0)
    assert err < temoin


def test_bloc_absent_rend_des_valeurs_manquantes():
    X, y, blocs = _jeu()
    err, temoin = _m.erreur_hors_bloc(X, y, blocs, evalue=99)
    assert np.isnan(err) and np.isnan(temoin)


def test_part_expliquee_seffondre_sur_un_bloc_sans_dispersion():
    """La raison pour laquelle l'outil ne rend plus la part de variance par défaut."""
    X, y, blocs = _jeu(decalage=0.0)
    err, temoin = _m.erreur_hors_bloc(X, y, blocs, evalue=2)
    part = _m.part_expliquee(X, y, blocs, evalue=2)
    assert err < 1.0, "l'erreur sur le bloc plat reste petite"
    assert temoin < 0.2, "le témoin aussi : ce bloc tombe sur la moyenne des autres"
    assert part < -5.0, "la part de variance, elle, s'effondre sur un dénominateur minuscule"
