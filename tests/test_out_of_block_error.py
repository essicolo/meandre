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
                  "experiments", "identifiability", "baseflow-index-identifiability.py")
pytest.importorskip("sklearn")
_m = SourceFileLoader("idb", _F).load_module()


def _jeu(n_per_block=30, seed=0, shift=5.0):
    """Trois blocs. Les deux premiers portent une relation nette, le bloc 2 est PLAT.

    `decalage` éloigne le bloc plat de la moyenne des autres. À zéro, il tombe dessus : son
    erreur devient minuscule alors que sa part de variance expliquée part très bas, parce que
    le dénominateur de celle-ci est sa propre dispersion.
    """
    rng = np.random.default_rng(seed)
    X, y, blocks = [], [], []
    for b, (pente, noise) in enumerate([(1.0, 0.1), (1.0, 0.1), (0.0, 0.0)]):
        x = rng.normal(size=(n_per_block, 2))
        X.append(x)
        y.append(pente * x[:, 0] + rng.normal(scale=noise, size=n_per_block) + shift * (b == 2))
        blocks.append(np.full(n_per_block, b))
    return np.vstack(X), np.concatenate(y), np.concatenate(blocks)


def test_evaluated_block_does_not_mix_directions():
    X, y, blocks = _jeu()
    two = np.where(blocks == 2, 2, 0)
    mix, _ = _m.erreur_hors_bloc(X, y, two)
    only, _ = _m.erreur_hors_bloc(X, y, two, evalue=2)
    assert only > mix, "le bloc 2 est le plus dur : l'isoler doit donner une erreur plus grande"


def test_control_is_mean_of_the_rest():
    X, y, blocks = _jeu()
    _, control = _m.erreur_hors_bloc(X, y, blocks, evalue=2)
    expected = np.abs(y[blocks == 2] - y[blocks != 2].mean()).mean()
    assert control == pytest.approx(expected, rel=1e-9)


def test_informative_covariates_beat_the_control():
    X, y, blocks = _jeu()
    err, control = _m.erreur_hors_bloc(X, y, blocks, evalue=0)
    assert err < control


def test_missing_block_returns_missing_values():
    X, y, blocks = _jeu()
    err, control = _m.erreur_hors_bloc(X, y, blocks, evalue=99)
    assert np.isnan(err) and np.isnan(control)


def test_explained_share_collapses_on_block_without_dispersion():
    """La raison pour laquelle l'outil ne rend plus la part de variance par défaut."""
    X, y, blocks = _jeu(shift=0.0)
    err, control = _m.erreur_hors_bloc(X, y, blocks, evalue=2)
    part = _m.part_expliquee(X, y, blocks, evalue=2)
    assert err < 1.0, "l'erreur sur le bloc plat reste petite"
    assert control < 0.2, "le témoin aussi : ce bloc tombe sur la moyenne des autres"
    assert part < -5.0, "la part de variance, elle, s'effondre sur un dénominateur minuscule"
