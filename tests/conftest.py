"""Garde-fous communs à toute la suite.

Trois modules (nappe apprise, phénologie par classe, nappe libre) posent la double précision par
torch.set_default_dtype sans la rendre, et huit tests du routage par opérateur et de l'encodeur
de contexte échouaient alors pour un simple mélange Float/Double, uniquement quand la suite
tournait en entier (mesuré 2026-10-07). Le dtype par défaut est rendu après chaque test.
"""
import pytest
import torch


@pytest.fixture(autouse=True)
def _dtype_par_defaut_rendu():
    ancien = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(ancien)
