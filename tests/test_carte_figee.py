"""Gel par valeur des sorties du champ spatial, pour un depart a chaud (2026-10-01).

Le gel par poids nuls (`freeze_outputs`) ne tient pas apres un chargement : la ligne de
fc_out garde ses poids appris et la sortie suit les couches cachees. La carte capturee au
premier appel doit rester identique quand le reste du reseau change.
"""
import torch

from meandre.spatial.field_network import SpatialParams, SpatialFieldNetwork


def _sp(valeur):
    noms = [f.name for f in SpatialParams.__dataclass_fields__.values()]
    return SpatialParams.from_tensor(torch.full((5, len(noms)), float(valeur)))


def test_carte_capturee_puis_substituee():
    net = SpatialFieldNetwork.__new__(SpatialFieldNetwork)
    torch.nn.Module.__init__(net)
    net.figer_au_prochain_appel = {"K_sat_1"}
    premier = net._applique_multiplicateurs(_sp(0.2))
    second = net._applique_multiplicateurs(_sp(0.7))
    assert torch.allclose(premier.K_sat_1, torch.full((5,), 0.2))
    assert torch.allclose(second.K_sat_1, torch.full((5,), 0.2))
    assert torch.allclose(second.K_sat_2, torch.full((5,), 0.7))


def test_multiplicateur_applique_apres_le_gel():
    net = SpatialFieldNetwork.__new__(SpatialFieldNetwork)
    torch.nn.Module.__init__(net)
    net.figer_au_prochain_appel = {"K_sat_1"}
    net.multiplicateurs = {"K_sat_1": torch.tensor(0.3)}
    net._applique_multiplicateurs(_sp(0.2))
    sortie = net._applique_multiplicateurs(_sp(0.9))
    assert torch.allclose(sortie.K_sat_1, torch.full((5,), 0.06))
