"""Gel par valeur des sorties du champ spatial, pour un depart a chaud (2026-10-01).

Le gel par poids nuls (`freeze_outputs`) ne tient pas apres un chargement : la ligne de
fc_out garde ses poids appris et la sortie suit les couches cachees. La carte capturee au
premier appel doit rester identique quand le reste du reseau change.
"""
import torch

from meandre.spatial.field_network import SpatialParams, SpatialFieldNetwork


def _sp(value):
    name_list = [f.name for f in SpatialParams.__dataclass_fields__.values()]
    return SpatialParams.from_tensor(torch.full((5, len(name_list)), float(value)))


def test_map_captured_then_substituted():
    net = SpatialFieldNetwork.__new__(SpatialFieldNetwork)
    torch.nn.Module.__init__(net)
    net.freeze_on_next_call = {"K_sat_1"}
    first = net._apply_multipliers(_sp(0.2))
    second = net._apply_multipliers(_sp(0.7))
    assert torch.allclose(first.K_sat_1, torch.full((5,), 0.2))
    assert torch.allclose(second.K_sat_1, torch.full((5,), 0.2))
    assert torch.allclose(second.K_sat_2, torch.full((5,), 0.7))


def test_multiplier_applied_after_freezing():
    net = SpatialFieldNetwork.__new__(SpatialFieldNetwork)
    torch.nn.Module.__init__(net)
    net.freeze_on_next_call = {"K_sat_1"}
    net.multipliers = {"K_sat_1": torch.tensor(0.3)}
    net._apply_multipliers(_sp(0.2))
    step_output = net._apply_multipliers(_sp(0.9))
    assert torch.allclose(step_output.K_sat_1, torch.full((5,), 0.06))
