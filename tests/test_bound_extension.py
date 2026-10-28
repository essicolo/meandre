"""Plancher de gradient et prolongement sous la borne des sorties du champ (2026-10-26).

Une sortie collee a sa borne n'a plus de gradient, et abaisser la borne remappe tous les
troncons. Le plancher rend un gradient sans changer la valeur ; le prolongement descend sous
la borne seulement en dessous d'un seuil de sortie brute, si bien que les valeurs de depart
sont conservees.
"""
import pytest
import torch

from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams
from meandre.utils import settings

NAMES = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]


def _field():
    return SpatialFieldNetwork(n_territorial=4, n_coord_freqs=2, hidden=16)


def _output(net, field_name, raw_value, grad=False):
    raw = torch.zeros(len(raw_value), SpatialParams.N_PARAMS)
    raw[:, NAMES.index(field_name)] = torch.tensor(raw_value)
    raw.requires_grad_(grad)
    return raw, getattr(net._apply_constraints(raw), field_name)


@pytest.fixture(autouse=True)
def _unconfigured_settings():
    """Les reglages sont lus dans l'environnement tant que rien n'a ete configure : l'etat est
    vide pour le test, puis restaure, pour ne rien figer pour les tests suivants."""
    kept = dict(settings._configured)
    settings._configured.clear()
    yield
    settings._configured.clear()
    settings._configured.update(kept)


def _with(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_floor_keeps_value_and_returns_gradient(monkeypatch):
    net = _field()
    _, v0 = _output(net, "Z3", [-15.0, 0.0])
    _with(monkeypatch, MEANDRE_PLANCHER_GRADIENT="0.01", MEANDRE_PLANCHER_GRADIENT_SORTIES="Z3")
    raw, v = _output(net, "Z3", [-15.0, 0.0], grad=True)
    assert torch.allclose(v.detach(), v0)
    v.sum().backward()
    assert float(raw.grad[0, NAMES.index("Z3")]) > 0.01


def test_extension_keeps_starting_values(monkeypatch):
    net = _field()
    start = [-9.0, -4.0, 0.0, 3.0]
    _, v0 = _output(net, "Z3", start)
    _with(monkeypatch, MEANDRE_PROLONGE_SOUS="Z3:0.20")
    net._extension_thresholds = {}
    _, v = _output(net, "Z3", start)
    assert torch.allclose(v, v0)
    # Sous le seuil (plus basse sortie de depart moins 0,5), la valeur descend vers 0,20 m.
    _, bas = _output(net, "Z3", [-12.0, -40.0])
    assert float(bas[0]) < float(v0[0]) and float(bas[1]) > 0.20 - 1e-6 and float(bas[1]) < 0.21


def test_threshold_given_explicitly(monkeypatch):
    net = _field()
    _with(monkeypatch, MEANDRE_PROLONGE_SOUS="Z2:0.10:-5.0")
    net._extension_thresholds = {}
    _, v = _output(net, "Z2", [-4.0, -60.0])
    assert float(v[0]) > 0.30 and abs(float(v[1]) - 0.10) < 1e-4


def _fraction_fc3(net, raw_value, grad=False):
    raw = torch.zeros(len(raw_value), SpatialParams.N_PARAMS)
    raw[:, NAMES.index("theta_fc_3")] = torch.tensor(raw_value)
    raw.requires_grad_(grad)
    sp = net._apply_constraints(raw)
    return raw, sp.theta_fc_3 / sp.porosity_3, sp


def test_extension_above_a_single_layer(monkeypatch):
    """Une fraction de capacite au champ collee a 0,85 peut monter au-dessus, sans que les
    valeurs de depart changent ; la couche 1, non nommee, reste bornee a 0,85 (2026-10-27)."""
    net = _field()
    start = [0.0, 4.0, 9.0]
    _, f0, _ = _fraction_fc3(net, start)
    _with(monkeypatch, MEANDRE_PROLONGE_SUR="theta_fc_frac_3:0.98")
    net._extension_thresholds = {}
    _, f, _ = _fraction_fc3(net, start)
    assert torch.allclose(f, f0, atol=1e-6)
    _, high, _ = _fraction_fc3(net, [12.0, 60.0])
    assert float(high[0]) > 0.85 and abs(float(high[1]) - 0.98) < 1e-4
    raw = torch.zeros(1, SpatialParams.N_PARAMS)
    raw[:, NAMES.index("theta_fc_1")] = 60.0
    sp = net._apply_constraints(raw)
    assert float(sp.theta_fc_1 / sp.porosity_1) <= 0.85 + 1e-6


def test_floor_on_a_single_layer(monkeypatch):
    _with(monkeypatch, MEANDRE_PLANCHER_GRADIENT="0.01", MEANDRE_PLANCHER_GRADIENT_SORTIES="porosity_3")
    net = _field()
    raw = torch.zeros(1, SpatialParams.N_PARAMS)
    raw[:, NAMES.index("porosity_1")] = 15.0
    raw[:, NAMES.index("porosity_3")] = 15.0
    raw.requires_grad_(True)
    sp = net._apply_constraints(raw)
    (sp.porosity_1 + sp.porosity_3).sum().backward()
    assert float(raw.grad[0, NAMES.index("porosity_3")]) > 0.003
    assert float(raw.grad[0, NAMES.index("porosity_1")]) < 1e-5
