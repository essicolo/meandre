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

NOMS = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]


def _champ():
    return SpatialFieldNetwork(n_territorial=4, n_coord_freqs=2, hidden=16)


def _sortie(net, nom, brut, grad=False):
    raw = torch.zeros(len(brut), SpatialParams.N_PARAMS)
    raw[:, NOMS.index(nom)] = torch.tensor(brut)
    raw.requires_grad_(grad)
    return raw, getattr(net._apply_constraints(raw), nom)


@pytest.fixture(autouse=True)
def _reglages_non_configures():
    """Les reglages sont lus dans l'environnement tant que rien n'a ete configure : l'etat est
    vide pour le test, puis restaure, pour ne rien figer pour les tests suivants."""
    garde = dict(settings._configured)
    settings._configured.clear()
    yield
    settings._configured.clear()
    settings._configured.update(garde)


def _avec(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_plancher_garde_la_valeur_et_rend_un_gradient(monkeypatch):
    net = _champ()
    _, v0 = _sortie(net, "Z3", [-15.0, 0.0])
    _avec(monkeypatch, MEANDRE_PLANCHER_GRADIENT="0.01", MEANDRE_PLANCHER_GRADIENT_SORTIES="Z3")
    raw, v = _sortie(net, "Z3", [-15.0, 0.0], grad=True)
    assert torch.allclose(v.detach(), v0)
    v.sum().backward()
    assert float(raw.grad[0, NOMS.index("Z3")]) > 0.01


def test_prolongement_conserve_les_valeurs_de_depart(monkeypatch):
    net = _champ()
    depart = [-9.0, -4.0, 0.0, 3.0]
    _, v0 = _sortie(net, "Z3", depart)
    _avec(monkeypatch, MEANDRE_PROLONGE_SOUS="Z3:0.20")
    net._extension_thresholds = {}
    _, v = _sortie(net, "Z3", depart)
    assert torch.allclose(v, v0)
    # Sous le seuil (plus basse sortie de depart moins 0,5), la valeur descend vers 0,20 m.
    _, bas = _sortie(net, "Z3", [-12.0, -40.0])
    assert float(bas[0]) < float(v0[0]) and float(bas[1]) > 0.20 - 1e-6 and float(bas[1]) < 0.21


def test_seuil_donne_explicitement(monkeypatch):
    net = _champ()
    _avec(monkeypatch, MEANDRE_PROLONGE_SOUS="Z2:0.10:-5.0")
    net._extension_thresholds = {}
    _, v = _sortie(net, "Z2", [-4.0, -60.0])
    assert float(v[0]) > 0.30 and abs(float(v[1]) - 0.10) < 1e-4


def _fraction_fc3(net, brut, grad=False):
    raw = torch.zeros(len(brut), SpatialParams.N_PARAMS)
    raw[:, NOMS.index("theta_fc_3")] = torch.tensor(brut)
    raw.requires_grad_(grad)
    sp = net._apply_constraints(raw)
    return raw, sp.theta_fc_3 / sp.porosity_3, sp


def test_prolongement_au_dessus_d_une_seule_couche(monkeypatch):
    """Une fraction de capacite au champ collee a 0,85 peut monter au-dessus, sans que les
    valeurs de depart changent ; la couche 1, non nommee, reste bornee a 0,85 (2026-10-27)."""
    net = _champ()
    depart = [0.0, 4.0, 9.0]
    _, f0, _ = _fraction_fc3(net, depart)
    _avec(monkeypatch, MEANDRE_PROLONGE_SUR="theta_fc_frac_3:0.98")
    net._extension_thresholds = {}
    _, f, _ = _fraction_fc3(net, depart)
    assert torch.allclose(f, f0, atol=1e-6)
    _, haut, _ = _fraction_fc3(net, [12.0, 60.0])
    assert float(haut[0]) > 0.85 and abs(float(haut[1]) - 0.98) < 1e-4
    raw = torch.zeros(1, SpatialParams.N_PARAMS)
    raw[:, NOMS.index("theta_fc_1")] = 60.0
    sp = net._apply_constraints(raw)
    assert float(sp.theta_fc_1 / sp.porosity_1) <= 0.85 + 1e-6


def test_plancher_sur_une_seule_couche(monkeypatch):
    _avec(monkeypatch, MEANDRE_PLANCHER_GRADIENT="0.01", MEANDRE_PLANCHER_GRADIENT_SORTIES="porosity_3")
    net = _champ()
    raw = torch.zeros(1, SpatialParams.N_PARAMS)
    raw[:, NOMS.index("porosity_1")] = 15.0
    raw[:, NOMS.index("porosity_3")] = 15.0
    raw.requires_grad_(True)
    sp = net._apply_constraints(raw)
    (sp.porosity_1 + sp.porosity_3).sum().backward()
    assert float(raw.grad[0, NOMS.index("porosity_3")]) > 0.003
    assert float(raw.grad[0, NOMS.index("porosity_1")]) < 1e-5
