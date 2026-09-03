"""Tests for the spatial field network (NeRF-style MLP)."""

import torch
import pytest
from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams


def test_output_types():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(10, 2)
    features = torch.randn(10, 17)
    params = net(coords, features)
    assert isinstance(params, SpatialParams)


def test_K_sat_positive():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(20, 2)
    features = torch.randn(20, 17)
    params = net(coords, features)
    assert (params.K_sat_1 > 0).all()
    assert (params.K_sat_2 > 0).all()
    assert (params.K_sat_3 > 0).all()


def test_root_fractions_sum_to_one():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(15, 2)
    features = torch.randn(15, 17)
    params = net(coords, features)
    f_sum = params.f_root_1 + params.f_root_2 + params.f_root_3
    assert torch.allclose(f_sum, torch.ones(15), atol=1e-5)


def test_parameter_ranges():
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    coords = torch.randn(30, 2)
    features = torch.randn(30, 17)
    params = net(coords, features)

    assert (params.C_f >= 0).all() and (params.C_f <= 10).all()
    assert (params.T_melt >= -2).all() and (params.T_melt <= 2).all()
    assert (params.manning_n >= 0.01).all() and (params.manning_n <= 0.2).all()


def test_gradients_flow():
    net = SpatialFieldNetwork(n_territorial=5, hidden=32)
    coords = torch.randn(8, 2, requires_grad=True)
    features = torch.randn(8, 5)
    params = net(coords, features)
    params.C_f.sum().backward()
    assert coords.grad is not None


def test_sortie_rembourree_vaut_le_milieu_des_bornes():
    """Une ligne NULLE de fc_out rend le milieu des bornes, pas la valeur de litterature.

    C'est le mecanisme qui a fige cinq parametres dans les modeles retenus (mesure du
    2026-09-03) : un point de reprise plus ancien que le champ voit ses sorties
    manquantes remplies de zeros au chargement, et le parametre devient une constante
    identique sur tous les troncons. Le test fixe le comportement pour qu'il reste
    visible : sortie brute nulle donne la meme valeur partout, au centre des bornes.
    """
    net = SpatialFieldNetwork(n_territorial=17, hidden=64)
    with torch.no_grad():
        net.fc_out.weight.zero_()
        net.fc_out.bias.zero_()
    coords = torch.randn(32, 2)
    features = torch.randn(32, 17)
    p = net(coords, features)

    # dT_canopee_feu est borne [0, 3] : une sortie brute nulle doit donner 1.5 partout.
    v = p.dT_canopee_feu
    assert float(v.max() - v.min()) < 1e-6, "un parametre rembourre doit etre CONSTANT"
    assert float(v.median()) == pytest.approx(1.5, abs=1e-4)


def test_bornes_de_muskingum_configurables(monkeypatch):
    """MEANDRE_KMUSK deplace les bornes du temps de transfert.

    Le troncon median mesure 2.9 a 3.5 km, parcouru en une heure a 1 m/s, alors que la
    borne inferieure par defaut vaut 4 h : elle excede le temps physique d'un facteur
    quatre a cinq. Le levier doit permettre de descendre dans la plage physique.
    """
    import importlib
    import meandre.spatial.field_network as fn

    monkeypatch.setenv("MEANDRE_KMUSK", "0.5,12,3")
    importlib.reload(fn)
    try:
        assert (fn._KMUSK_MIN, fn._KMUSK_MAX, fn._KMUSK_INIT) == (0.5, 12.0, 3.0)
        net = fn.SpatialFieldNetwork(n_territorial=17, hidden=64)
        p = net(torch.randn(64, 2), torch.randn(64, 17))
        k = p.K_musk_hours
        assert float(k.min()) >= 0.5 - 1e-6
        assert float(k.max()) <= 12.0 + 1e-6
    finally:
        monkeypatch.delenv("MEANDRE_KMUSK", raising=False)
        importlib.reload(fn)
