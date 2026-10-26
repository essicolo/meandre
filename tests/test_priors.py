"""Priors par tronçon (étape 3, 2026-10-26) : centres tirés de la texture et des dépôts, champ en
écart pénalisé, valeurs conservées à l'adoption.
"""
import numpy as np
import torch

from meandre import priors
from meandre.spatial.field_network import SpatialFieldNetwork


def _champ(n_terr=4):
    torch.manual_seed(0)
    net = SpatialFieldNetwork(n_territorial=n_terr, n_coord_freqs=2, hidden=16)
    net.init_from_literature()
    return net


def _entrees(n=12, n_terr=4):
    g = torch.Generator().manual_seed(1)
    coords = torch.stack([-72.0 + torch.rand(n, generator=g), 46.0 + torch.rand(n, generator=g)], dim=1)
    return coords, torch.randn(n, n_terr, generator=g)


def test_les_centres_sont_physiques_et_le_repli_est_un_loam():
    n = 5
    sable = np.array([0.8, 0.1, np.nan, 0.4, 0.3])
    argile = np.array([0.05, 0.5, np.nan, 0.2, 0.3])
    p = priors.soil_priors(n, {1: (sable, argile)}, np.array([0.5, 10.0, np.nan, 2.0, 1.0]))
    por, _ = p["porosity_1"]
    assert por[0] > por[1] - 0.2 and 0.3 < por.min() and por.max() < 0.7
    assert p["K_sat_1"][0][0] > p["K_sat_1"][0][1], "le sable conduit mieux que l'argile"
    loam = priors.saxton_rawls(*priors.LOAM)
    assert abs(por[2] - float(loam["theta_s"])) < 1e-9, "sans texture, le centre est le loam"
    z3, _ = p["Z3"]
    assert z3[0] == 0.2 and z3[1] == 4.0 and z3[2] == priors.Z3_DEFAULT_M and abs(z3[3] - (2.0 - 0.1 - 0.3)) < 1e-9
    assert set(p) == set(priors.KEYS)


def test_le_brut_nul_rend_le_centre_et_la_penalite_est_nulle():
    net = _champ()
    coords, terr = _entrees()
    n = len(coords)
    centre = np.linspace(0.35, 0.55, n)
    net.set_prior("porosity_1", centre, 0.1)
    net.set_prior("Z3", np.full(n, 1.7), 0.5)
    raw = torch.zeros(n, net.fc_out.out_features)
    sp = net._apply_constraints(raw)
    assert np.allclose(sp.porosity_1.numpy(), centre, atol=1e-6)
    assert np.allclose(sp.Z3.numpy(), 1.7, atol=1e-6)
    assert float(net.deviation_prior_loss()) == 0.0
    # la capacite au champ reste une fraction de la porosite centree
    assert bool((sp.theta_fc_1 < sp.porosity_1).all())


def test_l_adoption_conserve_les_valeurs_et_compte_la_deviation():
    net = _champ()
    coords, terr = _entrees()
    avant = net(coords, terr)
    n = len(coords)
    pri = priors.soil_priors(n, None, None)
    rapport = net.adopt_priors(coords, terr, pri, preserve=True)
    apres = net(coords, terr)
    for nom in ("porosity_1", "porosity_3", "theta_fc_2", "theta_wp_1", "K_sat_1", "K_sat_3", "Z2", "Z3"):
        a, b = getattr(avant, nom), getattr(apres, nom)
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7), nom
    assert set(rapport) == set(priors.KEYS)
    assert float(net.deviation_prior_loss()) > 0.0, "le modele de depart n'est pas a ses centres"
    # un tampon par cle, persistant : il voyage avec le point de reprise
    sd = net.state_dict()
    assert "_pr_o_Z3" in sd and "_pr_c_porosity_2" in sd and "_pr_s_K_sat_1" in sd


def test_une_cle_inconnue_ou_un_centre_nul_sont_refuses():
    import pytest
    net = _champ()
    with pytest.raises(KeyError):
        net.set_prior("C_f", np.ones(3), 0.1)
    with pytest.raises(ValueError):
        net.set_prior("Z2", np.array([0.3, 0.0, 0.3]), 0.1)
