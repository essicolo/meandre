"""Priors par tronçon (étape 3, 2026-10-26) : centres tirés de la texture et des dépôts, champ en
écart pénalisé, valeurs conservées à l'adoption.
"""
import numpy as np
import torch

from meandre import priors
from meandre.spatial.field_network import SpatialFieldNetwork


def _field(n_terr=4):
    torch.manual_seed(0)
    net = SpatialFieldNetwork(n_territorial=n_terr, n_coord_freqs=2, hidden=16)
    net.init_from_literature()
    return net


def _entrees(n=12, n_terr=4):
    g = torch.Generator().manual_seed(1)
    coords = torch.stack([-72.0 + torch.rand(n, generator=g), 46.0 + torch.rand(n, generator=g)], dim=1)
    return coords, torch.randn(n, n_terr, generator=g)


def test_centers_are_physical_and_fallback_is_loam():
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


def test_zero_raw_returns_center_and_zero_penalty():
    net = _field()
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


def test_adoption_keeps_values_and_counts_deviation():
    net = _field()
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


def test_unknown_key_or_zero_center_refused():
    import pytest
    net = _field()
    with pytest.raises(KeyError):
        net.set_prior("C_f", np.ones(3), 0.1)
    with pytest.raises(ValueError):
        net.set_prior("Z2", np.array([0.3, 0.0, 0.3]), 0.1)


def test_relative_modes_leave_level_free():
    net = _field()
    coords, terr = _entrees()
    n = len(coords)
    centre = np.linspace(0.05, 0.5, n)
    net.set_prior("K_sat_1", centre, 0.7)
    raw = torch.zeros(n, net.fc_out.out_features)
    from meandre.spatial.field_network import SpatialParams
    noms = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]
    raw[:, noms.index("K_sat_1")] = 1.5   # tout le champ deux fois plus conducteur que la carte, motif identique
    net._apply_constraints(raw)
    assert float(net.deviation_prior_loss("level")) > 1.0
    assert float(net.deviation_prior_loss("anomaly")) < 1e-10
    assert float(net.deviation_prior_loss("shape")) < 1e-8
    raw[:, noms.index("K_sat_1")] = torch.linspace(-1.0, 1.0, n)   # motif deforme
    net._apply_constraints(raw)
    assert float(net.deviation_prior_loss("anomaly")) > 0.1


def test_output_stuck_at_bound_is_freed_under_prior(monkeypatch):
    """Porosite 3 collee a 0,60 : sous prior restreint a cette sortie, valeur conservee et gradient
    plein ; les autres sorties gardent leur sigmoide (2026-10-27)."""
    from meandre.utils import settings
    garde = dict(settings._configured)
    settings._configured.clear()
    monkeypatch.setenv("MEANDRE_PRIOR_CLES", "porosity_3")
    try:
        net = _field()
        coords, terr = _entrees()
        with torch.no_grad():
            net.fc_out.bias[[f for f in __import__("meandre.spatial.field_network", fromlist=["SpatialParams"]).SpatialParams.__dataclass_fields__].index("porosity_3")] += 20.0
        avant = net(coords, terr)
        assert torch.allclose(avant.porosity_3, torch.full_like(avant.porosity_3, 0.60), atol=1e-4)
        pri = priors.soil_priors(len(coords), None, None)
        assert set(pri) == {"porosity_3"}
        net.adopt_priors(coords, terr, pri, preserve=True)
        apres = net(coords, terr)
        assert torch.allclose(avant.porosity_3, apres.porosity_3, atol=1e-5)
        apres.porosity_3.sum().backward()
        assert float(net.fc_out.bias.grad.abs().max()) > 0.01
    finally:
        settings._configured.clear()
        settings._configured.update(garde)


def test_start_at_center_and_level_mode_per_key(monkeypatch):
    """`start_at_center` pose la valeur du prior exactement ; `priors.level_keys` juge cette clé
    en niveau même quand le mode global est anomalie (2026-10-27)."""
    from meandre.utils import settings
    garde = dict(settings._configured)
    settings._configured.clear()
    try:
        net = _field()
        coords, terr = _entrees()
        n = len(coords)
        centre = np.linspace(0.001, 0.002, n)
        pri = priors.soil_priors(n, None, None, k_sat_3_center_m_per_day=centre)
        assert np.allclose(pri["K_sat_3"][0], centre)
        pri = {"K_sat_3": pri["K_sat_3"], "K_sat_1": pri["K_sat_1"]}
        avant = net(coords, terr)
        net.adopt_priors(coords, terr, pri, preserve=True, start_at_center=("K_sat_3",))
        apres = net(coords, terr)
        assert np.allclose(apres.K_sat_3.detach().numpy(), centre, rtol=1e-4), "K_sat_3 part au centre"
        assert torch.allclose(avant.K_sat_1, apres.K_sat_1, rtol=1e-5), "K_sat_1 conserve"
        # tout le champ K_sat_3 deux fois le centre : rien en anomalie, une penalite en niveau
        raw = torch.zeros(n, net.fc_out.out_features)
        from meandre.spatial.field_network import SpatialParams
        noms = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]
        net._pr_o_K_sat_3.zero_()
        net._pr_o_K_sat_1.zero_()
        raw[:, noms.index("K_sat_3")] = float(np.log(2.0) / pri["K_sat_3"][1])
        net._apply_constraints(raw)
        assert float(net.deviation_prior_loss("anomaly")) < 1e-10
        monkeypatch.setenv("MEANDRE_PRIOR_CLES_NIVEAU", "K_sat_3")
        assert float(net.deviation_prior_loss("anomaly")) > 0.05
    finally:
        settings._configured.clear()
        settings._configured.update(garde)
