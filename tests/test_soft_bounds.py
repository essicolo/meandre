"""Bornes douces et configurables des sorties du champ spatial (2026-10-01).

Les huit sorties en exp(clamp()) collaient au plancher sans gradient (K_sat_2 a exp(-8)).
Toutes passent par une sigmoide, lineaire ou logarithmique, et leurs bornes viennent d'une
table que le TOML peut remplacer.
"""
import math

import pytest
import torch

from meandre.spatial.field_network import FIELD_BOUNDS, SpatialFieldNetwork, SpatialParams, field_bounds_from_toml


def _field(**kw):
    return SpatialFieldNetwork(n_territorial=4, n_coord_freqs=2, hidden=16, **kw)


def _output(net, nom, brut):
    raw = torch.zeros(len(brut), SpatialParams.N_PARAMS)
    noms = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]
    raw[:, noms.index(nom)] = torch.tensor(brut)
    return getattr(net._apply_constraints(raw), nom)


def test_center_kept_for_log_outputs():
    net = _field()
    assert math.isclose(float(_output(net, "K_sat_2", [0.0])[0]), 0.1, rel_tol=1e-4)
    assert math.isclose(float(_output(net, "K_sat_1", [0.0])[0]), 0.5, rel_tol=1e-4)


def test_nonzero_gradient_near_floor():
    net = _field()
    raw = torch.zeros(1, SpatialParams.N_PARAMS)
    noms = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]
    raw[0, noms.index("K_sat_2")] = -40.0
    raw.requires_grad_(True)
    v = net._apply_constraints(raw).K_sat_2.sum()
    v.backward()
    lo = FIELD_BOUNDS["K_sat_2"][0]
    assert float(v) >= lo
    assert float(raw.grad[0, noms.index("K_sat_2")]) > 0.0


def test_bounds_from_toml():
    cfg = {"field": {"bounds": {"K_sat_2": [1e-3, 10.0], "C_f": [1.0, 6.0]}}, "soil": {"z2_min": 0.4}}
    fb = field_bounds_from_toml(cfg)
    net = _field(field_bounds=fb)
    assert float(_output(net, "K_sat_2", [-50.0])[0]) >= 1e-3 * 0.999
    assert float(_output(net, "C_f", [50.0])[0]) <= 6.0 + 1e-6
    assert float(_output(net, "Z2", [-50.0])[0]) >= 0.4 - 1e-6


def test_unknown_output_refused():
    with pytest.raises(KeyError):
        field_bounds_from_toml({"field": {"bounds": {"pas_une_sortie": [0, 1]}}})


def test_initialization_inverts_the_law():
    net = _field()
    raw = net._literature_raw_vector().unsqueeze(0)
    sp = net._apply_constraints(raw)
    assert math.isclose(float(sp.K_sat_1[0]), net._prior_targets["K_sat_1"], rel_tol=1e-3)
    assert math.isclose(float(sp.k_sub[0]), net._prior_targets["k_sub"], rel_tol=1e-3)
