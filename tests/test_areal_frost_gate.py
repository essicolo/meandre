"""Porte de gel surfacique : la part gelée de la couche de surface est la part imperméable.

Sans neige protectrice, un gel qui occupe la moitié de la couche de surface doit faire
ruisseler environ la moitié d'une pluie inférieure à la conductivité saturée. La règle tout
ou rien du clone fait tout ruisseler, la porte continue presque rien.
"""
import torch

import meandre.hydrotel.bv3c2 as bv

N = 4


def _runoff(gate, frost_cm, neige_mm=0.0):
    bv._SEMI_IMPLICITE = False
    p = bv.make_params("loam", "loam", "loam")
    for k in list(p):
        if torch.is_tensor(p[k]) and p[k].ndim == 0:
            p[k] = p[k].expand(N).clone()
    m = bv.BV3C2Clone(n_substep=32, frozen_gate_continuous=(gate == "continue"))
    m.frozen_gate_areal = gate == "surfacique"
    t = [0.5 * p[f"thetas{i}"] for i in (1, 2, 3)]
    frost_state = torch.full((N,), float(frost_cm) if frost_cm != "moitie" else 50.0 * float(p["z1"][0]))
    out = m(t[0], t[1], t[2], torch.full((N,), 5.0), torch.zeros(N), frost_state, torch.full((N,), neige_mm), p)
    return float(out[0][0])


def test_half_frozen_runs_off_half():
    r = _runoff("surfacique", "moitie")
    base = _runoff("surfacique", 0.0)
    assert 2.0 < r - base < 3.0


def test_bracketed_by_the_two_other_rules():
    r_b = _runoff("binaire", "moitie")
    r_c = _runoff("continue", "moitie")
    r_s = _runoff("surfacique", "moitie")
    assert r_c < r_s < r_b


def test_soil_stays_permeable_under_snow():
    without_frost = _runoff("surfacique", 0.0, neige_mm=50.0)
    assert abs(_runoff("surfacique", "moitie", neige_mm=50.0) - without_frost) < 1e-9


def test_without_frost_identical_to_original_rule():
    assert abs(_runoff("surfacique", 0.0) - _runoff("binaire", 0.0)) < 1e-9
