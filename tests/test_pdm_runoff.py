"""Ruissellement par distribution de capacités de stockage (INF_PDM), processus déclaré « runoff ».

Deux propriétés mesurées sur les débits observés (R447) doivent sortir de la forme : le coefficient
de ruissellement croît avec la taille de l'averse et avec le stock antérieur. Et la déclaration ne
touche à rien d'autre : sans processus « runoff », le clone est inchangé.
"""
import torch

from meandre.hydrotel.bv3c2 import BV3C2Clone, make_params
from meandre.vertical import soil_processes as sp


def _coefficient(storage_fraction, input_m, exponent=0.5, factor=1.0):
    cap = torch.tensor([0.20])
    ctx = {"storage": cap * storage_fraction, "capacity": cap, "input": torch.tensor([input_m])}
    return float(sp.inf_pdm(ctx, {"exponent": exponent, "factor": factor}) / input_m)


def test_coefficient_grows_with_storm_size():
    small, large = _coefficient(0.5, 0.010), _coefficient(0.5, 0.040)
    assert 0.0 < small < large < 1.0


def test_coefficient_grows_with_antecedent_storage():
    dry, wet = _coefficient(0.2, 0.020), _coefficient(0.8, 0.020)
    assert dry < wet


def test_full_store_sheds_everything_and_empty_store_almost_nothing():
    assert abs(_coefficient(1.0, 0.020) - 1.0) < 1e-5
    assert _coefficient(0.0, 0.002) < 0.05


def test_quadratic_onset_for_small_storms():
    """Pour une petite averse sur un stock vide, le ruissellement croît comme le carré de l'apport,
    quel que soit l'exposant : pente log-log du coefficient proche de un (R447)."""
    for b in (0.3, 1.0, 3.0):
        c1, c2 = _coefficient(0.0, 0.004, b), _coefficient(0.0, 0.008, b)
        slope = (torch.tensor(c2).log() - torch.tensor(c1).log()) / torch.tensor(2.0).log()
        assert 0.85 < float(slope) < 1.05, f"b={b} : pente {float(slope):.2f}"


def test_factor_scales_capacity_and_is_a_free_parameter():
    assert _coefficient(0.5, 0.020, factor=1.0) > _coefficient(0.5, 0.020, factor=3.0)


def test_declared_runoff_changes_surface_only_and_conserves_water():
    p = make_params()
    n = 4
    theta = [torch.full((n,), v) for v in (0.30, 0.32, 0.34)]
    apport = torch.full((n,), 25.0)
    etp = torch.zeros(n)
    zero = torch.zeros(n)
    base = sp.hydrotel_profile(krec=p["krec"])
    with_runoff = sp.SoilProfile(layers=3, processes=base.processes + (sp.SoilProcess(layer=2, kind="runoff", form="INF_PDM", params={"exponent": 0.5}),))
    p2 = {**p, "soil_profile": base, "thetapf1": torch.full((n,), 0.10), "thetapf2": torch.full((n,), 0.10), "thetapf3": torch.full((n,), 0.10)}
    p3 = {**p2, "soil_profile": with_runoff}
    col = BV3C2Clone(n_substep=32)
    a = col.forward(*theta, apport, etp, zero, zero, p2)
    b = col.forward(*theta, apport, etp, zero, zero, p3)
    assert torch.all(b[0] > a[0]), "le ruissellement déclaré doit accroître la production de surface"
    def total(out):
        # production + recharge + variation de stock des trois couches, en mm ; fsa pondère le sol
        stock = sum((out[4][i] - theta[i]) * p[f"z{i + 1}"] * 1000.0 for i in range(3)) * p["fsa"]
        return out[0] + out[1] + out[2] + out[3] + stock
    assert torch.allclose(total(a), total(b), atol=1e-2), "le bilan d'eau doit fermer avec et sans ruissellement déclaré"


def test_form_requires_exponent_and_is_runoff_only():
    import pytest
    with pytest.raises(ValueError):
        sp.SoilProcess(layer=1, kind="runoff", form="INF_PDM", params={})
    with pytest.raises(ValueError):
        sp.SoilProcess(layer=1, kind="lateral", form="INF_PDM", params={"exponent": 0.5})
