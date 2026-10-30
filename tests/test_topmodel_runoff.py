"""Ruissellement par aire saturée commandée par la nappe libre (forme TOPMODEL), processus
déclaré « runoff ». Le levier mesuré sans simulation (R447) : le stock de nappe du modèle gelé
prédit le coefficient d'épisode observé à pente log-log 0,63 en Montérégie et 0,33 sur l'Outaouais.
La forme refuse de tourner sans nappe : sans levier, elle n'a pas de sens.
"""
import pytest
import torch

from meandre.hydrotel.bv3c2 import BV3C2Clone, make_params
from meandre.vertical import soil_processes as sp


def _runoff(s_gw_mm, input_m, exponent=0.6, scale=0.1):
    ctx = {"input": torch.tensor([input_m]), "s_gw": torch.tensor([s_gw_mm / 1000.0])}
    return float(sp.topmodel_runoff(ctx, {"exponent": exponent, "scale": scale}) / input_m)


def test_coefficient_grows_with_water_table_stock():
    low, high = _runoff(10.0, 0.020), _runoff(50.0, 0.020)
    assert 0.0 <= low < high <= 1.0


def test_log_log_slope_equals_exponent():
    """Sur deux stocks à une octave d'écart, la pente log-log du coefficient vaut l'exposant,
    tant que la portion reste loin de sa borne."""
    a1, a2 = _runoff(12.5, 0.020), _runoff(25.0, 0.020)
    slope = (torch.tensor(a2).log() - torch.tensor(a1).log()) / torch.tensor(2.0).log()
    assert abs(float(slope) - 0.6) < 0.02


def test_saturated_below_scale_and_sheds_everything_at_scale():
    a_low, a_high = _runoff(50.0, 0.030, scale=0.1), _runoff(100.0, 0.030, scale=0.1)
    assert a_high == 1.0
    assert abs(a_low - 0.5 ** 0.6) < 1e-4


def test_declared_form_refuses_without_water_table_context():
    ctx = {"input": torch.tensor([0.020])}
    with pytest.raises(ValueError):
        sp.topmodel_runoff(ctx, {"exponent": 0.6, "scale": 0.1})


def test_declare_refuses_topmodel_without_free_water_table():
    from meandre.vertical.hydrotel_column import HydrotelColumn
    profile = sp.SoilProfile(layers=3, processes=(sp.SoilProcess(layer=1, kind="runoff", form="TOPMODEL", params={"exponent": 0.6, "scale": 0.1}),))
    col = HydrotelColumn()
    with pytest.raises(ValueError):
        col.declare_soil_profile(profile)
    col.enable_free_water_table()
    names = col.declare_soil_profile(profile)
    assert names == []


def test_declared_topmodel_conserves_water_in_column():
    p = make_params()
    n = 4
    theta = [torch.full((n,), v) for v in (0.30, 0.32, 0.34)]
    apport = torch.full((n,), 25.0)
    etp = torch.zeros(n)
    zero = torch.zeros(n)
    base = sp.hydrotel_profile(krec=p["krec"])
    with_runoff = sp.SoilProfile(layers=3, processes=base.processes + (sp.SoilProcess(layer=1, kind="runoff", form="TOPMODEL", params={"exponent": 0.6, "scale": 0.2}),))
    p2 = {**p, "soil_profile": base, "thetapf1": torch.full((n,), 0.10), "thetapf2": torch.full((n,), 0.10), "thetapf3": torch.full((n,), 0.10)}
    p3 = {**p2, "soil_profile": with_runoff, "s_gw": torch.full((n,), 0.02)}
    col = BV3C2Clone(n_substep=32)
    a = col.forward(*theta, apport, etp, zero, zero, p2)
    b = col.forward(*theta, apport, etp, zero, zero, p3)
    assert torch.all(b[0] >= a[0]), "le ruissellement déclaré doit accroître la production de surface"
    def total(out):
        stock = sum((out[4][i] - theta[i]) * p[f"z{i + 1}"] * 1000.0 for i in range(3)) * p["fsa"]
        return out[0] + out[1] + out[2] + out[3] + stock
    assert torch.allclose(total(a), total(b), atol=1e-2), "le bilan d'eau doit fermer avec et sans ruissellement déclaré"