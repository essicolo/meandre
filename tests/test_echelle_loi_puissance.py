"""L'échelle de la vidange en puissance de l'excès se donne en millimètres.

Sans échelle, l'excès est rapporté à la capacité gravitaire de toute la couche : sur 2,65 m
cela fait 185 mm, et quelques millimètres d'excès donnent une fraction infime qui éteint la
sortie. Avec `scale_mm`, la loi est linéaire à l'échelle et amplifie au-delà.
"""
import torch

from meandre.vertical import soil_processes as sp


def _ctx(exces_mm):
    return {"theta": torch.tensor([0.30 + exces_mm / 1000.0 / 2.65]), "theta_fc": torch.tensor([0.30]), "porosity": torch.tensor([0.37]), "thickness": torch.tensor([2.65])}


def test_sans_echelle_la_sortie_s_eteint():
    q1 = sp.base_thresh_power(_ctx(5.0), {"tau": 72.0})
    q2 = sp.base_thresh_power(_ctx(5.0), {"tau": 72.0, "exponent": 2.0})
    assert q2 < 0.05 * q1


def test_a_l_echelle_la_loi_est_lineaire_puis_amplifie():
    lin = sp.base_thresh_power(_ctx(10.0), {"tau": 72.0})
    a_l_echelle = sp.base_thresh_power(_ctx(10.0), {"tau": 72.0, "exponent": 2.0, "scale": 0.010})
    assert torch.allclose(lin, a_l_echelle)
    double = sp.base_thresh_power(_ctx(20.0), {"tau": 72.0, "exponent": 2.0, "scale": 0.010})
    assert torch.allclose(double, 4.0 * a_l_echelle)


def test_from_toml_convertit_l_echelle():
    prof = sp.from_toml({"layers": 3, "process": [{"layer": 3, "kind": "lateral", "form": "BASE_THRESH_POWER", "tau_days": 3.0, "exponent": 2.0, "scale_mm": 10.0}]})
    p = prof.processes[0].params
    assert abs(p["scale"] - 0.010) < 1e-12 and "scale_mm" not in p and p["tau"] == 72.0
