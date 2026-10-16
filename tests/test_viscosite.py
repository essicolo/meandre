"""Correction par la viscosité (R384) : les vidanges à constante de temps suivent le facteur, BASE_LINEAR non."""
import torch

from meandre.vertical.soil_processes import SoilProcess


def _ctx(visc=None):
    c = {"theta": torch.tensor([0.45]), "theta_fc": torch.tensor([0.30]), "thickness": torch.tensor([1.0]),
         "conductivity": torch.tensor([0.01]), "sin_slope": torch.tensor([0.05]), "porosity": torch.tensor([0.50])}
    if visc is not None:
        c["visc"] = torch.tensor([visc])
    return c


def test_vidange_a_constante_de_temps_suit_la_viscosite():
    p = SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER", params={"tau": 48.0})
    sans = p.flux(_ctx())
    avec = p.flux(_ctx(0.6))
    assert torch.allclose(avec, 0.6 * sans)


def test_base_linear_ne_double_pas_la_correction():
    p = SoilProcess(layer=2, kind="lateral", form="BASE_LINEAR")
    assert torch.allclose(p.flux(_ctx(0.6)), p.flux(_ctx()))
