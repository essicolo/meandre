"""La porte de gel d'un processus déclaré (`frost_factor`) réduit le flux avec la profondeur de gel."""
import torch

from meandre.vertical.soil_processes import SoilProcess


def _ctx(frost):
    n = 4
    return {"theta": torch.full((n,), 0.40), "theta_fc": torch.full((n,), 0.30), "thickness": torch.full((n,), 2.0),
            "conductivity": torch.full((n,), 1e-3), "sin_slope": torch.full((n,), 0.05), "porosity": torch.full((n,), 0.45),
            "theta_wp": torch.full((n,), 0.12), "frost_frac": frost}


def test_porte_de_gel_proportionnelle():
    sans = SoilProcess(layer=3, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 72.0})
    avec = SoilProcess(layer=3, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 72.0, "frost_factor": 0.3})
    frost = torch.tensor([0.0, 0.5, 1.0, 1.0])
    q0, q1 = sans.flux(_ctx(frost)), avec.flux(_ctx(frost))
    assert torch.allclose(q1[0], q0[0])
    assert torch.allclose(q1[1], q0[1] * 0.65)
    assert torch.allclose(q1[2], q0[2] * 0.3)


def test_sans_cle_rien_ne_change():
    p = SoilProcess(layer=3, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 72.0})
    c = _ctx(torch.ones(4))
    assert torch.allclose(p.flux(c), p.flux({**c, "frost_frac": None}))
