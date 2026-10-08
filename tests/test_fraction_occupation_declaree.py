"""Un processus déclaré peut se restreindre à une fraction d'occupation (`cover`, `share`)."""
import pytest
import torch

from meandre.vertical.soil_processes import SoilProcess


def _ctx(agri):
    n = 3
    return {"theta": torch.full((n,), 0.40), "theta_fc": torch.full((n,), 0.30), "thickness": torch.full((n,), 1.0),
            "conductivity": torch.full((n,), 1e-3), "sin_slope": torch.full((n,), 0.05), "porosity": torch.full((n,), 0.45),
            "theta_wp": torch.full((n,), 0.12), "agri_frac": agri}


def test_flux_proportionnel_a_la_fraction_agricole():
    sans = SoilProcess(layer=2, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 24.0})
    avec = SoilProcess(layer=2, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 24.0, "cover": "agri", "share": 0.5})
    agri = torch.tensor([0.0, 0.4, 1.0])
    q0, q1 = sans.flux(_ctx(agri)), avec.flux(_ctx(agri))
    assert torch.allclose(q1, q0 * agri * 0.5)


def test_occupation_inconnue_refusee():
    p = SoilProcess(layer=2, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 24.0, "cover": "urbain"})
    with pytest.raises(KeyError):
        p.flux(_ctx(torch.ones(3)))
