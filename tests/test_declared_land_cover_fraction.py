"""Un processus déclaré peut se restreindre à une fraction d'occupation (`cover`, `share`)."""
import pytest
import torch

from meandre.vertical.soil_processes import SoilProcess


def _ctx(agri):
    n = 3
    return {"theta": torch.full((n,), 0.40), "theta_fc": torch.full((n,), 0.30), "thickness": torch.full((n,), 1.0),
            "conductivity": torch.full((n,), 1e-3), "sin_slope": torch.full((n,), 0.05), "porosity": torch.full((n,), 0.45),
            "theta_wp": torch.full((n,), 0.12), "agri_frac": agri}


def test_flux_proportional_to_agricultural_fraction():
    without = SoilProcess(layer=2, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 24.0})
    avec = SoilProcess(layer=2, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 24.0, "cover": "agri", "share": 0.5})
    agri = torch.tensor([0.0, 0.4, 1.0])
    q0, q1 = without.flux(_ctx(agri)), avec.flux(_ctx(agri))
    assert torch.allclose(q1, q0 * agri * 0.5)


def test_unknown_land_cover_refused():
    p = SoilProcess(layer=2, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 24.0, "cover": "urbain"})
    with pytest.raises(KeyError):
        p.flux(_ctx(torch.ones(3)))
