"""Lacs hors réseau : les clés nommées de la section [lakes] reproduisent l'ancienne déclaration par chaînes."""
import tomllib
from pathlib import Path

import pytest
import torch
from torch import nn

from meandre.model import HydroModel
from meandre.utils import settings

RECIPE = Path(__file__).resolve().parents[1] / ".runs" / "quebec" / "recipe.toml"


@pytest.fixture
def clean_settings():
    kept = dict(settings._configured)
    settings._configured.clear()
    yield
    settings._configured.clear()
    settings._configured.update(kept)


def _bare_model():
    model = HydroModel.__new__(HydroModel)
    nn.Module.__init__(model)
    return model


def _inputs():
    f = torch.tensor([0.0, 0.002, 0.03, 0.12, 0.4])
    precip = torch.tensor([2.1, 2.6, 3.0, 2.4, 2.8])
    return f, precip


def test_recipe_keys_match_old_string_form(clean_settings):
    settings.configure(tomllib.loads(RECIPE.read_text(encoding="utf-8")))
    assert settings.get("lakes.enabled") and settings.get("lakes.residence_from_volume")
    f, precip = _inputs()
    new = _bare_model()
    new.set_distributed_lakes_from_settings(f, precip)
    # Ancienne forme : distributed = "6.7,171.0,1.0,0.5", distributed_bounds = "c:0.5:50,T:20:2000",
    # distributed_learn = "c,T", residence_from_volume, R = 0,5 fois la précipitation moyenne.
    c, T, beta, h_ref = (float(x) for x in "6.7,171.0,1.0,0.5".split(","))
    bounds = {x.split(":")[0]: (float(x.split(":")[1]), float(x.split(":")[2])) for x in "c:0.5:50,T:20:2000".split(",")}
    old = _bare_model()
    old.set_distributed_lakes(f, c, T, beta, h_ref, bounds, learn=("c", "T"), runoff_mm_day=precip * 0.5)
    a, b = new.distributed_lakes_values(), old.distributed_lakes_values()
    assert set(a) == set(b) == {"c", "T", "D", "k"}
    for key in a:
        assert torch.equal(a[key], b[key]), key
    # Noms stockés des points de reprise inchangés.
    assert {n for n, _ in new.named_parameters()} == {"lakes_learn_c", "lakes_learn_T"}


def test_residence_days_and_unknown_learn(clean_settings):
    settings.configure({"lakes": {"enabled": True, "share_coefficient": 5.0, "residence_days": 60.0, "residence_bounds_days": [2.0, 2000.0], "learn": ["residence_days"]}})
    f, _ = _inputs()
    model = _bare_model()
    model.set_distributed_lakes_from_settings(f, None)
    values = model.distributed_lakes_values()
    assert "D" not in values
    assert torch.allclose(values["T"], torch.tensor(60.0), rtol=1e-5)
    assert torch.allclose(values["c"], torch.tensor(5.0), rtol=1e-5)
    assert {n for n, _ in model.named_parameters()} == {"lakes_learn_T"}
    with pytest.raises(ValueError):
        model.set_distributed_lakes_from_settings(f, None, learn=("mobilizable_depth_mm",))


def test_learn_list_from_environment(clean_settings, monkeypatch):
    monkeypatch.setenv("MEANDRE_LACS_APPRIS", "share_coefficient,residence_days")
    assert settings.get("lakes.learn") == ("share_coefficient", "residence_days")
    with pytest.raises(ValueError):
        settings._from_file("lakes.learn", "share_coefficient")
