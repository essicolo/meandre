"""Les clés déclarées d'une section du pilote ([phenology]) se lisent dans le fichier (2026-11-11) :
`crop_params` était ignorée en silence, si bien que la phénologie des cultures n'agissait que posée
par variable d'environnement."""
import os

from meandre.utils import settings


def test_crop_params_read_from_file(monkeypatch):
    monkeypatch.delenv(settings.SCHEMA["phenology.crop_params"].env, raising=False)
    cfg = {"phenology": {"enabled": True, "params": [56.81, 11.82, 21.1, 0.9], "classes": ["feuillus", "agri"], "crop_params": [500.0, 11.82, 100.0, 0.9]}}
    from_file = settings.configure(cfg)
    assert settings.get("phenology.crop_params") == (500.0, 11.82, 100.0, 0.9)
    assert "phenology.crop_params" in from_file
    settings.configure({})


def test_pilot_keys_still_accepted():
    settings.configure({"phenology": {"enabled": True, "params": [1.0, 2.0, 3.0, 4.0], "classes": ["agri"]}})
    assert settings.get("phenology.crop_params") is None
    settings.configure({})
