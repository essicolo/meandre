"""Réglages déclarés : priorité environnement, fichier, défaut ; refus de l'ambigu."""
import pytest

from meandre.utils import settings


@pytest.fixture(autouse=True)
def _reinitialise():
    settings._configured.clear()
    yield
    settings._configured.clear()


def test_defaut_sans_configuration(monkeypatch):
    monkeypatch.delenv("MEANDRE_NSUBSTEP", raising=False)
    assert settings.get("column.n_substeps") is None
    assert settings.get("column.z_tensor") is True


def test_alias_environnement_sans_configuration(monkeypatch):
    monkeypatch.setenv("MEANDRE_VSA_B", "8")
    assert settings.get("column.saturated_area_exponent") == 8.0


def test_fichier_puis_environnement(monkeypatch):
    monkeypatch.delenv("MEANDRE_NSUBSTEP", raising=False)
    monkeypatch.delenv("MEANDRE_PENMAN_NEIGE", raising=False)
    pris = settings.configure({"column": {"n_substeps": 64, "penman_snow": True}})
    assert settings.get("column.n_substeps") == 64
    assert settings.get("column.penman_snow") is True
    assert set(pris) == {"column.n_substeps", "column.penman_snow"}
    monkeypatch.setenv("MEANDRE_NSUBSTEP", "16")
    settings.configure({"column": {"n_substeps": 64}})
    assert settings.get("column.n_substeps") == 16


def test_cle_inconnue_refusee():
    with pytest.raises(ValueError, match="inconnu"):
        settings.configure({"column": {"n_substep": 64}})


def test_booleen_ambigu_refuse(monkeypatch):
    monkeypatch.setenv("MEANDRE_Z_TENSEUR", "true")
    with pytest.raises(ValueError, match="ni 0 ni 1"):
        settings.get("column.z_tensor")


def test_instantane_ne_garde_que_les_ecarts(monkeypatch):
    monkeypatch.delenv("MEANDRE_NSUBSTEP", raising=False)
    monkeypatch.delenv("MEANDRE_Z_TENSEUR", raising=False)
    settings.configure({"column": {"n_substeps": 64, "z_tensor": True}})
    snap = settings.snapshot()
    assert snap.get("column.n_substeps") == 64
    assert "column.z_tensor" not in snap


def test_section_du_pilote_ni_lue_ni_refusee(monkeypatch):
    monkeypatch.delenv("MEANDRE_PHENOLOGIE_MODE", raising=False)
    pris = settings.configure({"phenology": {"enabled": True, "mode": "photo", "classes": ["feuillus"]}})
    assert not [k for k in pris if k.startswith("phenology.")]
    assert settings.get("phenology.mode") == "gdd"
    monkeypatch.setenv("MEANDRE_PHENOLOGIE_MODE", "photo")
    assert settings.get("phenology.mode") == "photo"


def test_reglage_texte_du_pilote(monkeypatch):
    for v in ("ETL_AQUIFER", "ETL_NAPPE_EXP", "ETL_SEUIL_NEIGE"):
        monkeypatch.delenv(v, raising=False)
    settings.configure({"groundwater": {"aquifer": True, "exponent": 2.0}, "snow": {"project_threshold": False}})
    assert settings.raw("groundwater.aquifer") == "1"
    assert settings.raw("groundwater.exponent") == "2.0"
    assert settings.raw("snow.project_threshold") == "0"
    assert settings.raw("groundwater.specific_yield", 0.05) == 0.05
