"""Réglages déclarés : priorité environnement, fichier, défaut ; refus de l'ambigu."""
import pytest

from meandre.utils import settings


@pytest.fixture(autouse=True)
def _reinitialise():
    settings._configured.clear()
    yield
    settings._configured.clear()


def test_default_without_configuration(monkeypatch):
    monkeypatch.delenv("MEANDRE_NSUBSTEP", raising=False)
    assert settings.get("column.n_substeps") is None
    assert settings.get("column.semi_implicit_soil") is False


def test_environment_alias_without_configuration(monkeypatch):
    monkeypatch.setenv("MEANDRE_VSA_B", "8")
    assert settings.get("column.saturated_area_exponent") == 8.0


def test_file_then_environment(monkeypatch):
    monkeypatch.delenv("MEANDRE_NSUBSTEP", raising=False)
    monkeypatch.delenv("MEANDRE_PENMAN_NEIGE", raising=False)
    taken = settings.configure({"column": {"n_substeps": 64, "penman_snow": True}})
    assert settings.get("column.n_substeps") == 64
    assert settings.get("column.penman_snow") is True
    assert set(taken) == {"column.n_substeps", "column.penman_snow"}
    monkeypatch.setenv("MEANDRE_NSUBSTEP", "16")
    settings.configure({"column": {"n_substeps": 64}})
    assert settings.get("column.n_substeps") == 16


def test_unknown_key_refused():
    with pytest.raises(ValueError, match="inconnu"):
        settings.configure({"column": {"n_substep": 64}})


def test_ambiguous_boolean_refused(monkeypatch):
    monkeypatch.setenv("MEANDRE_SOL_SEMI_IMPLICITE", "true")
    with pytest.raises(ValueError, match="ni 0 ni 1"):
        settings.get("column.semi_implicit_soil")


def test_snapshot_keeps_only_differences(monkeypatch):
    monkeypatch.delenv("MEANDRE_NSUBSTEP", raising=False)
    monkeypatch.delenv("MEANDRE_SOL_SEMI_IMPLICITE", raising=False)
    settings.configure({"column": {"n_substeps": 64, "semi_implicit_soil": False}})
    snap = settings.snapshot()
    assert snap.get("column.n_substeps") == 64
    assert "column.semi_implicit_soil" not in snap


def test_pilot_section_neither_read_nor_refused(monkeypatch):
    monkeypatch.delenv("MEANDRE_PHENOLOGIE_CLASSES", raising=False)
    taken = settings.configure({"phenology": {"enabled": True, "classes": ["feuillus"]}})
    assert not [k for k in taken if k.startswith("phenology.")]
    assert settings.get("phenology.classes") is None
    monkeypatch.setenv("MEANDRE_PHENOLOGIE_CLASSES", "feuillus")
    assert settings.get("phenology.classes") == "feuillus"


def test_pilot_text_setting(monkeypatch):
    for v in ("ETL_AQUIFER", "ETL_NAPPE_EXP", "ETL_EVAP_EAU_LIBRE"):
        monkeypatch.delenv(v, raising=False)
    settings.configure({"groundwater": {"aquifer": True, "exponent": 2.0}, "evaporation": {"open_water": False}})
    assert settings.raw("groundwater.aquifer") == "1"
    assert settings.raw("groundwater.exponent") == "2.0"
    assert settings.raw("evaporation.open_water") == "0"
    assert settings.raw("groundwater.specific_yield", 0.05) == 0.05
