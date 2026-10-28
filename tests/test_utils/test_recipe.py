"""La recette d'exécution : ce qui a produit un point de reprise est-il conservé ?

Inventaire du 2026-09-03 : sur les 54 fichiers de configuration du domaine québécois,
deux sont réellement chargés par le code, et le pilote d'entraînement lit à lui seul 79
variables d'environnement, dont l'essentiel de la physique. Soixante-dix réglages
n'avaient aucune représentation persistée ailleurs que dans le script shell de lancement.
La recette les capture dans le point de reprise et à côté de lui, en TOML lisible.
"""
import tomllib

import pytest
import torch

from meandre.utils.recipe import (
    capture_recipe, compare_recipe, write_recipe_toml,
)


def test_captures_project_variables(monkeypatch):
    monkeypatch.setenv("ETL_SEUIL_TWB", "-0.8")
    monkeypatch.setenv("MEANDRE_KMUSK", "4,48,24")
    monkeypatch.setenv("SANS_RAPPORT", "ignore")

    r = capture_recipe()

    assert r["variables"]["ETL_SEUIL_TWB"] == "-0.8"
    assert r["variables"]["MEANDRE_KMUSK"] == "4,48,24"
    assert "SANS_RAPPORT" not in r["variables"]


def test_never_captures_a_secret(monkeypatch):
    """La recette est faite pour être lue et partagée."""
    monkeypatch.setenv("MEANDRE_TOKEN", "tres-secret")
    monkeypatch.setenv("ETL_API_KEY", "aussi-secret")

    r = capture_recipe()

    assert not any("secret" in str(v) for v in r["variables"].values())


def test_written_toml_reloads(tmp_path, monkeypatch):
    monkeypatch.setenv("ETL_REGION", "gasp")
    monkeypatch.setenv("ETL_MELT_SAISON", "0.5")
    target = tmp_path / "modele.pt"

    p = write_recipe_toml(target, capture_recipe(),
                            record={"et_mode": "linacre", "hgm": True},
                            init_kwargs={"n_nodes": 42})

    assert p is not None and p.name == "modele.recipe.toml"
    read = tomllib.loads(p.read_text(encoding="utf-8"))
    assert read["recipe"]["ETL_REGION"] == "gasp"
    assert read["recipe"]["ETL_MELT_SAISON"] == "0.5"
    assert read["modele"]["et_mode"] == "linacre"
    assert read["modele"]["hgm"] is True
    assert read["construction"]["n_nodes"] == 42


def test_recipe_difference_is_reported(monkeypatch):
    monkeypatch.setenv("ETL_SEUIL_TWB", "-0.8")
    saved = capture_recipe()
    monkeypatch.setenv("ETL_SEUIL_TWB", "0.0")

    gaps = compare_recipe(saved)

    assert any("ETL_SEUIL_TWB" in e for e in gaps)


def test_deployment_path_is_not_a_difference(monkeypatch):
    """Les racines de données changent d'une machine à l'autre sans changer la physique."""
    monkeypatch.setenv("MEANDRE_DATA", "D:/meandre-data")
    saved = capture_recipe()
    monkeypatch.setenv("MEANDRE_DATA", "/scratch/atlas01/donnees")

    assert compare_recipe(saved) == []


def test_checkpoint_carries_its_recipe(tmp_path, monkeypatch):
    """Bout en bout : sauvegarde, TOML voisin, et relecture par torch."""
    from meandre.model import HydroModel

    monkeypatch.setenv("ETL_MELT_SAISON", "0.5")
    m = HydroModel(n_nodes=8, n_territorial=8, n_forcing=6, param_mode="nerf",
                   column_mode="hydrotel", compile_soil=False)
    p = tmp_path / "essai.pt"
    m.save(p)

    assert (tmp_path / "essai.recipe.toml").exists()
    # Le chargement securise de torch (weights_only, defaut depuis 2.6) doit accepter la
    # recette : elle ne doit contenir que des types simples. Un objet TorchVersion y a
    # rendu tout point de reprise illisible pendant la mise au point.
    ck = torch.load(p, map_location="cpu", weights_only=True)
    assert ck["recipe"]["variables"]["ETL_MELT_SAISON"] == "0.5"
    assert isinstance(ck["recipe"]["contexte"].get("torch"), str)


def test_file_recipe_never_overrides_environment(monkeypatch):
    """Un script de grappe qui pose son bloc de variables doit se comporter comme avant."""
    from meandre.utils.recipe import apply_recipe

    monkeypatch.setenv("ETL_SEUIL_TWB", "-2.0")
    monkeypatch.delenv("ETL_MELT_SAISON", raising=False)

    posees = apply_recipe({"ETL_SEUIL_TWB": -0.8, "ETL_MELT_SAISON": 0.5})

    import os
    assert os.environ["ETL_SEUIL_TWB"] == "-2.0", "l'environnement prime"
    assert os.environ["ETL_MELT_SAISON"] == "0.5", "le fichier comble le manque"
    assert posees == ["ETL_MELT_SAISON"]


def test_boolean_becomes_readable_flag(monkeypatch):
    from meandre.utils.recipe import apply_recipe
    import os

    monkeypatch.delenv("ETL_AQUIFER", raising=False)
    monkeypatch.delenv("ETL_WSNOW", raising=False)

    apply_recipe({"ETL_AQUIFER": True, "ETL_WSNOW": False})

    assert os.environ["ETL_AQUIFER"] == "1"
    assert os.environ["ETL_WSNOW"] == "0"


def test_key_without_project_prefix_refused():
    """Une cle que personne ne lira doit echouer bruyamment, pas dormir dans un fichier."""
    from meandre.utils.recipe import apply_recipe

    with pytest.raises(ValueError, match="prefixe"):
        apply_recipe({"SEUIL_TWB": -0.8})


def test_socle_file_is_readable_and_applicable(monkeypatch):
    """Le socle livre encode la loi des ancrages et doit s'appliquer sans erreur."""
    import os
    from pathlib import Path
    from meandre.utils.recipe import apply_recipe

    file_path = Path(__file__).resolve().parents[2] / "experiments/archive/config/socle.toml"
    if not file_path.exists():
        pytest.skip("socle.toml absent")
    _cfg = tomllib.loads(file_path.read_text(encoding="utf-8"))
    section = _cfg.get("recipe") or _cfg["recette"]
    for key in section:
        monkeypatch.delenv(key, raising=False)

    posees = apply_recipe(section)

    assert "ETL_INIT_HYDROTEL" in posees
    assert os.environ["ETL_INIT_HYDROTEL"] == "sauf_ks"
    assert os.environ["ETL_ETP"] == "linacre"
    assert os.environ["ETL_SEUIL_TWB"] == "-0.8"
