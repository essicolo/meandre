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

from meandre.utils.recette import (
    capturer_recette, comparer_recette, ecrire_recette_toml,
)


def test_capture_les_variables_du_projet(monkeypatch):
    monkeypatch.setenv("ETL_SEUIL_TWB", "-0.8")
    monkeypatch.setenv("MEANDRE_KMUSK", "4,48,24")
    monkeypatch.setenv("SANS_RAPPORT", "ignore")

    r = capturer_recette()

    assert r["variables"]["ETL_SEUIL_TWB"] == "-0.8"
    assert r["variables"]["MEANDRE_KMUSK"] == "4,48,24"
    assert "SANS_RAPPORT" not in r["variables"]


def test_ne_capture_jamais_un_secret(monkeypatch):
    """La recette est faite pour être lue et partagée."""
    monkeypatch.setenv("MEANDRE_TOKEN", "tres-secret")
    monkeypatch.setenv("ETL_API_KEY", "aussi-secret")

    r = capturer_recette()

    assert not any("secret" in str(v) for v in r["variables"].values())


def test_le_toml_ecrit_se_relit(tmp_path, monkeypatch):
    monkeypatch.setenv("ETL_REGION", "gasp")
    monkeypatch.setenv("ETL_MELT_SAISON", "0.5")
    cible = tmp_path / "modele.pt"

    p = ecrire_recette_toml(cible, capturer_recette(),
                            fiche={"et_mode": "linacre", "hgm": True},
                            init_kwargs={"n_nodes": 42})

    assert p is not None and p.name == "modele.recette.toml"
    lu = tomllib.loads(p.read_text(encoding="utf-8"))
    assert lu["recette"]["ETL_REGION"] == "gasp"
    assert lu["recette"]["ETL_MELT_SAISON"] == "0.5"
    assert lu["modele"]["et_mode"] == "linacre"
    assert lu["modele"]["hgm"] is True
    assert lu["construction"]["n_nodes"] == 42


def test_l_ecart_de_recette_est_signale(monkeypatch):
    monkeypatch.setenv("ETL_SEUIL_TWB", "-0.8")
    sauvee = capturer_recette()
    monkeypatch.setenv("ETL_SEUIL_TWB", "0.0")

    ecarts = comparer_recette(sauvee)

    assert any("ETL_SEUIL_TWB" in e for e in ecarts)


def test_un_chemin_de_deploiement_ne_compte_pas_pour_un_ecart(monkeypatch):
    """Les racines de données changent d'une machine à l'autre sans changer la physique."""
    monkeypatch.setenv("MEANDRE_DATA", "D:/meandre-data")
    sauvee = capturer_recette()
    monkeypatch.setenv("MEANDRE_DATA", "/scratch/atlas01/donnees")

    assert comparer_recette(sauvee) == []


def test_le_point_de_reprise_porte_sa_recette(tmp_path, monkeypatch):
    """Bout en bout : sauvegarde, TOML voisin, et relecture par torch."""
    from meandre.model import HydroModel

    monkeypatch.setenv("ETL_MELT_SAISON", "0.5")
    m = HydroModel(n_nodes=8, n_territorial=8, n_forcing=6, use_temporal=False,
                   use_residual=False, use_travel_time_attn=False, param_mode="nerf",
                   column_mode="hydrotel", compile_soil=False)
    p = tmp_path / "essai.pt"
    m.save(p)

    assert (tmp_path / "essai.recette.toml").exists()
    # Le chargement securise de torch (weights_only, defaut depuis 2.6) doit accepter la
    # recette : elle ne doit contenir que des types simples. Un objet TorchVersion y a
    # rendu tout point de reprise illisible pendant la mise au point.
    ck = torch.load(p, map_location="cpu", weights_only=True)
    assert ck["recette"]["variables"]["ETL_MELT_SAISON"] == "0.5"
    assert isinstance(ck["recette"]["contexte"].get("torch"), str)
