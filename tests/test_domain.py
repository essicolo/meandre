"""Contrat de domaine (2026-10-26) : vocabulaire, validation, manifeste et recette.

Un domaine incohérent est refusé avant d'atteindre le modèle ; un fichier d'entrée qui change
entre l'entraînement et l'évaluation est signalé par la recette.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from meandre.domain import ATTRIBUTES, Domain, Manifest, Observation, check_columns, fingerprint
from meandre.domain import manifest as _manifest
from meandre.routing.graph import RiverGraph
from meandre.spatial.territorial import TerritorialFeatures

REQUIRED = [n for n, a in ATTRIBUTES.items() if a.required]


def _graph(edge_index):
    ei = torch.tensor(edge_index, dtype=torch.long).reshape(2, -1)
    n_e = ei.shape[1]
    return RiverGraph(edge_index=ei, edge_attr=torch.zeros(n_e, 3), topo_order=torch.arange(3), is_lake=torch.zeros(3, dtype=torch.bool), travel_time_days=torch.zeros(n_e, dtype=torch.long))


def _domain(edge_index=((0, 1), (2, 2)), columns=None, forcing_shape=None, obs=None):
    n, T = 3, 5
    cols = columns if columns is not None else REQUIRED
    field_cols = [c for c in cols if c in ATTRIBUTES and ATTRIBUTES[c].role == "field"]
    phys = {c: torch.ones(n) for c in cols if c in ATTRIBUTES and ATTRIBUTES[c].role == "physical"}
    attrs = TerritorialFeatures(torch.zeros(n, len(field_cols)), field_cols, phys)
    forcing = torch.zeros(forcing_shape or (T, n, 2))
    times = pd.date_range("2020-01-01", periods=T)
    return Domain(name="test", node_ids=[1, 2, 3], node_coords=torch.zeros(n, 2), graph=_graph(edge_index), attributes=attrs, attribute_columns=list(cols), forcing=forcing, forcing_variables=["P", "T"], times=times, initial_state=None, withdrawals=None, observations=obs or {}, manifest=Manifest("test"))


def test_un_domaine_sain_est_accepte():
    assert _domain().validate() == []


def test_le_reseau_doit_etre_une_foret():
    d = _domain(edge_index=((0, 0), (1, 2)))   # le tronçon 0 a deux avals
    assert any("forêt" in m for m in d.validate())


def test_attribut_obligatoire_absent_et_colonne_inconnue():
    cols = [c for c in REQUIRED if c != "f_forest"] + ["couleur"]
    issues = check_columns(cols)
    assert any("f_forest" in m for m in issues) and any("couleur" in m for m in issues)
    assert check_columns(REQUIRED + ["f_forest_deciduous_raw", "territorial_siigsol__argile"]) == []


def test_forcage_et_observation_incoherents():
    d = _domain(forcing_shape=(4, 3, 2))
    assert any("forçage" in m for m in d.validate())
    o = Observation("discharge", torch.zeros(5, 1), "site", ["s1"], torch.tensor([7]))
    assert any("tronçon inexistant" in m for m in _domain(obs={"discharge": o}).validate())


def test_le_manifeste_detecte_un_fichier_change(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 1000)
    m = Manifest("t")
    m.add("attributes", f, "test", "attributes")
    saved = m.as_dict()
    assert m.differences(saved) == []
    f.write_bytes(b"y" * 1000)
    m2 = Manifest("t")
    m2.add("attributes", f, "test", "attributes")
    assert any("empreinte" in s for s in m2.differences(saved))
    assert fingerprint(f) != saved["attributes"]["fingerprint"]


def test_la_recette_porte_le_manifeste(tmp_path, monkeypatch):
    from meandre.utils.recipe import capture_recipe, compare_recipe, write_recipe_toml
    f = tmp_path / "forcing.nc"
    f.write_bytes(b"f" * 10)
    m = Manifest("t")
    m.add("forcing", f, "CaSR", "forcing")
    _manifest.register(m)
    try:
        r = capture_recipe()
        assert r["domain"]["forcing"]["fingerprint"] == fingerprint(f)
        p = write_recipe_toml(tmp_path / "modele.pt", r)
        assert "[domain.forcing]" in p.read_text(encoding="utf-8")
        f.write_bytes(b"g" * 10)
        m2 = Manifest("t")
        m2.add("forcing", f, "CaSR", "forcing")
        _manifest.register(m2)
        assert any("domaine : forcing" in e for e in compare_recipe(r))
    finally:
        _manifest.register(None)
