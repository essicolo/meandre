"""Une table d'attributs supplementaire entre dans le champ par MEANDRE_TERRITORIAL_EXTRA."""
import duckdb
import numpy as np
import pytest
import torch

from meandre.data.basin_cache import BasinCache


def _base(tmp_path, avec_extra):
    f = tmp_path / "b.duckdb"
    con = duckdb.connect(str(f))
    con.execute("CREATE TABLE territorial AS SELECT * FROM (VALUES (0, 1.0, 10.0), (1, 2.0, 20.0), (2, 3.0, 30.0)) t(node_idx, f_forest, area_km2_local)")
    if avec_extra:
        con.execute("CREATE TABLE territorial_siigsol AS SELECT * FROM (VALUES (0, 4.0), (2, 8.0)) t(node_idx, argile)")
    con.close()
    return f


def test_sans_variable_rien_ne_change(tmp_path, monkeypatch):
    monkeypatch.delenv("MEANDRE_TERRITORIAL_EXTRA", raising=False)
    con = duckdb.connect(str(_base(tmp_path, True)), read_only=True)
    t = BasinCache._load_territorial(BasinCache.__new__(BasinCache), con, "cpu")
    assert t.columns == ["f_forest"]


def test_table_jointe_normalisee_et_manquant_a_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("MEANDRE_TERRITORIAL_EXTRA", "territorial_siigsol")
    con = duckdb.connect(str(_base(tmp_path, True)), read_only=True)
    t = BasinCache._load_territorial(BasinCache.__new__(BasinCache), con, "cpu")
    assert t.columns == ["f_forest", "territorial_siigsol__argile"]
    v = t.data[:, 1].numpy()
    # Valeurs 4 et 8 centrees-reduites : -1 et +1 ; le noeud 1, absent, vaut 0.
    assert np.allclose(v, [-1.0, 0.0, 1.0])


def test_table_absente_refusee(tmp_path, monkeypatch):
    monkeypatch.setenv("MEANDRE_TERRITORIAL_EXTRA", "territorial_siigsol")
    con = duckdb.connect(str(_base(tmp_path, False)), read_only=True)
    with pytest.raises(KeyError):
        BasinCache._load_territorial(BasinCache.__new__(BasinCache), con, "cpu")
