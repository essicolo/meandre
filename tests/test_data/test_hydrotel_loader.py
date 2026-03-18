"""Tests for the HYDROTEL project loader.

All tests use synthetic in-memory data (no real HYDROTEL files needed).
They validate the parsing logic, graph construction, and data aggregation.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers to write synthetic HYDROTEL files
# ---------------------------------------------------------------------------

def _write_uhrh(path: Path, rows: list[dict]) -> None:
    """Write a minimal uhrh.csv."""
    lines = [
        "RESUMER ZONES HYDROTEL VERSION;4.1.2.0000\n",
        "\n",
        "UHRH ID; TYPE; ALTITUDE MOYENNE (m); PENTE MOYENNE; ORIENTATION MOYENNE;"
        " NB PIXEL; SUPERFICIE (km2); LONGITUDE; LATITUDE\n",
    ]
    for r in rows:
        lines.append(
            f"{r['id']};{r.get('type','SOUS-BASSIN')};"
            f"{r.get('alt',10.0)};{r.get('slope',0.04)};{r.get('orient',180.0)};"
            f"{r.get('npix',10)};{r.get('area',1.0)};"
            f"{r.get('lon',-72.0)};{r.get('lat',46.5)}\n"
        )
    path.write_text("".join(lines), encoding="latin-1")


def _write_occupation_sol_cla(path: Path, rows: list[dict]) -> None:
    """Write occupation_sol.cla with 9 land cover classes."""
    lines = [
        "1\n",
        "9\n",
        'uhrh "no_data" "eau" "sol_nu" "foret_feuillus" "agricole_paturage"'
        ' "foret_coniferes" "impermeable" "tourbiere" "milieu_humide"\n',
    ]
    for r in rows:
        counts = r.get("counts", [0, 0, 0, 100, 50, 100, 10, 20, 20])
        lines.append(f"{r['id']} " + " ".join(str(c) for c in counts) + "\n")
    path.write_text("".join(lines), encoding="latin-1")


def _write_type_sol_cla(path: Path, rows: list[dict]) -> None:
    """Write type_sol.cla with one soil class per UHRH."""
    lines = ["1\n"]
    for r in rows:
        lines.append(f"{r['id']} {r.get('cls', 4)}\n")  # 4 = loam
    path.write_text("".join(lines), encoding="latin-1")


def _write_troncon(path: Path, reaches: list[dict]) -> None:
    """Write a minimal troncon.trl."""
    lines = [
        "2\n",
        f"{len(reaches)}\n",
        "TRONCONS\n",
    ]
    for r in reaches:
        rtype = r.get("type", 1)
        uhrh_ids = r.get("uhrh_ids", [])
        ds = r.get("downstream_id", r["id"])  # default: self (outlet)
        length = r.get("length_m", 1000.0)
        width = r.get("width_m", 5.0)
        slope = r.get("slope", 0.04)
        n_uhrh = len(uhrh_ids)
        uhrh_str = " ".join(str(u) for u in uhrh_ids)

        if rtype == 1:
            upstream = r.get("upstream_node", r["id"])
            downstream_node = r.get("downstream_node", r["id"] + 1)
            lines.append(
                f"    {r['id']} {rtype} {upstream} {downstream_node} "
                f"{length} {width} {slope} {n_uhrh} {uhrh_str} {ds}\n"
            )
        else:
            # Lake: no path nodes for simplicity
            upstream = r.get("upstream_node", r["id"])
            lines.append(
                f"    {r['id']} {rtype} {upstream} 0 "
                f"{length} {width} {slope} 1.5 {n_uhrh} {uhrh_str} {ds}\n"
            )
    path.write_text("".join(lines), encoding="latin-1")


def _write_bilan_vertical(path: Path, rows: list[dict]) -> None:
    lines = [
        "ETATS BILAN VERTICAL;BV3C( 4.1.5.0014 )\n",
        "DATE_HEURE;2023-08-01 00:00\n",
        "\n",
        "UHRH;THETA 1;THETA 2;THETA 3\n",
    ]
    for r in rows:
        t1, t2, t3 = r.get("t1", 0.35), r.get("t2", 0.30), r.get("t3", 0.25)
        lines.append(f"{r['id']};{t1};{t2};{t3}\n")
    path.write_text("".join(lines), encoding="latin-1")


def _write_fonte_neige(path: Path, rows: list[dict]) -> None:
    lines = [
        "ETATS FONTE NEIGE;DEGRE JOUR MODIFIE( 4.1.5.0014 )\n",
        "DATE_HEURE;2023-08-01 00:00\n",
        "\n",
        "UHRH;STOCK CONIFERS;STOCK FEUILLUS;STOCK DECOUVERT;"
        "HAUTEUR CONIFERS;HAUTEUR FEUILLUS;HAUTEUR DECOUVERT;"
        "CHALEUR CONIFERS;CHALEUR FEUILLUS;CHALEUR DECOUVERT;"
        "EAU RETENUE CONIFERS;EAU RETENUE FEUILLUS;EAU RETENUE DECOUVERT;"
        "ALBEDO CONIFERS;ALBEDO FEUILLUS;ALBEDO DECOUVERT\n",
    ]
    for r in rows:
        swe = r.get("swe", 0.0)
        # 15 values after UHRH: stock_con stock_feu stock_dec height... albedo
        vals = [0, 0, swe] + [0] * 12
        lines.append(f"{r['id']};" + ";".join(str(v) for v in vals) + "\n")
    path.write_text("".join(lines), encoding="latin-1")


def _make_simple_project(tmp: Path, n_uhrh: int = 4) -> Path:
    """Create a minimal HYDROTEL project with 2 river troncons."""
    physi = tmp / "physitel"
    etat = tmp / "etat"
    physi.mkdir()
    etat.mkdir()

    # 4 UHRHs in 2 troncons
    uhrh_rows = [
        {"id": i + 1, "alt": 100.0 + i * 10, "slope": 0.05, "orient": 180.0,
         "area": 1.0, "lon": -72.0 + i * 0.01, "lat": 46.5}
        for i in range(n_uhrh)
    ]
    _write_uhrh(physi / "uhrh.csv", uhrh_rows)

    lc_rows = [{"id": i + 1} for i in range(n_uhrh)]
    _write_occupation_sol_cla(physi / "occupation_sol.cla", lc_rows)

    soil_rows = [{"id": i + 1, "cls": 4} for i in range(n_uhrh)]  # loam
    _write_type_sol_cla(physi / "type_sol.cla", soil_rows)

    # 2 river troncons: troncon 1 (UHRHs 1,2) → troncon 2 (UHRHs 3,4) → outlet
    reaches = [
        {
            "id": 1, "type": 1, "upstream_node": 1, "downstream_node": 2,
            "length_m": 5000.0, "width_m": 10.0, "slope": 0.01,
            "uhrh_ids": [1, 2], "downstream_id": 2,
        },
        {
            "id": 2, "type": 1, "upstream_node": 2, "downstream_node": 3,
            "length_m": 3000.0, "width_m": 8.0, "slope": 0.005,
            "uhrh_ids": [3, 4], "downstream_id": 2,  # self → outlet
        },
    ]
    _write_troncon(physi / "troncon.trl", reaches)

    bv_rows = [{"id": i + 1, "t1": 0.35, "t2": 0.30, "t3": 0.25}
               for i in range(n_uhrh)]
    _write_bilan_vertical(etat / "bilan_vertical_2023080100.csv", bv_rows)

    fn_rows = [{"id": i + 1, "swe": 0.0} for i in range(n_uhrh)]
    _write_fonte_neige(etat / "fonte_neige_2023080100.csv", fn_rows)

    return tmp


# ---------------------------------------------------------------------------
# Import the loader
# ---------------------------------------------------------------------------

from meandre.data.physitel_loader import (
    _parse_uhrh,
    _parse_occupation_sol_cla,
    _parse_type_sol_cla,
    _parse_troncon,
    load_hydrotel,
)


# ---------------------------------------------------------------------------
# Unit tests: parsers
# ---------------------------------------------------------------------------

class TestParseUhrh:
    def test_basic_parse(self, tmp_path):
        uhrh_rows = [
            {"id": 1, "alt": 50.0, "slope": 0.05, "orient": 90.0,
             "area": 2.5, "lon": -72.1, "lat": 46.6},
            {"id": 2, "alt": 80.0, "slope": 0.08, "orient": 270.0,
             "area": 1.5, "lon": -72.0, "lat": 46.7},
        ]
        _write_uhrh(tmp_path / "uhrh.csv", uhrh_rows)
        uhrh = _parse_uhrh(tmp_path / "uhrh.csv")

        assert len(uhrh) == 2
        assert uhrh[1]["altitude_m"] == pytest.approx(50.0)
        assert uhrh[2]["area_km2"] == pytest.approx(1.5)
        assert uhrh[1]["lon"] == pytest.approx(-72.1)

    def test_slope_converted_to_pct(self, tmp_path):
        uhrh_rows = [{"id": 1, "slope": 0.04, "area": 1.0}]
        _write_uhrh(tmp_path / "uhrh.csv", uhrh_rows)
        uhrh = _parse_uhrh(tmp_path / "uhrh.csv")
        assert uhrh[1]["slope_pct"] == pytest.approx(4.0)  # 0.04 * 100


class TestParseOccupationSol:
    def test_pixel_counts_shape(self, tmp_path):
        lc_rows = [
            {"id": 1, "counts": [0, 100, 50, 200, 150, 100, 30, 40, 30]},
            {"id": 2, "counts": [0, 0, 10, 500, 200, 300, 0, 0, 0]},
        ]
        uhrh = {1: {}, 2: {}}
        _write_occupation_sol_cla(tmp_path / "occupation_sol.cla", lc_rows)
        lc = _parse_occupation_sol_cla(tmp_path / "occupation_sol.cla", uhrh)

        assert len(lc) == 2
        assert lc[1].shape == (9,)
        assert lc[1][1] == pytest.approx(100.0)  # eau pixels

    def test_pixel_total(self, tmp_path):
        counts = [0, 100, 50, 200, 150, 100, 30, 40, 30]
        lc_rows = [{"id": 1, "counts": counts}]
        _write_occupation_sol_cla(tmp_path / "occupation_sol.cla", lc_rows)
        lc = _parse_occupation_sol_cla(tmp_path / "occupation_sol.cla", {})
        assert lc[1].sum() == pytest.approx(sum(counts))


class TestParseTypeSol:
    def test_returns_class_per_uhrh(self, tmp_path):
        soil_rows = [{"id": 1, "cls": 4}, {"id": 2, "cls": 8}]
        _write_type_sol_cla(tmp_path / "type_sol.cla", soil_rows)
        soil = _parse_type_sol_cla(tmp_path / "type_sol.cla", {})

        assert soil[1] == 4
        assert soil[2] == 8


class TestParseTroncon:
    def test_river_reach_parsed(self, tmp_path):
        reaches = [
            {"id": 1, "type": 1, "upstream_node": 1, "downstream_node": 2,
             "length_m": 1000.0, "width_m": 5.0, "slope": 0.04,
             "uhrh_ids": [1, 2], "downstream_id": 2},
            {"id": 2, "type": 1, "upstream_node": 2, "downstream_node": 3,
             "length_m": 2000.0, "width_m": 6.0, "slope": 0.02,
             "uhrh_ids": [3], "downstream_id": 2},
        ]
        _write_troncon(tmp_path / "troncon.trl", reaches)
        troncons = _parse_troncon(tmp_path / "troncon.trl")

        assert len(troncons) == 2
        assert troncons[0]["id"] == 1
        assert troncons[0]["length_m"] == pytest.approx(1000.0)
        assert troncons[0]["uhrh_ids"] == [1, 2]
        assert troncons[0]["downstream_id"] == 2

    def test_lake_reach_parsed(self, tmp_path):
        reaches = [
            {"id": 1, "type": 2, "upstream_node": 1,
             "length_m": 50000.0, "width_m": 500.0, "slope": 0.001,
             "uhrh_ids": [1, 2, 3], "downstream_id": 1},  # outlet (self)
        ]
        _write_troncon(tmp_path / "troncon.trl", reaches)
        troncons = _parse_troncon(tmp_path / "troncon.trl")

        assert troncons[0]["type"] == 2
        assert troncons[0]["length_m"] == pytest.approx(50000.0)
        assert troncons[0]["uhrh_ids"] == [1, 2, 3]

    def test_reach_count(self, tmp_path):
        reaches = [
            {"id": i, "type": 1, "upstream_node": i, "downstream_node": i + 1,
             "length_m": 500.0, "width_m": 3.0, "slope": 0.04,
             "uhrh_ids": [i], "downstream_id": i + 1 if i < 5 else i}
            for i in range(1, 6)
        ]
        _write_troncon(tmp_path / "troncon.trl", reaches)
        troncons = _parse_troncon(tmp_path / "troncon.trl")
        assert len(troncons) == 5


# ---------------------------------------------------------------------------
# Integration tests: load_hydrotel
# ---------------------------------------------------------------------------

class TestLoadHydrotel:
    @pytest.fixture
    def project(self, tmp_path):
        return _make_simple_project(tmp_path)

    def test_returns_dict_with_required_keys(self, project):
        result = load_hydrotel(project, normalise=False)
        assert "graph" in result
        assert "territorial" in result
        assert "node_coords" in result
        assert "initial_state" in result
        assert "node_ids" in result
        assert "n_nodes" in result

    def test_n_nodes_matches_troncons(self, project):
        result = load_hydrotel(project, normalise=False)
        # 2 troncons
        assert result["n_nodes"] == 2

    def test_graph_is_river_graph(self, project):
        from meandre.routing.graph import RiverGraph
        result = load_hydrotel(project, normalise=False)
        assert isinstance(result["graph"], RiverGraph)

    def test_graph_has_one_edge(self, project):
        result = load_hydrotel(project, normalise=False)
        g = result["graph"]
        # Troncon 1 → troncon 2 (troncon 2 is its own downstream = outlet, no edge)
        assert g.n_edges == 1

    def test_topo_order_valid(self, project):
        result = load_hydrotel(project, normalise=False)
        g = result["graph"]
        assert g.topo_order.shape[0] == 2

    def test_territorial_correct_shape(self, project):
        result = load_hydrotel(project, normalise=False)
        t = result["territorial"]
        # Feature columns are in t.data, accessed by column index
        assert "drainage_area_km2" in t.columns
        assert "f_forest" in t.columns
        idx = t.columns.index("drainage_area_km2")
        assert t.data[:, idx].shape == (2,)

    def test_territorial_fractions_in_range(self, project):
        result = load_hydrotel(project, normalise=False)
        t = result["territorial"]
        # Before normalisation, fractions should be in [0,1]
        for field in ("f_forest", "f_agriculture", "f_urban", "f_wetland", "f_water"):
            idx = t.columns.index(field)
            vals = t.data[:, idx]
            assert vals.min() >= -0.01, f"{field} below 0"
            assert vals.max() <= 1.01, f"{field} above 1"

    def test_node_coords_shape(self, project):
        result = load_hydrotel(project, normalise=False)
        coords = result["node_coords"]
        assert coords.shape == (2, 2)

    def test_initial_state_shape(self, project):
        result = load_hydrotel(project, normalise=False)
        state = result["initial_state"]
        assert state.theta1.shape == (2,)
        assert state.swe.shape == (2,)

    def test_initial_state_theta_in_range(self, project):
        result = load_hydrotel(project, normalise=False)
        state = result["initial_state"]
        assert (state.theta1 >= 0.0).all()
        assert (state.theta1 <= 1.0).all()
        assert (state.theta2 >= 0.0).all()
        assert (state.theta3 >= 0.0).all()

    def test_initial_state_swe_nonneg(self, project):
        result = load_hydrotel(project, normalise=False)
        assert (result["initial_state"].swe >= 0.0).all()

    def test_node_ids_list(self, project):
        result = load_hydrotel(project, normalise=False)
        assert result["node_ids"] == [1, 2]

    def test_is_lake_false_for_river_troncons(self, project):
        result = load_hydrotel(project, normalise=False)
        assert not result["graph"].is_lake.any()

    def test_lake_troncon_is_flagged(self, tmp_path):
        physi = tmp_path / "physitel"
        etat = tmp_path / "etat"
        physi.mkdir(); etat.mkdir()

        uhrh_rows = [{"id": i + 1, "area": 1.0} for i in range(3)]
        _write_uhrh(physi / "uhrh.csv", uhrh_rows)
        _write_occupation_sol_cla(physi / "occupation_sol.cla",
                                  [{"id": i + 1} for i in range(3)])
        _write_type_sol_cla(physi / "type_sol.cla",
                            [{"id": i + 1, "cls": 4} for i in range(3)])

        # Troncon 1 = river, troncon 2 = lake (outlet)
        reaches = [
            {"id": 1, "type": 1, "upstream_node": 1, "downstream_node": 2,
             "length_m": 2000.0, "width_m": 5.0, "slope": 0.04,
             "uhrh_ids": [1], "downstream_id": 2},
            {"id": 2, "type": 2, "upstream_node": 2,
             "length_m": 10000.0, "width_m": 500.0, "slope": 0.001,
             "uhrh_ids": [2, 3], "downstream_id": 2},  # self → outlet
        ]
        _write_troncon(physi / "troncon.trl", reaches)
        _write_bilan_vertical(etat / "bilan_vertical_2023080100.csv",
                              [{"id": i + 1} for i in range(3)])
        _write_fonte_neige(etat / "fonte_neige_2023080100.csv",
                           [{"id": i + 1} for i in range(3)])

        result = load_hydrotel(tmp_path, normalise=False)
        g = result["graph"]

        # Node index for troncon 2
        ni2 = result["node_ids"].index(2)
        assert g.is_lake[ni2].item()
        # Troncon 1 should not be lake
        ni1 = result["node_ids"].index(1)
        assert not g.is_lake[ni1].item()

    def test_drainage_area_increases_downstream(self, project):
        result = load_hydrotel(project, normalise=False)
        t = result["territorial"]
        g = result["graph"]
        # Upstream node drains to downstream — downstream should have larger area
        src = int(g.edge_index[0, 0])
        dst = int(g.edge_index[1, 0])
        idx = t.columns.index("drainage_area_km2")
        assert t.data[dst, idx] > t.data[src, idx]

    def test_no_crash_without_etat(self, tmp_path):
        """Loader issues a warning and falls back to default warm start."""
        physi = tmp_path / "physitel"
        etat = tmp_path / "etat"
        physi.mkdir(); etat.mkdir()

        uhrh_rows = [{"id": 1, "area": 1.0}]
        _write_uhrh(physi / "uhrh.csv", uhrh_rows)
        _write_occupation_sol_cla(physi / "occupation_sol.cla", [{"id": 1}])
        _write_type_sol_cla(physi / "type_sol.cla", [{"id": 1, "cls": 4}])
        reaches = [{"id": 1, "type": 1, "upstream_node": 1, "downstream_node": 2,
                    "length_m": 1000.0, "width_m": 5.0, "slope": 0.04,
                    "uhrh_ids": [1], "downstream_id": 1}]
        _write_troncon(physi / "troncon.trl", reaches)
        # No etat files

        with pytest.warns(UserWarning, match="bilan_vertical"):
            result = load_hydrotel(tmp_path, normalise=False)

        assert result["initial_state"].theta1.shape == (1,)
