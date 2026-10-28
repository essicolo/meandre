"""La nappe apprise part exactement de la nappe posée, et ses scalaires portent un gradient.

L'équivalence numérique sur une simulation complète est vérifiée sur le banc : une passe avant
avec ETL_NAPPE_APPRIS=1 doit rendre le même KGE que la nappe posée (registre, 2026-09-30).
"""
import torch

from meandre.hydrotel.frost import n_intervalles
from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params

N = 3
OCC = dict(feuillus=0.6, conifers=0.3, ouverts=0.1, humides=0.0, urbain=0.0, routes=0.0, eau=0.0, mixtes=0.0)


def _column(apprise):
    torch.set_default_dtype(torch.float64)
    psnow, psoil, petr = build_static_params(N, lat=45.3, slope=0.02, orientation=7, texture="loam", z=(0.2, 0.4, 1.0), land_cover_table=OCC)
    col = HydrotelColumn(et_mode="mcguinness", use_frost=False, use_aquifer=True)
    col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(1.5, 0.05))
    col.enable_free_water_table(sy=0.05, k_b=1.5e-3, z_riv=5.0, h_ref=2.0, e_frac=0.15, z_ext=6.0, exponent=2.0, learned=apprise)
    return col


def _drainage(col, apprise, day_count=30):
    p = col._water_table
    if apprise:
        kb = torch.exp(col.nappe_log_kb)
        col.free_water_table.exponent = 1.0 + torch.exp(col.nappe_log_exp1)
        ef = torch.sigmoid(col.nappe_logit_efrac)
    else:
        kb, ef = torch.tensor(p["k_b"]), torch.tensor(p["e_frac"])
    z = torch.full((N,), 2.0)
    full = lambda v: torch.full((N,), float(v))
    q_tot = torch.zeros(N)
    for _ in range(day_count):
        z, q, e = col.free_water_table(z, full(0.0005), full(p["sy"]), torch.ones(N) * kb, full(p["z_riv"]), full(p["h_ref"]), full(0.003) * ef, full(p["z_ext"]))
        q_tot = q_tot + q
    return q_tot


def test_identical_at_initialization():
    a = _drainage(_column(False), False).detach()
    b = _drainage(_column(True), True).detach()
    assert torch.allclose(a, b, rtol=1e-10, atol=1e-12)


def test_values_and_gradients():
    col = _column(True)
    v = col.water_table_values()
    assert abs(v["k_b"] - 1.5e-3) < 1e-12 and abs(v["e_frac"] - 0.15) < 1e-12 and abs(v["exposant"] - 2.0) < 1e-12
    _drainage(col, True).sum().backward()
    for nom in ("nappe_log_kb", "nappe_logit_efrac", "nappe_log_exp1"):
        g = getattr(col, nom).grad
        assert g is not None and torch.isfinite(g) and float(g) != 0.0, nom
