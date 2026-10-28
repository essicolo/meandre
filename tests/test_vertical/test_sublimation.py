"""Sublimation du manteau, Kuzmin 1957 (R32, opt-in).

Deuxieme extension de physique assumee (avec la fonte saisonniere) : sur 3203
intervalles apparies CanSWE, 31 % de la neige tombee n'atteint jamais le stock, hors
seuil pluie-neige et hors fonte. La sublimation boreale attendue est de 15 a 40 mm par
hiver -- une part du trou. Kuzmin est la parametrisation la plus simple de Raven
(SUBLIM_KUZMIN) et ne demande que u2 et e_a, deja dans le forcage.

Deux contrats non negociables : la fidelite par defaut (mode absent = clone identique),
et la CONSERVATION -- la sublimation est une sortie atmospherique qui doit apparaitre
au diagnostic, sinon l'audit de fermeture lit une fuite (le piege du milieu humide,
dette #12).
"""
import torch
import pytest

from meandre.hydrotel.snow import sublimation_kuzmin


def test_winter_order_of_magnitude():
    """Un hiver boreal type doit sublimer 15-40 mm, pas 3 ni 300.

    Conditions moyennes : -10 degres, vent 3 m/s, humidite relative 70 % (e_a = 0.7 x
    saturation). 150 jours d'hiver.
    """
    t = torch.tensor([-10.0])
    es_kpa = 0.6112 * torch.exp(torch.tensor(21.87 * -10.0 / (-10.0 + 265.5)))
    ea = 0.7 * es_kpa
    e_day = sublimation_kuzmin(torch.tensor([3.0]), ea, t)
    winter = float(e_day) * 150
    assert 10.0 < winter < 60.0, f"hiver simule {winter:.0f} mm, hors de la plage boreale"


def test_saturated_air_does_not_sublimate():
    """Deficit de vapeur nul => sublimation nulle. Le clamp interdit la CONDENSATION
    (deficit negatif) : Kuzmin ne modelise pas le givre, on ne cree pas d'eau."""
    t = torch.tensor([-5.0])
    es_kpa = 0.6112 * torch.exp(torch.tensor(21.87 * -5.0 / (-5.0 + 265.5)))
    assert float(sublimation_kuzmin(torch.tensor([4.0]), es_kpa * 1.0001, t)) == 0.0
    assert float(sublimation_kuzmin(torch.tensor([4.0]), es_kpa * 1.5, t)) == 0.0


def test_wind_increases_sublimation():
    t = torch.tensor([-8.0]); ea = torch.tensor([0.2])
    e1 = sublimation_kuzmin(torch.tensor([1.0]), ea, t)
    e5 = sublimation_kuzmin(torch.tensor([5.0]), ea, t)
    assert float(e5) > float(e1) > 0.0


def test_saturation_over_ice_not_water():
    """A -20 degres, la saturation sur GLACE est ~20 % sous la courbe eau liquide.
    Utiliser la courbe eau surestimerait la sublimation d'autant. Constantes glace
    de Magnus-Tetens : 21.87 / 265.5."""
    es_ice = 6.112 * torch.exp(torch.tensor(21.87 * -20.0 / (-20.0 + 265.5)))
    es_water = 6.112 * torch.exp(torch.tensor(17.62 * -20.0 / (-20.0 + 243.12)))
    assert float(es_ice) < float(es_water) * 0.90


def test_snowpack_surface_capped_at_zero():
    """Par temps doux la surface du manteau reste a 0 degre : la sublimation ne doit
    pas exploser avec la temperature de l'air."""
    ea = torch.tensor([0.4]); u = torch.tensor([3.0])
    e_doux = sublimation_kuzmin(u, ea, torch.tensor([8.0]))
    e_zero = sublimation_kuzmin(u, ea, torch.tensor([0.0]))
    assert torch.allclose(e_doux, e_zero)


# ── Integration dans la colonne : fidelite par defaut et conservation ────────

def _column(mode=None, n=3):
    """Colonne reelle, calquee sur tests/smoke_hydrotel_column.py (build_static_params)."""
    from meandre.vertical.hydrotel_column import HydrotelColumn, build_static_params
    from meandre.hydrotel.frost import n_intervalles
    occ = dict(feuillus=0.3, ouverts=0.2, humides=0.0, urbain=0.2, routes=0.1, eau=0.2)
    psnow, psoil, petr = build_static_params(
        n, lat=46.0, slope=0.03, orientation=7, texture="sandy_loam",
        z=(0.1, 0.4, 1.0), land_cover_table=occ)
    col = HydrotelColumn(et_mode="mcguinness", use_frost=True)
    if mode is not None:
        col.sublimation_mode = mode
    col.set_static(psnow, psoil, petr, wetland=None, n_depth=n_intervalles(1.5, 0.05))
    return col


def _winter(col, n=3):
    """Trois jours de neige puis un jour sec, froid et vente."""
    st = col.init_state(n, theta_init=(0.3, 0.3, 0.3))
    t = lambda v: torch.full((n,), float(v))
    diag = None
    subl = torch.zeros(n)
    for j, (p, tmn, tmx) in enumerate([(12.0, -14, -6), (8.0, -12, -5),
                                       (10.0, -15, -7), (0.0, -13, -6)]):
        prod, st, diag = col(t(p), t(tmn), t(tmx), t(15.0), t(4.0), t(0.25),
                             float(20 + j), st)
        if "subl_mm" in diag:
            subl = subl + diag["subl_mm"]
    return diag, subl


def test_without_mode_clone_is_identical():
    d0, _ = _winter(_column())
    d1, _ = _winter(_column(mode=None))
    assert torch.equal(d0["couvert_nival_mm"], d1["couvert_nival_mm"])
    assert "subl_mm" not in d0


def test_kuzmin_removes_from_storage_and_reports_it():
    """La masse sublimee doit (1) sortir du manteau et (2) etre declaree au diagnostic.

    L'egalite stock_sans - stock_avec == cumul_sublime est la CONSERVATION elle-meme :
    si elle casse, l'audit ETL_BILAN lit une fuite, exactement comme pour le milieu
    humide avant le 2026-08-20 (dette #12).
    """
    d_without, _ = _winter(_column())
    d_with, subl = _winter(_column(mode="kuzmin"))
    assert (subl > 0).all(), "hiver froid, sec et vente : Kuzmin doit sublimer"
    gap = d_without["couvert_nival_mm"] - d_with["couvert_nival_mm"]
    assert torch.allclose(gap, subl, atol=1e-6), (
        "ce qui a quitte le manteau doit EGALER ce qui est declare au diagnostic")
