"""Le bilan d'eau de la colonne ferme-t-il aussi quand un profil est DÉCLARÉ.

La fermeture du bilan était acquise sur le chemin de code historique, et vérifiée par l'audit
du pilote régional, qui exige un territoire entier. Le profil déclaré ouvert le 2026-09-20
REMPLACE les sorties de la colonne par celles d'un catalogue : c'est précisément le genre de
recâblage qui perd de l'eau sans que rien ne le signale, la perte se confondant avec un
paramètre mal réglé.

Le test ferme le bilan sur la colonne seule, sans forçage réel ni carte. L'évapotranspiration
potentielle est mise à zéro, les fractions de surface à l'unité : l'entrée est alors la pluie,
les sorties sont les trois productions et la recharge, et la variation de stock est celle des
trois teneurs en eau.
"""
import pytest
import torch

from meandre.hydrotel.bv3c2 import BV3C2Clone, EPAISSEUR, make_params
from meandre.vertical import soil_processes as sp

N = 5
RAIN = 6.0
DAYS = 40


def _params():
    p = dict(make_params(fsa=1.0, fse=0.0, fsi=0.0))
    p["thetacc2"] = torch.full((N,), 0.30)
    p["thetacc3"] = torch.full((N,), 0.30)
    return p


def _simuler(p, n_substep=64, theta=(0.32, 0.34, 0.40)):
    """Cumul des entrées, des sorties et de la variation de stock, en millimètres."""
    col = BV3C2Clone(n_substep=n_substep)
    t1 = torch.full((N,), theta[0])
    t2 = torch.full((N,), theta[1])
    t3 = torch.full((N,), theta[2])
    z1, z2, z3 = EPAISSEUR
    stock0 = (t1 * z1 + t2 * z2 + t3 * z3) * 1000.0
    zero = torch.zeros(N)
    entree = outputs = 0.0
    for _ in range(DAYS):
        ps, ph, pb, rech, (t1, t2, t3), _d = col.forward(
            t1, t2, t3, torch.full((N,), RAIN), zero, zero, zero, p)
        entree += RAIN
        outputs += float((ps + ph + pb + rech).mean())
    stock1 = float(((t1 * z1 + t2 * z2 + t3 * z3) * 1000.0).mean())
    return entree, outputs, stock1 - float(stock0.mean())


def _relative_gap(p, **kw):
    entree, outputs, delta = _simuler(p, **kw)
    return abs(entree - outputs - delta) / max(entree, 1e-9)


PROFILES = {
    "hydrotel déclaré": lambda p: sp.hydrotel_profile(krec=p["krec"]),
    "pile à seuil": lambda p: sp.SoilProfile(layers=3, processes=(
        sp.SoilProcess(layer=2, kind="lateral", form="BASE_LINEAR"),
        sp.SoilProcess(layer=3, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 72.0}),
        sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                       params={"tau": 48.0}))),
    "pile plafonnée": lambda p: sp.SoilProfile(layers=3, processes=(
        sp.SoilProcess(layer=2, kind="lateral", form="BASE_LINEAR"),
        sp.SoilProcess(layer=3, kind="lateral", form="BASE_THRESH_POWER", params={"tau": 72.0}),
        sp.SoilProcess(layer=3, kind="percolation", form="PERC_THRESH_POWER",
                       params={"tau": 48.0}, ceiling=1.0e-3 / 24.0))),
    "exposant non linéaire": lambda p: sp.SoilProfile(layers=3, processes=(
        sp.SoilProcess(layer=2, kind="lateral", form="BASE_LINEAR"),
        sp.SoilProcess(layer=3, kind="lateral", form="BASE_THRESH_POWER",
                       params={"tau": 72.0, "exponent": 2.0}),
        sp.SoilProcess(layer=3, kind="percolation", form="PERC_LINEAR",
                       params={"krec": 2e-5}))),
}


def test_historical_path_closes():
    """Référence : sans déclaration, la colonne ferme à la précision numérique."""
    assert _relative_gap(_params()) < 1e-4


@pytest.mark.parametrize("field_name", list(PROFILES))
def test_each_declared_profile_closes(field_name):
    p = _params()
    p["soil_profile"] = PROFILES[field_name](p)
    assert _relative_gap(p) < 1e-4, f"{field_name} : le bilan ne ferme pas"


def test_empty_profile_loses_no_water_either():
    """Aucun flux déclaré : toute la pluie doit rester en stock ou partir en surface."""
    p = _params()
    p["soil_profile"] = sp.SoilProfile(layers=3, processes=())
    assert _relative_gap(p) < 1e-4


@pytest.mark.parametrize("n_substep", [16, 64])
def test_closure_holds_even_under_substep_cap(n_substep):
    """Sous le plafond, le chemin de l'eau est faux mais la MASSE reste conservée.

    C'est la propriété qui rend la troncature invisible sur un bilan et visible seulement sur
    la partition entre ruissellement et écoulement souterrain.
    """
    p = _params()
    p["soil_profile"] = PROFILES["pile à seuil"](p)
    assert _relative_gap(p, n_substep=n_substep, theta=(0.40, 0.42, 0.47)) < 1e-4
