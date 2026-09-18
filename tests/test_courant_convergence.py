"""Convergence de la colonne BV3C2 au plafond de sous-pas de la condition de Courant.

Le binaire C++ boucle jusqu'à épuisement du temps de la journée ; le clone plafonne à
n_substep itérations pour tenir sur le GPU. Le sous-pas étant plafonné à une heure dès
qu'il y a infiltration, une journée de pluie exige au moins vingt-quatre itérations, et
chaque subdivision imposée par Courant multiplie ce nombre. Sous le plafond, le temps non
traité est versé en ruissellement de surface par la fermeture du bilan de masse : la masse
est conservée, le chemin de l'eau ne l'est pas.

Mesuré le 2026-09-18 : sur un sol à 80 pour cent de la teneur en eau à saturation, la
solution convergée ne ruisselle pas du tout, alors que le plafond de production produit
73 mm en trois jours. Ces valeurs sont FIGÉES ici pour qu'une correction du défaut se
voie, et non parce qu'elles seraient acceptables.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydrotel_clone.bv3c2 import CIN_DEFAULT, EPAISSEUR, KREC_DEFAULT, SOIL_TEXTURES, BV3C2Clone

N = 200
TEXTURE = SOIL_TEXTURES["silt_loam"]


def _params(n=N):
    z1, z2, z3 = EPAISSEUR
    t = TEXTURE
    z = lambda v: torch.full((n,), float(v))
    omegpi = (1 + 2 * t["lam"]) / (2 + 2 * t["lam"])
    return dict(
        thetas1=z(t["thetas"]), thetas2=z(t["thetas"]), thetas3=z(t["thetas"]),
        thetacc1=z(t["thetacc"]), thetacc2=z(t["thetacc"]), thetacc3=z(t["thetacc"]),
        thetapf1=z(t["thetapf"]), thetapf2=z(t["thetapf"]), thetapf3=z(t["thetapf"]),
        ks1=z(t["ks"]), ks2=z(t["ks"]), ks3=z(t["ks"]),
        psis1=z(t["psis"]), psis2=z(t["psis"]), psis3=z(t["psis"]),
        b1=z(t["lam"]), b2=z(t["lam"]), b3=z(t["lam"]),
        omegpi1=z(omegpi), omegpi2=z(omegpi), omegpi3=z(omegpi),
        mm1=z(0.1), mm2=z(0.1), mm3=z(0.1), nn1=z(0.1), nn2=z(0.1), nn3=z(0.1),
        krec=z(KREC_DEFAULT), cin=z(CIN_DEFAULT),
        z1=z(z1), z2=z(z2), z3=z(z3),
        slope=z(0.03), fsa=z(1.0), fse=z(0.0), fsi=z(0.0),
        coef_recharge=z(0.0),
    )


def _production(n_substep, saturation, pluie_mm=40.0, jours=3):
    """Production cumulée (surface, hypodermique, profond) en mm, à plafond donné."""
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        clone = BV3C2Clone(n_substep=n_substep, static=True)
        p = _params()
        z = lambda v: torch.full((N,), float(v))
        t1 = t2 = t3 = z(saturation * TEXTURE["thetas"])
        totaux = [0.0, 0.0, 0.0]
        for _ in range(jours):
            surf, hypo, base, _rech, (t1, t2, t3), _diag = clone(
                t1, t2, t3, z(pluie_mm), z(0.1), z(0.0), z(0.0), p, etr1_mm=z(0.0))
            for i, v in enumerate((surf, hypo, base)):
                totaux[i] += float(v.mean())
        return totaux
    finally:
        torch.set_default_dtype(dtype)


@pytest.mark.parametrize("saturation", [0.80, 0.95])
def test_solution_convergee_stable(saturation):
    """Au-delà de 512 sous-pas la solution ne bouge plus : la référence existe."""
    a = _production(512, saturation)
    b = _production(1152, saturation)
    for x, y in zip(a, b):
        assert abs(x - y) < 0.01, f"non convergé à 512 sous-pas : {a} contre {b}"


def test_plafond_production_surestime_le_ruissellement():
    """Le plafond employé fabrique un ruissellement que la solution convergée n'a pas.

    Valeurs figées du 2026-09-18. Un écart à ces chiffres signale que le défaut a bougé,
    dans un sens ou dans l'autre, et demande une relecture du registre.
    """
    surf_64, hypo_64, _ = _production(64, 0.80)
    surf_ref, hypo_ref, _ = _production(512, 0.80)
    assert surf_ref == pytest.approx(0.0, abs=0.01)
    assert surf_64 == pytest.approx(73.19, abs=0.5)
    assert hypo_64 == pytest.approx(0.25, abs=0.02)
    assert hypo_ref == pytest.approx(0.29, abs=0.02)
