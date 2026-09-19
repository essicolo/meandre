"""Nappe libre : équilibre analytique, conservation, et seuils de l'extraction.

La leçon du 2026-09-18 est qu'un schéma peut fabriquer une saison entière sans le dire.
Ces tests vérifient donc la pièce sur ce qui se calcule à la main : la profondeur
d'équilibre sous recharge constante, la fermeture du bilan sur un pas, et le fait que
l'extraction depuis la zone saturée s'éteint là où elle doit.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.vertical.nappe import NappeLibre

SY = 0.05
Z_RIV = 8.0
H_REF = 4.0
K_B = 2.0e-3


def _col(v, n=4):
    return torch.full((n,), float(v), dtype=torch.float64)


def _module(exposant=1.0, n_substep=8):
    return NappeLibre(n_substep=n_substep, exposant=exposant)


def test_equilibre_analytique_cas_lineaire():
    """Sous recharge constante, K_b (h / h_ref) = R fixe la profondeur d'équilibre."""
    torch.set_default_dtype(torch.float64)
    m = _module()
    r = 4.0e-4
    z = _col(6.0)
    for _ in range(20000):
        z, _q, _e = m(z, _col(r), _col(SY), _col(K_B), _col(Z_RIV), _col(H_REF))
    z_eq = Z_RIV - r / K_B * H_REF
    assert float(z[0]) == pytest.approx(z_eq, rel=1e-6)


def test_bilan_ferme_sur_un_pas():
    """Le stock perdu par la nappe vaut exactement ce qui en sort, moins ce qui y entre."""
    torch.set_default_dtype(torch.float64)
    m = _module(exposant=2.0, n_substep=32)
    r = 6.0e-4
    z0 = _col(5.0)
    z1, q, e = m(z0, _col(r), _col(SY), _col(K_B), _col(Z_RIV), _col(H_REF),
                 _col(2.0e-3), _col(9.0))
    variation = SY * (float(z0[0]) - float(z1[0]))
    assert variation == pytest.approx(r - float(q[0]) - float(e[0]), rel=1e-9, abs=1e-12)


def test_extraction_nulle_sous_la_profondeur_d_extinction():
    """Une nappe plus profonde que z_ext ne peut rien céder à l'atmosphère."""
    torch.set_default_dtype(torch.float64)
    m = _module(exposant=2.0)
    z = _col(7.0)
    _z, _q, e_hors = m(z, _col(0.0), _col(SY), _col(K_B), _col(Z_RIV), _col(H_REF),
                       _col(4.0e-3), _col(3.0))
    _z, _q, e_dans = m(z, _col(0.0), _col(SY), _col(K_B), _col(Z_RIV), _col(H_REF),
                       _col(4.0e-3), _col(15.0))
    assert float(e_hors[0]) == 0.0
    assert float(e_dans[0]) > 0.0


def test_convergence_du_sous_pas():
    """Le résultat ne doit pas dépendre du nombre de sous-pas au pas journalier."""
    torch.set_default_dtype(torch.float64)
    r = 8.0e-4
    finals = []
    for ns in (1, 8, 64):
        m = _module(exposant=2.0, n_substep=ns)
        z = _col(6.0)
        for _ in range(2000):
            z, _q, _e = m(z, _col(r), _col(SY), _col(K_B), _col(Z_RIV), _col(H_REF),
                          _col(2.0e-3), _col(9.0))
        finals.append(float(z[0]))
    assert max(finals) - min(finals) < 0.01


def test_amplitude_et_retard_du_cas_lineaire_suivent_la_theorie():
    """Sous recharge sinusoïdale, le réservoir linéaire doit suivre sa fonction de transfert.

    C'est cette borne que la loi non linéaire doit franchir : elle lie amplitude et retard
    par le même coefficient, ce qui interdit les 0,93 m de battement mesurés avec un
    maximum une trentaine de jours après la fonte.
    """
    import math

    torch.set_default_dtype(torch.float64)
    k = K_B / (SY * H_REF)
    gain, retard = NappeLibre.reponse_analytique(k)
    w = 2.0 * math.pi / 365.25
    assert gain == pytest.approx(1.0 / math.sqrt(k * k + w * w), rel=1e-12)
    assert retard == pytest.approx(math.atan2(w, k) / w, rel=1e-12)
    # Un temps de réponse long achète de l'amplitude et paie du retard, toujours.
    gain_lent, retard_lent = NappeLibre.reponse_analytique(k / 10.0)
    assert gain_lent > gain
    assert retard_lent > retard


def test_facteur_de_gradient_neutre_a_un_et_nul_a_zero():
    """Le couplage nappe-colonne éteint le drainage profond quand la nappe affleure.

    Absent du dictionnaire de paramètres, il ne doit rien changer : c'est la condition
    pour que le clone reste fidèle tant que le couplage n'est pas demandé.
    """
    from hydrotel_clone.bv3c2 import SOIL_TEXTURES, BV3C2Clone, make_params

    torch.set_default_dtype(torch.float64)
    p = make_params("silt_loam", "loam", "loam")
    p = {k: (v.expand(4).clone() if torch.is_tensor(v) and v.numel() == 1 else v)
         for k, v in p.items()}
    z = lambda v: torch.full((4,), float(v), dtype=torch.float64)
    t = z(0.9 * SOIL_TEXTURES["silt_loam"]["thetas"])
    cl = BV3C2Clone(n_substep=16)
    args = (t, t, t, z(10.0), z(0.5), z(0.0), z(0.0))
    sans = float(cl(*args, p)[2][0])
    avec_un = float(cl(*args, {**p, "l3_gradient": z(1.0)})[2][0])
    avec_zero = float(cl(*args, {**p, "l3_gradient": z(0.0)})[2][0])
    assert avec_un == pytest.approx(sans, rel=1e-12)
    assert avec_zero == 0.0
    assert sans > 0.0
