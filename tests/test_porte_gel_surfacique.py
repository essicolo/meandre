"""Porte de gel surfacique : la part gelée de la couche de surface est la part imperméable.

Sans neige protectrice, un gel qui occupe la moitié de la couche de surface doit faire
ruisseler environ la moitié d'une pluie inférieure à la conductivité saturée. La règle tout
ou rien du clone fait tout ruisseler, la porte continue presque rien.
"""
import torch

import hydrotel_clone.bv3c2 as bv

N = 4


def _ruissellement(porte, gel_cm, neige_mm=0.0):
    bv._SEMI_IMPLICITE = False
    p = bv.make_params("loam", "loam", "loam")
    for k in list(p):
        if torch.is_tensor(p[k]) and p[k].ndim == 0:
            p[k] = p[k].expand(N).clone()
    m = bv.BV3C2Clone(n_substep=32, frozen_gate_continuous=(porte == "continue"))
    m.frozen_gate_areal = porte == "surfacique"
    t = [0.5 * p[f"thetas{i}"] for i in (1, 2, 3)]
    gel = torch.full((N,), float(gel_cm) if gel_cm != "moitie" else 50.0 * float(p["z1"][0]))
    out = m(t[0], t[1], t[2], torch.full((N,), 5.0), torch.zeros(N), gel, torch.full((N,), neige_mm), p)
    return float(out[0][0])


def test_moitie_gelee_ruisselle_la_moitie():
    r = _ruissellement("surfacique", "moitie")
    base = _ruissellement("surfacique", 0.0)
    assert 2.0 < r - base < 3.0


def test_encadree_par_les_deux_autres_regles():
    r_b = _ruissellement("binaire", "moitie")
    r_c = _ruissellement("continue", "moitie")
    r_s = _ruissellement("surfacique", "moitie")
    assert r_c < r_s < r_b


def test_sous_la_neige_le_sol_reste_permeable():
    sans_gel = _ruissellement("surfacique", 0.0, neige_mm=50.0)
    assert abs(_ruissellement("surfacique", "moitie", neige_mm=50.0) - sans_gel) < 1e-9


def test_sans_gel_identique_a_la_regle_d_origine():
    assert abs(_ruissellement("surfacique", 0.0) - _ruissellement("binaire", 0.0)) < 1e-9
