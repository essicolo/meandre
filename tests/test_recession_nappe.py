"""La nappe codée produit-elle l'exposant de récession que sa loi annonce.

Contrôle de cohérence entre trois choses qui doivent s'accorder : la loi écrite dans le code,
l'algèbre qui la relie à l'exposant de Brutsaert-Nieber, et les 1,65 mesurés sur 94 stations.

L'algèbre. Pour un réservoir vidangé par Q = c·S^n, sans recharge, dS/dt = −Q donne
dQ/dt = n·c·S^(n−1)·(−c·S^n) = −n·c²·S^(2n−1), et −dQ/dt = a·Q^b impose 2n − 1 = b·n,
soit b = 2 − 1/n. Une loi LINÉAIRE, n = 1, donne b = 1 ; la loi en CARRÉ de la charge de
Dupuit-Boussinesq, n = 2, donne b = 1,5 ; il faut n ≈ 2,9 pour atteindre les 1,65 observés.

Le test vérifie la relation sur la vidange telle que le code la calcule, et non sur une
réécriture à côté : un test qui refait le travail du code ne teste rien.
"""
import numpy as np
import pytest
import torch


def _vidange(n, pas=200000, fraction=0.002, s0=100.0, sous_pas=20):
    """Récession libre d'un réservoir Q = c·S^n, sans recharge.

    Le coefficient est NORMALISÉ pour que le débit initial vaille la même fraction du stock
    quel que soit l'exposant : à coefficient fixe, un exposant de trois vide le réservoir en
    un pas et l'intégration mesure la discrétisation au lieu de la loi. Le sous-pas sert au
    même but, l'erreur d'Euler croissant avec l'exposant.
    """
    c = fraction * s0 ** (1.0 - n)
    s = s0
    dt = 1.0 / sous_pas
    q = []
    for _ in range(pas):
        debit = c * s ** n
        for _ in range(sous_pas):
            s = max(s - c * s ** n * dt, 1e-12)
        q.append(debit)
        if debit < 1e-12:
            break
    return np.array(q)


def _exposant(q, bas=0.05, haut=0.95):
    """Pente de log(−dQ/dt) contre log(Q), sur la partie centrale de la récession."""
    qm = 0.5 * (q[1:] + q[:-1])
    dq = -np.diff(q)
    garde = (dq > 0) & (qm > 0)
    qm, dq = qm[garde], dq[garde]
    lo, hi = np.quantile(qm, bas), np.quantile(qm, haut)
    m = (qm >= lo) & (qm <= hi)
    return float(np.polyfit(np.log(qm[m]), np.log(dq[m]), 1)[0])


@pytest.mark.parametrize("n,attendu", [(1.0, 1.0), (2.0, 1.5), (3.0, 5.0 / 3.0),
                                       (2.9, 2.0 - 1.0 / 2.9)])
def test_la_relation_entre_exposant_de_stock_et_exposant_de_recession(n, attendu):
    assert _exposant(_vidange(n)) == pytest.approx(attendu, abs=0.03)


def test_la_nappe_du_depot_suit_la_loi_de_boussinesq():
    """La pièce réellement utilisée, et non une réécriture, doit donner b proche de 1,5."""
    from meandre.vertical.nappe import NappeLibre

    nappe = NappeLibre()
    sy = torch.tensor([0.05])
    k_b = torch.tensor([1.5e-3])
    z_riv = torch.tensor([5.0])
    h_ref = torch.tensor([2.0])
    z = torch.tensor([1.0])                      # surface libre a un metre sous le sol
    q = []
    for _ in range(3000):
        # forward rend (profondeur, debit, evaporation) : le debit est le DEUXIEME terme,
        # et la profondeur mise a jour le premier.
        z, debit, _e = nappe(z, torch.zeros(1), sy, k_b, z_riv, h_ref)
        v = float(debit)
        if v <= 1e-12:
            break
        q.append(v)
    assert len(q) > 200, "la recession s'arrete trop tot pour ajuster un exposant"
    b = _exposant(np.array(q))
    assert b == pytest.approx(1.5, abs=0.25), (
        f"la nappe rend b = {b:.2f} ; la loi en carre de la charge annonce 1,5")


def test_l_exposant_observe_demande_un_stock_plus_non_lineaire_que_le_carre():
    """1,65 mesuré sur 94 stations correspond à n proche de 2,9, non à n = 2."""
    b_observe = 1.65
    n = 1.0 / (2.0 - b_observe)
    assert n == pytest.approx(2.86, abs=0.05)
    assert _exposant(_vidange(n)) == pytest.approx(b_observe, abs=0.03)
    # La loi en carre, elle, reste sous la valeur observee.
    assert _exposant(_vidange(2.0)) < b_observe
