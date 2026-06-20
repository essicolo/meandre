"""Clone FIDÈLE du routage de canal ONDE CINÉMATIQUE MODIFIÉE d'Hydrotel, porté
ligne-à-ligne du C++ (source/onde_cinematique_modifiee.cpp), différentiable.
Sous-projet PROPRE.

Onde cinématique diffusive (Manning + correction de pente de friction), résolue
par tronçon avec un schéma implicite Newton-Raphson (TransfertRiviere, l.1651).
Sous-pas internes par jour selon le Courant (Celerite, l.1563), sortie = débit
aval MOYEN sur les sous-pas. Ordre topologique amont→aval, qamont = somme des
qaval amont (non inclus ici : validation tronçon de tête, qamont=0).

Constantes (constantes.hpp) : MAXITER=20, EPSILON=1e-4. Géométrie par tronçon :
longueur/largeur/pente (troncon.trl), Manning rivière défaut 0.04.
"""
from __future__ import annotations
import torch
from torch import Tensor

MAXITER = 20
EPSILON = 1.0e-4


def celerite(lng, lrg, pte, man, qamont, qaval):
    """Célérité onde cinématique modifiée (onde_cinematique_modifiee.cpp:1563)."""
    qmoy = (qamont + qaval) / 2.0
    alpha = (man * lrg ** 0.67) ** 0.6
    beta = 0.6
    r = (alpha * pte ** (-beta / 2.0) / lrg)
    s = beta
    sf = pte - r * (qaval ** s - qamont ** s) / lng
    sf = max(sf, 0.00125) if not torch.is_tensor(sf) else torch.clamp(sf, min=0.00125)
    section = alpha * qmoy ** beta * sf ** (-beta / 2.0)
    if (section == 0.0) if not torch.is_tensor(section) else bool((section == 0).all()):
        return 0.0
    return 1.67 * qmoy / section


def transfert_riviere(dt, lng, lrg, pte, man, qa, ql, qb, qc, qm):
    """Onde cinématique modifiée, un sous-pas, un tronçon (TransfertRiviere, l.1651).
    qa=qamont_prev, ql=qapportlat_prev, qb=qaval_prev, qc=qamont_cur, qm=qapportlat_cur.
    Résolution Newton-Raphson implicite de qd (qaval). Tout en m3/s, dt en s.
    Vectorisable (tenseurs) ; ici scalaires/tenseurs par tronçon."""
    fPdts = float(dt)
    alpha = (man * lrg ** (2.0 / 3.0)) ** 0.6
    beta = 0.6
    r = (alpha / lrg * pte) ** (-0.3)
    s = 0.6
    c1 = 2.0 * alpha * lng / fPdts
    c2 = r / lng
    c3 = qb ** s
    c4 = qb ** beta
    c5 = qc - qb + qa + ql + qm

    # estimation initiale (l.1683-1698)
    qd = pte ** (beta / 2.0) / c1
    qd = qd * (2.0 * (qa - qb) + ql + qm) + c4
    if qd <= 0.0:
        qd = (qa + qb) / 2.0 + (ql + qm) / 2.0
    if qd <= 0.0:
        qd = (qa + qc) / 2.0 + (ql + qm) / 2.0
    if qd <= 0.0:
        qd = qc + qm
    qd = qd ** (1.0 / beta)
    if qd == 0.0:
        qd = 1.0e-20

    c2_eff = c2
    for _ in range(MAXITER):
        v1 = qd ** s
        v2 = qd ** beta
        v3 = pte - c2_eff * (v1 - c3)
        if v3 < pte:
            v3 = pte
            c2_eff = 0.0
        v4 = v3 ** (-beta / 2.0)
        f0 = qd + c1 * v4 * (v2 - c4) - c5
        f1 = beta / 2.0 * v4 / v3 * c2_eff * s * v1 / qd * (v2 - c4)
        f1 = f1 + v4 * beta * v2 / qd
        f1 = 1.0 + c1 * f1
        step = f0 / f1
        qd = qd - step
        if qd <= 0.0:
            qd = 1.0e-20
        if abs(step) < EPSILON:
            break
    return qd


def route_reach_day(apport_lat, lng, lrg, pte, man, state, pdts=86400):
    """Un PAS JOURNALIER pour UN tronçon de tête (qamont=0). apport_lat [m3/s]
    (constant sur les sous-pas). state = dict {qamont, qaval, qapportlat} (prev).
    Réplique la boucle de sous-pas + le débit aval MOYEN. Retourne (debit_aval_moyen,
    new_state)."""
    # nt depuis le Courant (Calcule, l.595-625), tronçon unique
    c = celerite(lng, lrg, pte, man, state["qamont"], state["qaval"])
    nt = int(pdts / int(lng / c + 1.0)) + 1 if c > 0 else 1
    if pdts / nt < 1800:
        dt = 1800
        nt = int(pdts / dt)
    else:
        dt = int(pdts / nt)
    qa = state["qamont"]; ql = state["qapportlat"]; qb = state["qaval"]
    qaval_moy = 0.0
    for t in range(nt):
        qc = 0.0                 # tête : qamont courant = 0
        qm = apport_lat
        qd = transfert_riviere(dt, lng, lrg, pte, man, qa, ql, qb, qc, qm)
        qd = max(0.0, qd)
        # debit aval moyen (running mean, l.1404-1418)
        qaval_moy = qd if t == 0 else (qaval_moy * t + qd) / (t + 1.0)
        # MAJ etat _ocm (l.1363-1365)
        qa, qb, ql = qc, qd, qm
    return qaval_moy, {"qamont": 0.0, "qaval": qb, "qapportlat": ql}
