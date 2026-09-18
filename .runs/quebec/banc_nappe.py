"""Banc de la nappe libre : une colonne fictive, une recharge imposée, une cible mesurée.

Les 118 puits appariés du réseau de suivi battent de 0,93 m en médiane et culminent en avril
ou en mai, soit quelques semaines après le maximum de la fonte. L'aquifère actuel du modèle
bat de 2 mm et culmine en août. Ce banc demande, sans réseau, sans routage et sans
entraînement, ce qu'il faut à une nappe pour tenir la cible.

Cinq questions, dans l'ordre, chacune avec sa référence.

  0. Le schéma converge-t-il ? La leçon du 2026-09-18 est qu'un schéma tronqué fabrique une
     fausse saison sans le dire. On le vérifie ici avant tout le reste.
  1. Le cas linéaire reproduit-il sa solution analytique sous recharge constante ?
  2. Sous recharge SINUSOÏDALE, amplitude et retard sont liés par le même coefficient. La
     cible est-elle atteignable ? La théorie dit non ; on le chiffre.
  3. Sous recharge en IMPULSION de printemps, la cible redevient-elle atteignable ? Un
     réservoir chargé par une impulsion culmine à la fin de l'impulsion quel que soit son
     temps de vidange, et son amplitude croît avec ce temps : les deux se découplent.
  4. Que gagne la loi non linéaire de Dupuit-Boussinesq par-dessus cela ?
  5. L'évapotranspiration depuis la zone saturée creuse-t-elle l'étiage d'été ?

    .venv/Scripts/python.exe .runs/quebec/banc_nappe.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.vertical.nappe import NappeLibre

torch.set_default_dtype(torch.float64)

JOURS = 365
ANNEES = 12
RECHARGE_ANNUELLE = 0.150      # m/an, ordre de grandeur mesuré au Québec méridional
SY = 0.05                      # porosité de drainage, till et roc fracturé
Z_RIV = 8.0                    # lit du cours d'eau à 8 m sous le sol
H_REF = 4.0                    # charge de référence
CIBLE_BATTEMENT = 0.93         # m, médiane des puits appariés
CIBLE_RETARD = 30.0            # jours après le maximum de la recharge


def serie_recharge(forme):
    """Recharge journalière sur ANNEES années, en m/j, de moyenne RECHARGE_ANNUELLE."""
    t = np.arange(JOURS * ANNEES)
    jour = t % JOURS
    moyen = RECHARGE_ANNUELLE / JOURS
    if forme == "constante":
        r = np.full_like(t, moyen, dtype=float)
    elif forme == "sinusoidale":
        # Maximum au jour 105, soit le 15 avril. Amplitude relative 1, donc la recharge
        # touche zéro : c'est le maximum de saisonnalité qu'une sinusoïde autorise.
        r = moyen * (1.0 + np.cos(2 * np.pi * (jour - 105) / JOURS))
    elif forme == "impulsion":
        # Fonte concentrée sur 30 jours autour du 15 avril, plus un fond continu. La part
        # de l'année portée par l'impulsion vaut 0,7, ordre de grandeur d'un bassin nival.
        pic = np.exp(-0.5 * ((jour - 105) / 12.0) ** 2)
        pic = pic / pic[:JOURS].sum() * JOURS
        r = moyen * (0.3 + 0.7 * pic)
    else:
        raise ValueError(forme)
    return r


def simuler(recharge, k_b, exposant=1.0, n_substep=8, e_max=None, z_ext=None, z0=4.0,
            saison_et=None):
    """Série de profondeur de nappe (m) pour PLUSIEURS jeux de paramètres à la fois.

    k_b, e_max et z_ext acceptent un scalaire ou une liste ; le module étant vectorisé sur
    les nœuds, un balayage entier tient dans une seule simulation. Rend un tableau de forme
    (jours, jeux) pour la profondeur et un autre pour le débit de base.
    """
    k_b = np.atleast_1d(np.asarray(k_b, dtype=float))
    n = len(k_b)
    col = lambda v: torch.as_tensor(np.broadcast_to(np.atleast_1d(np.asarray(v, dtype=float)), (n,)).copy())
    m = NappeLibre(n_substep=n_substep, exposant=exposant)
    z = col(z0)
    p_k, p_sy, p_riv, p_h = col(k_b), col(SY), col(Z_RIV), col(H_REF)
    p_e = None if e_max is None else col(e_max)
    p_z = None if z_ext is None else col(z_ext)
    zs, qs = [], []
    for i, r in enumerate(recharge):
        # La demande atmosphérique est SAISONNIÈRE : appliquer une extraction constante
        # toute l'année déplace la moyenne sans rien dire du creux d'été.
        e_jour = p_e if (p_e is None or saison_et is None) else p_e * float(saison_et[i])
        z, q, _e = m(z, col(r), p_sy, p_k, p_riv, p_h, e_jour, p_z)
        zs.append(z.numpy().copy())
        qs.append(q.numpy().copy())
    return np.array(zs), np.array(qs)


def cycle(serie, ignorer_annees=4):
    """Cycle saisonnier moyen, une valeur par jour de l'année, par jeu de paramètres."""
    s = serie[ignorer_annees * JOURS:]
    s = s[:len(s) // JOURS * JOURS]
    return s.reshape(-1, JOURS, s.shape[-1]).mean(axis=0)


def battement_et_retard(z, jour_max_recharge=105):
    """Battement (m) et retard du maximum de nappe sur celui de la recharge (jours)."""
    c = cycle(z)
    # La nappe est une PROFONDEUR : son maximum de niveau est son minimum de profondeur.
    jour_haut = np.argmin(c, axis=0)
    retard = (jour_haut - jour_max_recharge) % JOURS
    retard = np.where(retard < JOURS / 2, retard, retard - JOURS)
    return c.max(axis=0) - c.min(axis=0), retard.astype(float)


KS = (2.0e-2, 8.0e-3, 4.0e-3, 2.0e-3, 1.0e-3, 5.0e-4)


def temps_de_reponse(k_b):
    """Temps de vidange du réservoir linéarisé (jours)."""
    return SY * H_REF / k_b


def tableau(r, exposant, titre):
    z, _q = simuler(r, k_b=list(KS), exposant=exposant, n_substep=32)
    b, d = battement_et_retard(z)
    print(f"   {titre}")
    print("   temps de réponse | battement | retard")
    for i, k_b in enumerate(KS):
        print(f"   {temps_de_reponse(k_b):8.0f} j        | {b[i]:6.3f} m  | {d[i]:+4.0f} j")
    print(f"   cible mesurée    | {CIBLE_BATTEMENT:6.3f} m  | {CIBLE_RETARD:+4.0f} j")
    return b, d


def etape_0_convergence():
    print("0. Convergence du schéma. Battement et retard selon le nombre de sous-pas, loi")
    print("   non linéaire sous impulsion, cas le plus raide du banc.")
    r = serie_recharge("impulsion")
    for ns in (1, 2, 4, 8, 16, 32, 64):
        z, _q = simuler(r, k_b=2.0e-3, exposant=2.0, n_substep=ns)
        b, d = battement_et_retard(z)
        print(f"   {ns:3d} sous-pas : battement {b[0]:.4f} m, retard {d[0]:+.0f} j")
    print()


def etape_1_analytique():
    print("1. Cas linéaire sous recharge constante, contre la solution analytique.")
    k_b = 2.0e-3
    z, _q = simuler(serie_recharge("constante"), k_b=k_b, exposant=1.0, n_substep=32)
    # À l'équilibre le débit de base égale la recharge : K_b (h / h_ref) = R.
    h_eq = RECHARGE_ANNUELLE / JOURS / k_b * H_REF
    z_eq = Z_RIV - h_eq
    print(f"   profondeur d'équilibre simulée {z[-1, 0]:.5f} m, analytique {z_eq:.5f} m,"
          f" écart relatif {abs(z[-1, 0] - z_eq) / z_eq:.2e}")
    print()


def etape_2_sinusoidale():
    print("2. Recharge sinusoïdale, loi linéaire. Amplitude et retard sont gouvernés par le")
    print("   même coefficient, donc liés. La colonne de droite est la théorie.")
    z, _q = simuler(serie_recharge("sinusoidale"), k_b=list(KS), exposant=1.0, n_substep=32)
    b, d = battement_et_retard(z)
    print("   temps de réponse | battement | retard | théorie (battement, retard)")
    for i, k_b in enumerate(KS):
        k = 1.0 / temps_de_reponse(k_b)
        gain, retard = NappeLibre.reponse_analytique(k)
        b_a = 2.0 * (RECHARGE_ANNUELLE / JOURS / SY) * gain
        print(f"   {temps_de_reponse(k_b):8.0f} j        | {b[i]:6.3f} m  | {d[i]:+4.0f} j |"
              f" {b_a:6.3f} m, {retard:+4.0f} j")
    print(f"   cible mesurée    | {CIBLE_BATTEMENT:6.3f} m  | {CIBLE_RETARD:+4.0f} j")
    print()


def etape_3_impulsion():
    print("3. Recharge en impulsion de printemps, loi linéaire. Une impulsion place le")
    print("   maximum à sa propre fin, quel que soit le temps de vidange.")
    tableau(serie_recharge("impulsion"), 1.0, "")
    print()


def etape_4_non_lineaire():
    print("4. Même impulsion, loi de Dupuit-Boussinesq en carré de la charge.")
    tableau(serie_recharge("impulsion"), 2.0, "")
    print()


def etape_5_evapotranspiration():
    print("5. Extraction depuis la zone saturée, rampe linéaire jusqu'à la profondeur")
    print("   d'extinction. Effet sur le creux d'été.")
    r = serie_recharge("impulsion")
    # Demande atmosphérique en cloche, maximale à la mi-juillet, nulle l'hiver.
    jour = np.arange(JOURS * ANNEES) % JOURS
    saison = np.clip(np.cos(2 * np.pi * (jour - 196) / JOURS), 0.0, None)
    # La profondeur d'extinction doit ENCADRER la profondeur où la nappe s'établit, sinon
    # la rampe reste nulle et l'essai ne mesure rien.
    cas = ((1e-6, 0.0), (7.0, 0.004), (9.0, 0.004), (9.0, 0.008))
    z, _q = simuler(r, k_b=[2.0e-3] * len(cas), exposant=2.0, n_substep=8,
                    e_max=[c[1] for c in cas], z_ext=[c[0] for c in cas], saison_et=saison)
    b, d = battement_et_retard(z)
    c_cycle = cycle(z)
    print(f"   profondeur moyenne de la nappe sans extraction : {z[:, 0].mean():.2f} m")
    print("   extinction | E max mm/j | battement | retard | mois le plus bas")
    for i, (z_ext, e_max) in enumerate(cas):
        mois_bas = int(np.argmax(c_cycle[:, i]) / JOURS * 12) + 1
        print(f"   {z_ext:5.1f} m    | {e_max * 1000:8.1f}   | {b[i]:6.3f} m  | {d[i]:+4.0f} j |"
              f" {mois_bas:2d}")
    print("   mesuré : creux en septembre ou octobre")
    print()


def main():
    print(f"Nappe libre fictive. Recharge {RECHARGE_ANNUELLE * 1000:.0f} mm/an, porosité de")
    print(f"drainage {SY}, lit du cours d'eau à {Z_RIV} m. Cible : battement {CIBLE_BATTEMENT} m,")
    print(f"maximum {CIBLE_RETARD:.0f} jours après celui de la recharge.\n")
    etape_0_convergence()
    etape_1_analytique()
    etape_2_sinusoidale()
    etape_3_impulsion()
    etape_4_non_lineaire()
    etape_5_evapotranspiration()


if __name__ == "__main__":
    main()
