"""Fonctions de pédotransfert de Saxton & Rawls (2006), Soil Sci. Soc. Am. J. 70:1569.

Relient les fractions granulométriques (sable, argile) et la matière organique aux
propriétés hydrauliques du sol : porosité, capacité au champ, point de flétrissement,
conductivité à saturation. Relation PUBLIÉE, appliquée nœud par nœud, hors de portée du
gradient — conformément à la règle de conception du 5 août : un paramètre ne reste libre
que si une observation le contraint directement, sinon il est fixé par une loi physique.

Motif : les fractions granulométriques sont déjà dans les attributs territoriaux et
varient fortement (sable médian 0.34 en Abitibi contre 0.92 sur la Côte-Nord), alors que
`init_from_literature` applique un unique loam moyen à toute la province. Douze des
37 paramètres du modèle sont concernés.
"""
from __future__ import annotations

import numpy as np


def saxton_rawls(sand, clay, om_pct=2.5):
    """Propriétés hydrauliques depuis la texture. sand/clay : fractions [0,1].
    om_pct : matière organique en % (2.5 par défaut, sol forestier tempéré).

    Retourne un dict : theta_s (porosité), theta_fc (capacité au champ, -33 kPa),
    theta_wp (point de flétrissement, -1500 kPa), k_sat (m/jour), lam (Brooks-Corey).
    """
    S = np.clip(np.asarray(sand, dtype=float), 0.0, 1.0)
    C = np.clip(np.asarray(clay, dtype=float), 0.0, 1.0)
    OM = np.full_like(S, float(om_pct)) if np.isscalar(om_pct) else np.asarray(om_pct, float)

    # point de flétrissement (-1500 kPa)
    t15t = (-0.024*S + 0.487*C + 0.006*OM + 0.005*(S*OM) - 0.013*(C*OM)
            + 0.068*(S*C) + 0.031)
    theta_wp = t15t + (0.14*t15t - 0.02)

    # capacité au champ (-33 kPa)
    t33t = (-0.251*S + 0.195*C + 0.011*OM + 0.006*(S*OM) - 0.027*(C*OM)
            + 0.452*(S*C) + 0.299)
    theta_fc = t33t + (1.283*t33t**2 - 0.374*t33t - 0.015)

    # porosité au-delà de la capacité au champ
    ts33t = (0.278*S + 0.034*C + 0.022*OM - 0.018*(S*OM) - 0.027*(C*OM)
             - 0.584*(S*C) + 0.078)
    theta_s33 = ts33t + (0.636*ts33t - 0.107)

    theta_s = theta_fc + theta_s33 - 0.097*S + 0.043

    theta_wp = np.clip(theta_wp, 0.01, 0.40)
    theta_fc = np.clip(theta_fc, 0.05, 0.55)
    theta_s = np.clip(theta_s, np.maximum(theta_fc + 0.02, 0.20), 0.70)

    # pente de la courbe de rétention (Brooks-Corey) puis conductivité
    B = (np.log(1500.0) - np.log(33.0)) / (np.log(theta_fc) - np.log(theta_wp))
    lam = 1.0 / B
    k_mm_h = 1930.0 * np.clip(theta_s - theta_fc, 1e-4, None) ** (3.0 - lam)
    k_sat = k_mm_h * 24.0 / 1000.0   # mm/h -> m/jour

    return dict(theta_s=theta_s, theta_fc=theta_fc, theta_wp=theta_wp,
                k_sat=k_sat, lam=lam)
