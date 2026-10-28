"""Exposant de vidange souterraine MESURÉ sur les hydrogrammes d'un territoire.

Pourquoi un ancrage et non un paramètre appris. Il a été mesuré le 2026-09-20 qu'aucune
covariable de terrain ne prédit l'exposant de récession : la texture du sol le prédit à −8 pour
cent contre le témoin, la géologie du socle à −22, le relief à −26. Le faire sortir du champ
spatial est donc condamné d'avance. Mais il se MESURE directement sur les débits observés de
chaque territoire, par la méthode de Brutsaert et Nieber, sans simulation et sans circularité,
exactement comme le multiplicateur d'évapotranspiration de Linacre et les taux de fonte entrent
déjà par la loi des ancrages.

Ce que l'algèbre relie. Pour un réservoir vidangé par Q = c·S^n sans recharge, l'exposant de
Brutsaert-Nieber vaut exactement b = 2 − 1/n, donc n = 1/(2 − b). Un réservoir linéaire donne
b = 1, la loi en carré de la charge de Dupuit-Boussinesq donne b = 1,5. Mesuré sur 76 stations
de cinq territoires, b vaut 1,60 en médiane, soit n = 2,27 ; par territoire, de 1,36 en
Montérégie drainée à 4,65 en Outaouais.

La conversion DIVERGE quand b approche 2 : un exposant de 1,99 donnerait n = 100, ce qui n'a
aucun sens physique. Les stations au-delà de `B_MAX` sont donc écartées, et le résultat est une
médiane par territoire, jamais une valeur par station.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Au-dela de cette valeur, n = 1/(2-b) explose et la station n'informe plus.
B_MAX = 1.95
# Bornes physiques de l'exposant de stock. Un reservoir moins lineaire que 1 n'a pas de sens,
# et au-dela de 6 la vidange devient si brutale a stock plein qu'aucun aquifere ne la produit.
N_MIN, N_MAX = 1.0, 6.0
# Valeur par defaut quand un territoire n'a pas assez de stations : la loi de Boussinesq.
N_DEFAULT = 2.0


@dataclass(frozen=True)
class RecessionAnchor:
    """Exposant de vidange d'un territoire, et de quoi juger sa fiabilité."""
    storage_exponent: float
    recession_exponent: float
    n_stations: int
    etendue: tuple[float, float]

    @property
    def reliable(self) -> bool:
        """Au moins cinq stations : en dessous, la médiane n'est pas une médiane."""
        return self.n_stations >= 5


def recession_segments(q, delay=2, min_duration=5):
    """Jours de décrue franche, hors les `delai` jours suivant une pointe.

    Le délai écarte le ressuyage rapide, qui n'est pas de la vidange souterraine et dont la
    constante de temps est celle du versant.
    """
    decreasing = np.concatenate([[False], np.diff(q) < 0])
    good = np.zeros(len(q), dtype=bool)
    i = 0
    while i < len(q):
        if not decreasing[i]:
            i += 1
            continue
        j = i
        while j < len(q) and decreasing[j]:
            j += 1
        if j - i >= min_duration + delay:
            good[i + delay:j] = True
        i = j
    return good


def _lower_envelope(x, y, n_classes=20, quantile=0.10):
    """Bas du nuage, par quantile dans des tranches d'abscisse d'effectif égal.

    La méthode de Brutsaert et Nieber ajuste l'enveloppe INFÉRIEURE et non le nuage entier :
    les points hauts correspondent aux décrues perturbées par de la pluie, qu'on ne peut pas
    identifier autrement faute de forçage à la station. Le découpage sert uniquement à estimer
    cette enveloppe ; l'ajustement qui suit reste continu sur les points retenus.
    """
    order = np.argsort(x)
    x, y = x[order], y[order]
    bounds = np.linspace(0, len(x), n_classes + 1).astype(int)
    px, py = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 5:
            continue
        kept = y[a:b] <= np.quantile(y[a:b], quantile)
        if kept.any():
            px.append(x[a:b][kept])
            py.append(y[a:b][kept])
    if not px:
        return None, None
    return np.concatenate(px), np.concatenate(py)


def recession_exponent(q) -> float | None:
    """Exposant b de Brutsaert et Nieber pour une station, ou None si la série ne suffit pas."""
    q = np.asarray(q, dtype=float)
    finite = np.isfinite(q) & (q > 0)
    if finite.sum() < 700:
        return None
    q = q[finite]
    good = recession_segments(q)
    dqdt = np.concatenate([[np.nan], -np.diff(q)])
    qm = np.concatenate([[np.nan], 0.5 * (q[1:] + q[:-1])])
    kept = good & np.isfinite(dqdt) & np.isfinite(qm) & (dqdt > 0) & (qm > 0)
    if kept.sum() < 100:
        return None
    ex, ey = _lower_envelope(np.log(qm[kept]), np.log(dqdt[kept]))
    if ex is None or len(ex) < 20:
        return None
    return float(np.polyfit(ex, ey, 1)[0])


def anchor_result(q_obs) -> RecessionAnchor:
    """Exposant de vidange d'un territoire, depuis ses hydrogrammes observés.

    Args:
        q_obs: tableau (temps, stations) de débits observés, valeurs manquantes admises.
    """
    q_obs = np.asarray(q_obs, dtype=float)
    bs = [b for j in range(q_obs.shape[1])
          if (b := recession_exponent(q_obs[:, j])) is not None and b < B_MAX]
    if not bs:
        return RecessionAnchor(N_DEFAULT, 2.0 - 1.0 / N_DEFAULT, 0, (N_DEFAULT, N_DEFAULT))
    b_med = float(np.median(bs))
    ns = np.clip(1.0 / (2.0 - np.asarray(bs)), N_MIN, N_MAX)
    return RecessionAnchor(
        storage_exponent=float(np.clip(1.0 / (2.0 - b_med), N_MIN, N_MAX)),
        recession_exponent=b_med,
        n_stations=len(bs),
        etendue=(float(np.quantile(ns, 0.25)), float(np.quantile(ns, 0.75))))
