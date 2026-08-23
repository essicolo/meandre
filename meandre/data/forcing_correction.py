"""Corrections generiques de forcage precipitation (construction des intrants).

Trois operations composables, validees sur SLSO (voir reports/experiment_log.md et
reports/enquete_simat.md) :

1. ``local_day_index`` : bascule des pas horaires UTC vers le jour LOCAL (aligne la
   precipitation sur le jour civil des jauges ; corrige le decalage de frontiere UTC
   qui misplace les orages de fin de journee).
2. ``dedrizzle`` : retire les heures sous un seuil d'intensite (biais crachin des
   analyses CaPA/CaSR qui gonfle les jours pluvieux et etale le timing).
3. ``lapse_rate_adjust`` : ramene une temperature de l'altitude de la grille source a
   l'altitude reelle du noeud par gradient thermique fixe (meme principe que la grille
   SIMAT/GCQ : 0,5 degC/100 m, Bergeron 2016 ; la ou CaSR livre T a l'orographie lissee
   ~10 km du modele, l'ecart cellule-noeud peut depasser 200 m en relief).
4. ``monthly_ratio_merge`` : adopte la STRUCTURE mensuelle et spatiale d'un champ de
   reference krige de stations (SIMAT/GCQ, PyGMET) par ratio climatologique par noeud,
   avec volume cible optionnel (bilan d'eau) — la sequence temporelle de la source est
   intacte (non circulaire : stations meteo seulement, aucune donnee hydrometrique).

Les fonctions operent sur des tableaux numpy (time, node) et des index pandas ;
les drivers regionaux (lecture des tuiles CaSR, echantillonnage aux noeuds) restent
dans .runs/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["local_day_index", "dedrizzle", "scale_volume", "monthly_climatology",
           "monthly_ratio_merge", "lapse_rate_adjust", "attenuate_ratio",
           "confidence_from_uncertainty"]

DAYS_PER_YEAR = 365.25


def local_day_index(times_utc: pd.DatetimeIndex, shift_h: int = -5, dst: bool = False) -> pd.DatetimeIndex:
    """Index horaire decale vers l'heure locale (jour local par ``.normalize()`` ou resample).

    ``dst=True`` applique UTC-4 d'avril a octobre (approximation de l'heure avancee de
    l'Est), ``shift_h`` le reste de l'annee.
    """
    if dst:
        off = np.where(times_utc.month.isin(range(4, 11)), shift_h + 1, shift_h)
        return times_utc + pd.to_timedelta(off, unit="h")
    return times_utc + pd.Timedelta(hours=shift_h)


def dedrizzle(hourly: np.ndarray, threshold_mm_h: float = 0.3) -> np.ndarray:
    """Zero les heures d'intensite inferieure au seuil (mm/h). Ne modifie pas l'entree."""
    return np.where(hourly >= threshold_mm_h, hourly, 0.0)


def scale_volume(p_daily: np.ndarray, target_mm_yr: float) -> tuple[np.ndarray, float]:
    """Recale le volume moyen annuel sur ``target_mm_yr``. Retourne (champ, facteur)."""
    mean_yr = float(p_daily.mean()) * DAYS_PER_YEAR
    if mean_yr <= 0:
        raise ValueError("champ de precipitation nul, volume non recalable")
    factor = target_mm_yr / mean_yr
    return p_daily * factor, factor


def confidence_from_uncertainty(sigma: np.ndarray, clim: np.ndarray,
                                floor_mm_d: float = 0.05) -> np.ndarray:
    """Poids de confiance [0, 1] depuis l'incertitude de la reference.

    ``sigma`` : ecart-type d'estimation de la reference (meme forme que ``clim``,
    p. ex. incertitude PyGMET ou variance d'interpolation GCQ, en mm/j) ;
    ``clim`` : climatologie de la reference (mm/j). w = 1 / (1 + (sigma/clim)^2) :
    la ou l'erreur relative de la reference est faible, w -> 1 (ratio plein) ; la ou
    la reference extrapole (hors reseau, hors domaine), w -> 0 (ratio neutralise).
    """
    rel = np.asarray(sigma, dtype=np.float64) / np.maximum(np.asarray(clim, dtype=np.float64), floor_mm_d)
    return 1.0 / (1.0 + rel * rel)


def attenuate_ratio(ratio: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Attenue un ratio vers 1.0 selon un poids de confiance [0, 1].

    ``ratio_eff = 1 + weight * (ratio - 1)``. ``weight`` est diffusable sur la forme
    de ``ratio`` ((12, N), (N,) ou scalaire). Poids 1 = ratio plein, 0 = neutralise
    (la source est laissee intacte la ou la reference n'est pas fiable).
    """
    w = np.asarray(weight, dtype=np.float64)
    if np.any(w < 0) or np.any(w > 1):
        raise ValueError("poids de confiance hors [0, 1]")
    return 1.0 + w * (np.asarray(ratio, dtype=np.float64) - 1.0)


def lapse_rate_adjust(t: np.ndarray, z_src: np.ndarray, z_dst: np.ndarray,
                      gamma_degc_per_m: float = 0.005) -> np.ndarray:
    """Corrige une temperature de l'altitude source vers l'altitude cible.

    ``t`` (T, N) ou (N,), ``z_src``/``z_dst`` (N,) en metres. Gradient par defaut
    0,005 degC/m (0,5 degC/100 m, celui de la grille SIMAT/GCQ, Bergeron 2016).
    T plus froide quand la cible est plus haute que la source :
    ``t_adj = t - gamma * (z_dst - z_src)``. Ne modifie pas l'entree.
    """
    z_src = np.asarray(z_src, dtype=np.float64)
    z_dst = np.asarray(z_dst, dtype=np.float64)
    if z_src.shape != z_dst.shape:
        raise ValueError(f"altitudes incompatibles {z_src.shape} vs {z_dst.shape}")
    return t - gamma_degc_per_m * (z_dst - z_src)


def monthly_climatology(p_daily: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Moyenne par mois calendaire. ``p_daily`` (T, N), ``months`` (T,) 1..12 -> (12, N)."""
    clim = np.empty((12, p_daily.shape[1]), dtype=np.float64)
    for m in range(1, 13):
        sel = months == m
        if not sel.any():
            raise ValueError(f"aucun jour pour le mois {m}")
        clim[m - 1] = p_daily[sel].mean(0)
    return clim


def monthly_ratio_merge(p_src: np.ndarray, t_src: pd.DatetimeIndex,
                        p_ref: np.ndarray, t_ref: pd.DatetimeIndex,
                        bounds: tuple[float, float] = (0.5, 2.0),
                        clim_floor_mm_d: float = 0.05,
                        target_vol_mm_yr: float | None = None,
                        confidence: np.ndarray | None = None,
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Applique a ``p_src`` la structure mensuelle/spatiale de ``p_ref`` par ratio
    climatologique par noeud, calcule sur la periode commune.

    - ``bounds`` : bornes du ratio (anti-explosion sur climatologies faibles) ;
    - ``clim_floor_mm_d`` : plancher de la climatologie source au denominateur ;
    - ``target_vol_mm_yr`` : si fourni, rescale global final vers ce volume (bilan
      d'eau) ; sinon le volume devient celui de la reference.
    - ``confidence`` : poids [0, 1] diffusable sur (12, N), attenue le ratio vers 1
      la ou la reference n'est pas fiable (voir ``confidence_from_uncertainty`` ;
      p. ex. hors du reseau de stations : la GCQ extrapole hors Quebec sans stations
      americaines et son usage y est deconseille, Bergeron 2016).

    Retourne (p_merge (T, N), ratio (12, N), facteur global).
    La sequence temporelle (timing, jours secs) de la source est preservee.
    """
    if p_src.shape[1] != p_ref.shape[1]:
        raise ValueError(f"noeuds source {p_src.shape[1]} != reference {p_ref.shape[1]}")
    common = t_src.intersection(t_ref)
    if len(common) < 2 * DAYS_PER_YEAR:
        raise ValueError(f"periode commune trop courte ({len(common)} jours) pour une climatologie mensuelle")
    src_c = p_src[t_src.get_indexer(common)]
    ref_c = p_ref[t_ref.get_indexer(common)]
    months_c = common.month.values
    clim_src = np.maximum(monthly_climatology(src_c, months_c), clim_floor_mm_d)
    clim_ref = monthly_climatology(ref_c, months_c)
    ratio = np.clip(clim_ref / clim_src, bounds[0], bounds[1])
    if confidence is not None:
        ratio = attenuate_ratio(ratio, confidence)
    p_merge = p_src * ratio[t_src.month.values - 1]
    if target_vol_mm_yr is not None:
        p_merge, factor = scale_volume(p_merge, target_vol_mm_yr)
    else:
        factor = 1.0
    return p_merge, ratio, factor
