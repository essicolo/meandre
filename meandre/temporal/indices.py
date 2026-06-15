"""Indices hydrométéorologiques interprétables, calculables sur le forçage.

Tous les indices sont :
  - calculés par opérations différentiables (cumsum, conv, exp)
  - nommés et physiquement interprétables
  - mesurables/validables indépendamment du modèle (vs MODIS, SNOTEL, etc.)
  - utilisables soit comme feature de tête probabiliste, soit comme indicateur
    pour un IndexedModulator de paramètre physique

Convention dimensionnelle :
  - Tous les indices sortent en shape (T, N) (même grille temporelle/spatiale que forcing)
  - Sans unité ou avec unité explicite (mm, °C·j, ...)

Indices implémentés :
  - growing_degree_days(T_mean, T_base=10, doy=None)  → GDD cumulé annuel
  - antecedent_precip_index(P, alpha=0.95)            → API exponentiellement pondéré
  - rolling_p_zscore(P, window=30)                    → SPI-like z-score
  - frost_number(T_mean, window=90)                   → FN ∈ [0, 1]
  - snow_accumulation(P, T_min, T_snow=-1)            → SWE proxy (cumulatif)
  - doy_phase(doy)                                    → sin/cos saisonnier
"""
from __future__ import annotations
import math
import torch
from torch import Tensor


def growing_degree_days(
    T_mean: Tensor, T_base: float = 10.0, doy: Tensor | None = None,
) -> Tensor:
    """GDD cumulé annuel.

    GDD(t) = cumsum_t( relu(T_mean - T_base) ) avec reset au 1er janvier.

    Parameters
    ----------
    T_mean : (T, N) float — température moyenne quotidienne (°C)
    T_base : float — seuil base (default 10°C, standard agronomique)
    doy : (T,) int — jour de l'année. Si fourni, reset à doy=1.

    Returns
    -------
    gdd_cum : (T, N) — degrés-jours cumulés (°C·j)
    """
    dgd = torch.relu(T_mean - T_base)
    if doy is None:
        return torch.cumsum(dgd, dim=0)
    # Reset cumulé à chaque nouvel an (doy passe de >1 à 1)
    # Implémentation : on calcule un mask de "reset" et on remet à zéro par segment
    reset = doy == 1                                    # (T,)
    if reset.any():
        # Segments contigus entre resets : cumsum dans chaque segment
        # Approche vectorisée : cumsum global - cumsum value au dernier reset
        cs = torch.cumsum(dgd, dim=0)                   # (T, N)
        # Index du dernier reset pour chaque t : on garde cs[last_reset_t] et soustrait
        last_reset_idx = torch.cummax(
            torch.where(reset, torch.arange(len(doy), device=doy.device),
                        torch.full_like(reset, -1, dtype=torch.long)),
            dim=0,
        ).values                                        # (T,)
        # Pour chaque t, cs_at_last_reset = cs[last_reset_idx]
        # Si pas de reset depuis le début (last_reset_idx == -1), pas de soustraction
        valid_reset = last_reset_idx >= 0
        cs_at_reset = torch.zeros_like(cs)
        if valid_reset.any():
            # Indexer cs : cs[last_reset_idx] mais last_reset_idx peut être -1
            safe_idx = last_reset_idx.clamp(min=0)
            cs_at_reset = cs[safe_idx]                  # (T, N)
            cs_at_reset = cs_at_reset - dgd[safe_idx]   # soustrait la valeur AU reset (qui appartient au nouveau segment)
            cs_at_reset = torch.where(
                valid_reset.unsqueeze(-1), cs_at_reset, torch.zeros_like(cs_at_reset),
            )
        return cs - cs_at_reset
    return torch.cumsum(dgd, dim=0)


def antecedent_precip_index(
    P: Tensor, alpha: float = 0.95,
) -> Tensor:
    """API exponentiellement pondéré (Kohler & Linsley 1951).

    API(t) = alpha * API(t-1) + P(t)
           = somme pondérée de P passé avec demi-vie ≈ -log(2)/log(alpha) jours
    Default alpha=0.95 → demi-vie ≈ 13.5 jours.

    Returns
    -------
    api : (T, N) — mm "effectifs" antécédents
    """
    T = P.shape[0]
    api = torch.zeros_like(P)
    api[0] = P[0]
    for t in range(1, T):
        api[t] = alpha * api[t-1] + P[t]
    return api


def rolling_p_zscore(P: Tensor, window: int = 30) -> Tensor:
    """SPI-like : z-score de la somme P sur fenêtre glissante.

    Approximation simple du Standardized Precipitation Index :
    on cumule P sur `window` jours, puis on normalise par la moyenne/std
    de cette série cumulée. Pas de fit Gamma — c'est un proxy différentiable.

    Returns
    -------
    spi : (T, N) — z-score (≈ N(0, 1) si saisonnalité absorbée)
    """
    # Conv 1D pour somme glissante. Padding pour garder la longueur.
    # P : (T, N) → on traite par station via conv 1d
    T, N = P.shape
    kernel = torch.ones(1, 1, window, device=P.device, dtype=P.dtype) / window
    P_t = P.t().unsqueeze(1)                            # (N, 1, T)
    # padding causal (gauche) pour somme antécédente
    P_pad = torch.nn.functional.pad(P_t, (window - 1, 0))
    P_avg = torch.nn.functional.conv1d(P_pad, kernel)   # (N, 1, T)
    P_avg = P_avg.squeeze(1).t()                        # (T, N)
    # z-score par station
    mu = P_avg.mean(dim=0, keepdim=True)
    sd = P_avg.std(dim=0, keepdim=True) + 1e-6
    return (P_avg - mu) / sd


def frost_number(T_mean: Tensor, window: int = 90) -> Tensor:
    """Frost number (Nelson & Outcalt 1987) sur fenêtre glissante.

    FN = √(FDD) / (√(FDD) + √(TDD))
      où FDD = freezing degree days (T<0) cumulés sur la fenêtre
         TDD = thawing degree days (T>0) cumulés sur la fenêtre
    FN ∈ [0, 1] : 0 = pas de gel, 1 = gel permanent.

    Returns
    -------
    fn : (T, N) — fraction sans unité ∈ [0, 1]
    """
    fdd_daily = torch.relu(-T_mean)                     # (T, N)
    tdd_daily = torch.relu(T_mean)
    T_, N = T_mean.shape
    kernel = torch.ones(1, 1, window, device=T_mean.device, dtype=T_mean.dtype)
    def rolling(x):
        xt = x.t().unsqueeze(1)
        xt_pad = torch.nn.functional.pad(xt, (window - 1, 0))
        return torch.nn.functional.conv1d(xt_pad, kernel).squeeze(1).t()
    FDD = rolling(fdd_daily)
    TDD = rolling(tdd_daily)
    sqrt_F = torch.sqrt(FDD + 1e-3)
    sqrt_T = torch.sqrt(TDD + 1e-3)
    return sqrt_F / (sqrt_F + sqrt_T)


def snow_accumulation(
    P: Tensor, T_min: Tensor, T_snow: float = -1.0, melt_factor: float = 4.0,
    T_melt: float = 0.0,
) -> Tensor:
    """SWE proxy par bilan dégagement / accumulation simple.

    Si T_min < T_snow → P tombe en neige (s'accumule)
    Si T_min ≥ T_melt → fonte = melt_factor × max(0, T_mean - T_melt) (mm/jour)
    Pas de redistribution spatiale, pas de sublimation. Proxy grossier mais
    suffisant pour un indice (ce n'est PAS le SWE physique de la colonne).

    Returns
    -------
    swe_proxy : (T, N) — mm de neige stockée (proxy cumulatif)
    """
    T = P.shape[0]
    is_snow = (T_min < T_snow).float()
    snowfall = P * is_snow                              # (T, N)
    melt = torch.relu(T_min - T_melt) * melt_factor     # (T, N) approx avec T_min
    swe = torch.zeros_like(P)
    swe[0] = snowfall[0]
    for t in range(1, T):
        swe[t] = torch.relu(swe[t-1] + snowfall[t] - melt[t])
    return swe


def doy_phase(doy: Tensor) -> Tensor:
    """Encodage sinusoïdal du jour de l'année.

    Returns (T, 2) avec [sin(2π·doy/366), cos(2π·doy/366)]
    """
    doy_rad = 2 * math.pi * doy.float() / 366.0
    return torch.stack([torch.sin(doy_rad), torch.cos(doy_rad)], dim=-1)


def compute_all_indices(
    forcing: Tensor, doy: Tensor,
    api_alpha: float = 0.95,
    spi_window: int = 30,
    fn_window: int = 90,
) -> dict[str, Tensor]:
    """Calcule tous les indices d'un coup depuis le forçage (T, N, 6).

    Convention forcing (cf. vertical/column.py) :
      forcing[..., 0] = P (mm/j)
      forcing[..., 1] = T_min (°C)
      forcing[..., 2] = T_max (°C)
      forcing[..., 3] = R_n (MJ/m²/j)
      forcing[..., 4] = u2 (m/s)
      forcing[..., 5] = e_a (kPa)

    Returns dict :
      gdd_cum, api_30, spi_30, frost_number_90, swe_proxy, doy_phase (T, 2)
    """
    P = forcing[..., 0]
    T_min = forcing[..., 1]
    T_max = forcing[..., 2]
    T_mean = 0.5 * (T_min + T_max)
    return {
        "gdd_cum": growing_degree_days(T_mean, T_base=10.0, doy=doy),
        "api_30": antecedent_precip_index(P, alpha=api_alpha),
        "spi_30": rolling_p_zscore(P, window=spi_window),
        "frost_number_90": frost_number(T_mean, window=fn_window),
        "swe_proxy": snow_accumulation(P, T_min),
        "doy_phase": doy_phase(doy),                     # (T, 2)
    }
