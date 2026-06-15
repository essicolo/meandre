"""ContextualQuantileHead — tête probabiliste non-paramétrique avec features riches.

Tête quantile à K niveaux τ, monotone par cumsum d'exp(log_widths), médiane
LIBRE (pas ancrée à Q_sim — contrairement à QuantileHead legacy). Le centre
de la distribution est appris séparément.

Features attendues (concaténées) :
  - spatial_params (n_st, F_sp=36) répétés sur T
  - Q_sim, log(Q_sim+1) (T, n_st, 2)
  - Indices IHI : GDD, API, SPI, FN, SWE_proxy (T, n_st, 5) — normalisés z-score
  - doy sin/cos (T, n_st, 2)

Total dimensions : 36 + 2 + 5 + 2 = 45 features (configurable).

Sortie : K quantiles q_τ(t, n) monotones croissants en τ.

NLL = pinball loss sur K niveaux (proper scoring rule, équivalent à CRPS-K).
PIT analytique par interpolation linéaire entre quantile points.

Ref :
- Klotz et al. 2022 (LSTM-CMAL hydro)
- Koenker 2005 (Quantile Regression)
- Gneiting & Raftery 2007 (proper scoring)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ContextualQuantileHead(nn.Module):
    """K-quantile head avec médiane libre et features hydrométéorologiques.

    Parameters
    ----------
    n_features : int
        Dimension totale des features (typiquement 45 = 36 sp + 2 Q + 5 indices + 2 DOY).
    taus : tuple of float
        Niveaux de quantiles, croissants. Default = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95).
        Doit contenir 0.5 pour exposer la médiane.
    hidden : int
        Hidden width du MLP conditionneur.
    """

    def __init__(
        self,
        n_features: int,
        taus: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95),
        hidden: int = 64,
    ) -> None:
        super().__init__()
        assert 0.5 in taus, "taus doit contenir 0.5 pour exposer la médiane"
        self.taus = tuple(taus)
        self.K = len(taus)
        self.median_idx = taus.index(0.5)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * self.K),
        )
        # Init : sortie = 0 → q_τ ≈ Q_sim pour tout τ (départ "identité Gauss collapse")
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def _quantiles(self, features: Tensor, Q_sim: Tensor) -> Tensor:
        """Retourne (M, K) quantile values monotones croissants.

        features : (M, F)  — features tabulaires
        Q_sim    : (M,)    — débit simulé (offset central)
        """
        out = self.net(features)                                 # (M, 2K)
        log_w = out[..., :self.K]                                # log-largeurs
        center_raw = out[..., self.K:].mean(dim=-1, keepdim=True)  # offset libre depuis Q_sim
        widths = log_w.exp()                                     # ≥ 0
        cum = torch.cumsum(widths, dim=-1)                       # (M, K) croissant
        # Re-centrer pour que la médiane (index median_idx) soit à zéro
        median_value = cum[..., self.median_idx:self.median_idx+1]
        q_centered = cum - median_value                          # médiane à 0
        return Q_sim.unsqueeze(-1) + center_raw + q_centered

    def pinball(self, y_obs: Tensor, features: Tensor, Q_sim: Tensor) -> Tensor:
        """Pinball loss multi-τ.

        y_obs : (M,) observations
        features : (M, F), Q_sim : (M,)
        """
        q_pred = self._quantiles(features, Q_sim)                # (M, K)
        resid = y_obs.unsqueeze(-1) - q_pred                     # (M, K)
        taus_t = torch.tensor(self.taus, device=q_pred.device, dtype=q_pred.dtype)
        return torch.maximum(taus_t * resid, (taus_t - 1.0) * resid).mean()

    def cdf_interp(self, y_obs: Tensor, features: Tensor, Q_sim: Tensor) -> Tensor:
        """PIT analytique : interpolation linéaire de la CDF entre les K quantile points.

        Renvoie F(y_obs | x) ∈ [0, 1].
        """
        q = self._quantiles(features, Q_sim)                     # (M, K)
        taus_t = torch.tensor(self.taus, device=q.device, dtype=q.dtype)
        y = y_obs
        out = torch.zeros_like(y)
        # Intérieur : par segment k → k+1
        for k in range(self.K - 1):
            mask = (y >= q[:, k]) & (y < q[:, k + 1])
            w = (q[:, k + 1] - q[:, k]).clamp(min=1e-9)
            out = torch.where(
                mask,
                taus_t[k] + (taus_t[k + 1] - taus_t[k]) * (y - q[:, k]) / w,
                out,
            )
        # Extrapolation queue basse : F ramp linéaire de (0, 0) à (q[0], τ[0])
        below = y < q[:, 0]
        if below.any():
            ramp_low = (taus_t[0] * y / q[:, 0].clamp(min=1e-9)).clamp(min=0.0, max=float(taus_t[0]))
            out = torch.where(below, ramp_low, out)
        # Extrapolation queue haute : F → 1 linéairement au-delà de q[-1]
        above = y >= q[:, -1]
        if above.any():
            extra = ((1.0 - taus_t[-1]) * (y - q[:, -1]) / q[:, -1].clamp(min=1e-9)).clamp(min=0.0)
            out = torch.where(above, (taus_t[-1] + extra).clamp(max=1.0), out)
        return out.clamp(0.0, 1.0)

    def median(self, features: Tensor, Q_sim: Tensor) -> Tensor:
        """Médiane prédictive q_0.5 (utile pour KGE déterministe)."""
        q = self._quantiles(features, Q_sim)
        return q[..., self.median_idx]
