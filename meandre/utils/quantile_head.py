"""Tête de régression quantile per-nœud.

Prédit K offsets δ_τ depuis μ (médiane par construction = μ, donc δ_0.5 = 0
implicite — on ne prédit jamais le quantile 0.5). Architecture :

    spatial_params (N, P) → MLP → 2K sorties (K log-width centers + K Q-slopes)
    log_w_τ(t, n) = a_τ(n) + b_τ(n) · log(|Q(t, n)| + ε)
    w_τ = exp(log_w_τ)                              # largeurs positives
    Monotonie par cumsum de chaque côté de la médiane :
      δ_τ<0.5 = −cumsum(w des plus proches de 0.5 vers les extrêmes)
      δ_τ>0.5 = +cumsum                              (idem)

→ q_τ(t, n) = μ(t, n) + δ_τ(t, n), strictement croissant en τ.

Hétéroscédasticité : les largeurs grossissent avec |Q| via b_τ · log|Q|
(comme le SpatialNoiseHead actuel pour σ). Init b_τ ≈ 1 → mise à l'échelle
quasi-linéaire avec Q ; a_τ ≈ −1 → largeurs initiales modestes.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QuantileHead(nn.Module):
    """Per-node quantile head with monotone offsets from μ.

    Parameters
    ----------
    n_spatial_params : int
        Number of spatial parameters per node (input to the MLP).
    hidden : int
        Hidden layer width.
    taus : tuple of floats in (0, 1) excluding 0.5
        Quantile levels to predict. Sorted ascending internally.
    eps : float
        Stabiliser inside ``log(|Q| + ε)``.
    """

    def __init__(
        self,
        n_spatial_params: int = 36,
        hidden: int = 32,
        taus: tuple[float, ...] = (0.05, 0.10, 0.25, 0.75, 0.90, 0.95),
        eps: float = 1.0,
    ) -> None:
        super().__init__()
        taus_sorted = sorted(set(taus))
        if any(t <= 0.0 or t >= 1.0 or t == 0.5 for t in taus_sorted):
            raise ValueError(f"taus must be in (0,1) and exclude 0.5; got {taus}")
        self.taus = tuple(taus_sorted)
        self.K = len(self.taus)
        self.n_lower = sum(1 for t in self.taus if t < 0.5)
        self.n_upper = self.K - self.n_lower
        self.eps = eps

        # MLP : spatial_params → 2K (K width-centers a_τ + K Q-slopes b_τ)
        self.net = nn.Sequential(
            nn.Linear(n_spatial_params, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.K),
        )
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            # a_τ ≈ −1 → largeur init exp(−1) ≈ 0.37
            self.net[-1].bias[: self.K] = -1.0
            # b_τ ≈ 1 → mise à l'échelle hétéroscédastique log|Q|
            self.net[-1].bias[self.K:] = 1.0

    def forward(self, spatial_params: Tensor, Q: Tensor) -> Tensor:
        """
        Parameters
        ----------
        spatial_params : Tensor, shape (N, P)
        Q : Tensor, shape (T, N) — débit prédit (μ)

        Returns
        -------
        offsets : Tensor, shape (T, N, K) — δ_τ tel que q_τ = μ + δ_τ.
                  Strictement croissant le long de self.taus.
        """
        raw = self.net(spatial_params)              # (N, 2K)
        a = raw[:, : self.K]                         # (N, K) log-width centers
        b = raw[:, self.K:]                          # (N, K) Q-slopes
        log_q = torch.log(Q.abs() + self.eps)       # (T, N)

        # log_w (T, N, K) = a + b · log|Q|, broadcast
        log_w = a.unsqueeze(0) + b.unsqueeze(0) * log_q.unsqueeze(-1)
        w = log_w.exp()                              # (T, N, K) > 0

        # Côté bas (τ < 0.5) : largeurs ordonnées par τ ascendant déjà
        # → on reverse pour cumsumer depuis le plus proche de 0.5
        w_lower = w[..., : self.n_lower]             # (T, N, n_lower)
        w_lower_rev = w_lower.flip(-1)               # plus proche 0.5 en premier
        cumsum_lower_rev = w_lower_rev.cumsum(dim=-1)
        offsets_lower = -cumsum_lower_rev.flip(-1)   # négatif, ordre τ ascendant

        # Côté haut (τ > 0.5) : ordre τ ascendant = plus proche 0.5 en premier
        w_upper = w[..., self.n_lower:]              # (T, N, n_upper)
        offsets_upper = w_upper.cumsum(dim=-1)       # positif, ordre τ ascendant

        return torch.cat([offsets_lower, offsets_upper], dim=-1)  # (T, N, K)
