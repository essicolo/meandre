"""Mixture Density Network head for non-parametric conditional density of Q.

Produces a conditional distribution p(Q_obs | features) as a mixture of K
Gaussian components. Conditionneur MLP takes the same features as the rest
of the model (spatial_params, Q_sim, log Q_sim) and outputs (π_k, μ_k, σ_k)
per (t, n).

  p(y | x) = Σ_k π_k(x) · N(y | μ_k(x), σ_k(x)²)

CMAL precedent : Klotz et al. 2022 utilise un mélange de Laplaciens
asymétriques pour la même raison (queues lourdes + asymétrie). GMM est le
choix de premier ordre, plus stable ; on peut upgrader vers AL si besoin.

Sorties utiles :
  - log_prob(y, x)      : log densité (loss = -log_prob)
  - cdf(y, x)           : CDF analytique → PIT direct
  - quantile(τ, x)      : inversion numérique (Newton) si besoin
  - crps(y, x)          : CRPS closed-form pour GMM
  - sample(x, n)        : tirage MC pour diagnostic

Conditionnement : forward(features) → (log_pi, mu, log_sigma) de shape
(T, N, K) chacun. La somme softmax(log_pi) = 1 est gérée via log_softmax.

Ancrage initial : μ_k initialisé par défaut autour de Q_sim (offset zéro),
σ_k modérée (~log Q_sim std), π_k uniforme. À chaud, ressemble à une
gaussienne unique centrée sur Q_sim.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


SQRT_2 = math.sqrt(2.0)
SQRT_2_PI = math.sqrt(2.0 * math.pi)
LOG_SQRT_2_PI = 0.5 * math.log(2.0 * math.pi)


class MixtureDensityHead(nn.Module):
    """Conditional Gaussian Mixture density.

    Parameters
    ----------
    n_features : int
        Width of spatial features (typically 36 = SpatialParams.N_PARAMS).
    n_components : int
        Number of mixture components K. Default 10 (Klotz 2022).
    hidden : int
        Hidden width of conditioner MLP.
    use_log_q : bool
        Include log(Q_sim+1) as extra feature.
    sigma_min, sigma_max : float
        Bounds on σ_k (in linear m³/s) for numerical stability.
    """

    def __init__(
        self,
        n_features: int = 36,
        n_components: int = 10,
        hidden: int = 64,
        use_log_q: bool = True,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e4,
    ) -> None:
        super().__init__()
        self.K = n_components
        self.use_log_q = use_log_q
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        in_dim = n_features + 1 + (1 if use_log_q else 0)  # +Q_sim, +log Q_sim
        # Conditioner MLP : in_dim → 3K (log_pi, μ_offset, log_σ)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * n_components),
        )
        # Init : sortie ≈ (uniform π, zéro μ-offset, log σ ≈ log(Q_sim_typique)).
        # On veut que au démarrage : μ_k = Q_sim ∀k, σ_k = quelque chose modéré.
        # Last layer biases : π_k logits = 0 (uniform), μ_offset bias = 0,
        # log_σ bias initialisé à log(3.0) ≈ 1.1 (σ ~ 3 m³/s par défaut).
        with torch.no_grad():
            self.net[-1].weight.zero_()
            bias = self.net[-1].bias
            bias[:n_components].zero_()                       # log π = 0 → uniform
            bias[n_components:2*n_components].zero_()         # μ_offset = 0
            bias[2*n_components:].fill_(math.log(3.0))        # log σ = log 3

    def _conditioner(self, features: Tensor, Q_sim: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """features: (..., n_features), Q_sim: (...,). Returns log_π, μ, log_σ each (..., K)."""
        Q_safe = Q_sim.clamp(min=0.0)
        inp_parts = [features, Q_sim.unsqueeze(-1)]
        if self.use_log_q:
            inp_parts.append(torch.log(Q_safe + 1.0).unsqueeze(-1))
        x = torch.cat(inp_parts, dim=-1)
        out = self.net(x)                                       # (..., 3K)
        log_pi_raw = out[..., :self.K]
        mu_off = out[..., self.K:2*self.K]
        log_sigma_raw = out[..., 2*self.K:]
        # log π normalisé via log_softmax (somme π = 1)
        log_pi = F.log_softmax(log_pi_raw, dim=-1)
        # μ = Q_sim + offset learné (ancré sur la physique, mais libre de bouger)
        mu = Q_sim.unsqueeze(-1) + mu_off
        # σ borné pour stabilité numérique
        log_sigma = log_sigma_raw.clamp(
            min=math.log(self.sigma_min), max=math.log(self.sigma_max),
        )
        return log_pi, mu, log_sigma

    def forward(self, features: Tensor, Q_sim: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """features: (T, N, n_features), Q_sim: (T, N). Returns log_π, μ, log_σ each (T, N, K)."""
        return self._conditioner(features, Q_sim)

    def log_prob(self, y: Tensor, features: Tensor, Q_sim: Tensor) -> Tensor:
        """log p(y | x) via log-sum-exp pour stabilité.

        y : (M,) observations (1D flat)
        features : (M, n_features), Q_sim : (M,)
        Returns : (M,) log densité
        """
        log_pi, mu, log_sigma = self._conditioner(features, Q_sim)  # (M, K)
        sigma = log_sigma.exp()
        # log N(y | μ_k, σ_k²) = -0.5 * ((y - μ_k)/σ_k)² - log σ_k - log √(2π)
        z = (y.unsqueeze(-1) - mu) / sigma
        log_normal = -0.5 * z * z - log_sigma - LOG_SQRT_2_PI    # (M, K)
        return torch.logsumexp(log_pi + log_normal, dim=-1)      # (M,)

    def cdf(self, y: Tensor, features: Tensor, Q_sim: Tensor) -> Tensor:
        """F(y | x) = Σ_k π_k Φ((y - μ_k)/σ_k). Pour calcul PIT.

        y : (M,), features (M, n_features), Q_sim : (M,)
        Returns (M,) ∈ [0, 1]
        """
        log_pi, mu, log_sigma = self._conditioner(features, Q_sim)
        sigma = log_sigma.exp()
        z = (y.unsqueeze(-1) - mu) / sigma
        # Φ(z) = 0.5 * (1 + erf(z/√2))
        Phi_k = 0.5 * (1.0 + torch.erf(z / SQRT_2))               # (M, K)
        pi = log_pi.exp()
        return (pi * Phi_k).sum(dim=-1).clamp(0.0, 1.0)

    def crps_gaussian_mixture(
        self, y: Tensor, features: Tensor, Q_sim: Tensor,
    ) -> Tensor:
        """CRPS closed-form pour mélange gaussien (Grimit et al. 2006).

        CRPS(F, y) = Σ_k π_k · A(y, μ_k, σ_k)
                    − 0.5 · Σ_{k,l} π_k π_l · A_pairwise(μ_k, μ_l, σ_k, σ_l)

        avec A(y, μ, σ) = σ · [z·(2Φ(z) − 1) + 2φ(z) − 1/√π · 0]  Note: terme pour 1 composante
        Formule simplifiée Grimit 2006 :
          CRPS_N(y; μ, σ) = σ · (z · (2Φ(z) − 1) + 2φ(z) − 1/√π)
        Pour mélange, on doit aussi soustraire le terme de divergence interne.
        Voir Grimit, Gneiting, Berrocal, Johnson 2006 « The continuous ranked
        probability score for circular variables and its application to mesoscale
        forecast ensemble verification ».
        """
        log_pi, mu, log_sigma = self._conditioner(features, Q_sim)
        sigma = log_sigma.exp()
        # E[|X-y|] = Σ_k π_k · σ_k · [z_k·(2Φ(z_k)−1) + 2·φ(z_k)]
        # (le terme −1/√π appartient à E[|X-X'|] traité plus bas — ne pas l'inclure ici)
        z = (y.unsqueeze(-1) - mu) / sigma                        # (M, K)
        Phi_z = 0.5 * (1.0 + torch.erf(z / SQRT_2))
        phi_z = torch.exp(-0.5 * z * z) / SQRT_2_PI
        crps_per_k = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z)    # (M, K)
        pi = log_pi.exp()
        term_obs = (pi * crps_per_k).sum(dim=-1)                  # (M,)
        # Terme de divergence interne : 0.5 · Σ_{k,l} π_k π_l · A(μ_k, μ_l, σ_k, σ_l)
        # A(μ_k, μ_l, σ_k, σ_l) = (μ_k − μ_l) · (2Φ(δ) − 1) + 2 · √(σ_k² + σ_l²) · φ(δ)
        # avec δ = (μ_k − μ_l) / √(σ_k² + σ_l²)
        mu_k = mu.unsqueeze(-2)                                    # (M, 1, K)
        mu_l = mu.unsqueeze(-1)                                    # (M, K, 1)
        s_k2 = (sigma.unsqueeze(-2)) ** 2
        s_l2 = (sigma.unsqueeze(-1)) ** 2
        ss = torch.sqrt(s_k2 + s_l2 + 1e-12)
        diff = mu_k - mu_l
        delta = diff / ss
        Phi_d = 0.5 * (1.0 + torch.erf(delta / SQRT_2))
        phi_d = torch.exp(-0.5 * delta * delta) / SQRT_2_PI
        A_pair = diff * (2 * Phi_d - 1) + 2 * ss * phi_d           # (M, K, K)
        pi_k = pi.unsqueeze(-2)
        pi_l = pi.unsqueeze(-1)
        term_div = 0.5 * (pi_k * pi_l * A_pair).sum(dim=(-1, -2))  # (M,)
        return term_obs - term_div                                  # (M,)

    def sample(
        self, features: Tensor, Q_sim: Tensor, n_samples: int = 1,
    ) -> Tensor:
        """Tirage MC : pour chaque (t, n), tire n_samples.

        Returns : (n_samples, ...) où ... a la même forme que Q_sim.
        """
        log_pi, mu, log_sigma = self._conditioner(features, Q_sim)
        sigma = log_sigma.exp()
        # Tirer la composante k selon π_k
        K = self.K
        batch_shape = Q_sim.shape
        # log_pi a shape batch_shape + (K,)
        # On veut tirer n_samples × batch_shape indices k
        log_pi_flat = log_pi.reshape(-1, K)
        # Gumbel-softmax sampling pour différentiabilité (mais ici sample is for diag)
        k = torch.distributions.Categorical(logits=log_pi_flat).sample((n_samples,))  # (n_samples, batch)
        # Indices pour gather
        mu_flat = mu.reshape(-1, K)
        sigma_flat = sigma.reshape(-1, K)
        idx_exp = k.unsqueeze(-1)                                   # (n_samples, batch, 1)
        mu_sel = mu_flat.unsqueeze(0).expand(n_samples, -1, -1).gather(-1, idx_exp).squeeze(-1)
        sigma_sel = sigma_flat.unsqueeze(0).expand(n_samples, -1, -1).gather(-1, idx_exp).squeeze(-1)
        z = torch.randn_like(mu_sel)
        y = mu_sel + sigma_sel * z
        return y.reshape((n_samples,) + batch_shape)
