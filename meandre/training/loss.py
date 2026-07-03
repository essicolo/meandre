"""Multi-objective differentiable loss for hydrological calibration.

All terms are smooth functions of Q_sim so gradients flow back through the
full model. Use meandre.utils.metrics for non-differentiable evaluation.

L = w1*(1-NSE) + w2*|PBIAS|/100 + w3*(1-KGE)
  + w4*L_snow  + w5*L_ET
  + w6*L_physics  + w7*L_residual_reg
"""

import logging
import math

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def differentiable_nse_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """1 - NSE, formulated as a minimization target. Perfect = 0."""
    num = ((q_obs - q_sim) ** 2).sum()
    denom = ((q_obs - q_obs.mean()) ** 2).sum()
    return num / (denom + 1e-8)


def _kge_components(q_obs: Tensor, q_sim: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute KGE and its three components (r, beta, gamma).

    Returns:
        r:     Pearson correlation
        beta:  bias ratio (mu_sim / mu_obs)
        gamma: variability ratio (cv_sim / cv_obs)
        kge:   Kling-Gupta efficiency
    """
    x = q_obs - q_obs.mean()
    y = q_sim - q_sim.mean()
    var_obs = (x ** 2).mean()
    var_sim = (y ** 2).mean()
    std_obs = torch.sqrt(var_obs + 1e-8)
    std_sim = torch.sqrt(var_sim + 1e-8)
    r = (x * y).mean() / (std_obs * std_sim)
    r = r.clamp(-1.0, 1.0)

    mu_obs = q_obs.mean().clamp(min=1e-8)
    beta = q_sim.mean() / mu_obs
    gamma = (std_sim / std_obs) / (beta.abs().clamp(min=1e-8))

    sq = (r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2
    kge = 1.0 - torch.sqrt(sq + 1e-8)
    return r, beta, gamma, kge


def differentiable_kge_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """1 - KGE, formulated as a minimization target. Perfect = 0."""
    _, _, _, kge = _kge_components(q_obs, q_sim)
    return 1.0 - kge


def differentiable_composite_kge_loss(
    q_obs: Tensor, q_sim: Tensor, alpha: float = 0.5, eps: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Composite KGE loss: alpha*(1-KGE) + (1-alpha)*(1-KGE_log).

    KGE on normal flows captures peaks; KGE on log flows captures baseflow.
    Reference: Pool et al. (2018), Kratzert et al. (2019).

    Returns:
        loss: scalar
        info: dict with r, beta, gamma, kge (normal), r_log, beta_log, gamma_log, kge_log
    """
    r, beta, gamma, kge = _kge_components(q_obs, q_sim)

    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    r_log, beta_log, gamma_log, kge_log = _kge_components(log_obs, log_sim)

    loss = alpha * (1.0 - kge) + (1.0 - alpha) * (1.0 - kge_log)

    info = {
        "r": r, "beta": beta, "gamma": gamma, "kge": kge,
        "r_log": r_log, "beta_log": beta_log, "gamma_log": gamma_log, "kge_log": kge_log,
    }
    return loss, info


def differentiable_pbias_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """|PBIAS| / 100, in [0, ~1] range for loss weighting."""
    pbias = (q_sim - q_obs).sum() / (q_obs.sum() + 1e-8)
    return pbias.abs()


def box_cox(x: Tensor, lam: float, eps: float = 1e-3) -> Tensor:
    """Box-Cox transform sur Q. lam=1 → identité, lam=0.3 → standard hydro
    (Bates & Campbell 2001), lam=0 → log. Clamp à eps pour éviter Q≤0.
    """
    x_safe = x.clamp(min=eps)
    if lam == 0.0:
        return torch.log(x_safe)
    if lam == 1.0:
        return x_safe - 1.0
    return (x_safe.pow(lam) - 1.0) / lam


def gaussian_nll_loss(
    q_obs: Tensor, q_sim: Tensor, log_sigma: Tensor, lam: float = 1.0,
) -> Tensor:
    """Heteroscedastic Gaussian NLL en espace Box-Cox (lam=1 = linéaire).

        NLL = 0.5 * ((T(q_obs) - T(q_sim)) / σ)² + log σ

    σ est dans l'espace transformé (le noise_head s'adapte). lam=0.3 est
    le standard hydro (Bates & Campbell 2001) — résidus quasi-normaux.

    Empty inputs (no valid obs) return 0 with requires_grad=False so the
    chunked training loop can skip the backward pass.
    """
    if q_obs.numel() == 0:
        return torch.zeros((), device=q_sim.device)
    if lam != 1.0:
        q_obs = box_cox(q_obs, lam)
        q_sim = box_cox(q_sim, lam)
    sigma2 = (2.0 * log_sigma).exp()
    nll = 0.5 * (q_obs - q_sim).pow(2) / sigma2 + log_sigma
    return nll.mean()


def student_t_nll_loss(
    q_obs: Tensor, q_sim: Tensor, log_sigma: Tensor, log_df: Tensor,
    lam: float = 1.0,
) -> Tensor:
    """Heteroscedastic Student-t NLL (queues lourdes) en espace Box-Cox.

    Pour des résidus leptokurtiques (PIT gaussien = pic central, σ gonflé pour
    couvrir les queues), la Student-t ajuste séparément l'échelle σ et la
    lourdeur des queues via ν = exp(log_df). ν→∞ ⇒ gaussienne.

        z = (T(y) − T(μ)) / σ
        NLL = log σ + ½(ν+1)·log(1 + z²/ν)
              − logΓ((ν+1)/2) + logΓ(ν/2) + ½·log(ν·π)

    ``log_df`` peut être un scalaire global appris ou un tenseur par nœud.
    """
    if q_obs.numel() == 0:
        return torch.zeros((), device=q_sim.device)
    if lam != 1.0:
        q_obs = box_cox(q_obs, lam)
        q_sim = box_cox(q_sim, lam)
    nu = log_df.exp().clamp(min=1e-2)
    sigma = log_sigma.exp()
    z = (q_obs - q_sim) / sigma
    nll = (log_sigma + 0.5 * (nu + 1.0) * torch.log1p(z * z / nu)
           - torch.lgamma((nu + 1.0) / 2.0) + torch.lgamma(nu / 2.0)
           + 0.5 * torch.log(nu * math.pi))
    return nll.mean()


def flatness_loss(
    q_obs: Tensor,
    q_sim: Tensor,
    log_sigma: Tensor,
    lam: float = 0.3,
    n_bins: int = 21,
    bandwidth: float = 0.02,
) -> Tensor:
    """Loss de calibration directe : pénalise la déviation de l'histogramme
    PIT par rapport à uniforme.

    Pipeline (tout différentiable) :
        1. T(q) via Box-Cox(lam)
        2. PIT_i = Φ((T(y_i) − T(μ_i)) / σ_i)
        3. Histogramme soft via kernel gaussien (bandwidth)
        4. δ² = mean((freq_k − 1/K)²) / (1/K)²  (sans cste)

    Le surrogate soft remplace le binning dur : chaque échantillon contribue
    à chaque bin par exp(−(u−c_k)²/(2·bw²)), permettant au gradient de
    pousser σ et μ pour aplatir l'histogramme.

    Le 1/K facteur normalise pour que la loss soit comparable entre K
    différents (δ², comme dans flatness_metrics).
    """
    if q_obs.numel() == 0:
        return torch.zeros((), device=q_sim.device)
    if lam != 1.0:
        q_obs_t = box_cox(q_obs, lam)
        q_sim_t = box_cox(q_sim, lam)
    else:
        q_obs_t = q_obs
        q_sim_t = q_sim
    sigma = log_sigma.exp().clamp(min=1e-9)
    z = (q_obs_t - q_sim_t) / sigma
    # Φ(z) via erf : Φ(z) = 0.5·(1 + erf(z/√2))
    pit = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    # Soft histogram : (N, K) puis somme sur N
    centers = torch.linspace(
        0.5 / n_bins, 1.0 - 0.5 / n_bins, n_bins, device=pit.device, dtype=pit.dtype,
    )
    diff = pit.unsqueeze(-1) - centers.unsqueeze(0)  # (N, K)
    weights = torch.exp(-0.5 * (diff / bandwidth) ** 2)
    # Normalise par échantillon pour que chaque sample contribue ~1 au total
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
    soft_counts = weights.sum(dim=0)  # (K,)
    freq = soft_counts / (soft_counts.sum() + 1e-12)
    uniform = 1.0 / n_bins
    # δ² normalisé : équivalent à flatness_metrics().delta**2 sans la cste
    return ((freq - uniform) ** 2).mean() / (uniform ** 2)


def pinball_loss(y_obs: Tensor, q_pred: Tensor, tau: float) -> Tensor:
    """Pinball loss pour un seul quantile τ.

        L_τ(y, q̂) = max(τ·(y − q̂), (τ − 1)·(y − q̂))

    Sous-prédire (q̂ < y) est pénalisé en proportion τ ; sur-prédire en (1−τ).
    """
    if y_obs.numel() == 0:
        return torch.zeros((), device=q_pred.device)
    resid = y_obs - q_pred
    return torch.maximum(tau * resid, (tau - 1.0) * resid).mean()


def quantile_loss(y_obs: Tensor, q_pred: Tensor, taus: Tensor) -> Tensor:
    """Loss quantile multi-τ = moyenne des pinball sur K quantiles.

    Parameters
    ----------
    y_obs : Tensor, shape (...,) — observations en m³/s
    q_pred : Tensor, shape (..., K) — quantiles prédits (même unités que y)
    taus : Tensor, shape (K,) — niveaux de quantiles dans (0, 1)
    """
    if y_obs.numel() == 0:
        return torch.zeros((), device=q_pred.device)
    resid = y_obs.unsqueeze(-1) - q_pred                              # (..., K)
    taus_b = taus.to(q_pred.device).expand_as(resid)
    pinball = torch.maximum(taus_b * resid, (taus_b - 1.0) * resid)
    return pinball.mean()


def crps_from_quantiles(y_obs: Tensor, q_pred: Tensor, taus: Tensor) -> Tensor:
    """CRPS approximé depuis K quantiles (Gneiting & Ranjan 2011) :
    CRPS ≈ 2 · moyenne_τ(pinball_τ). Exact si τ uniforme dans (0,1) et K→∞."""
    return 2.0 * quantile_loss(y_obs, q_pred, taus)


def tws_anomaly_loss(
    storage_month: Tensor, grace_month: Tensor,
    sim_baseline: Tensor | float, grace_baseline: Tensor | float,
    sigma: Tensor | None = None,
) -> Tensor:
    """MSE (pondérée par l'incertitude) entre l'anomalie de stockage simulée et
    GRACE TWS. Centrage : on retire une baseline de chaque côté (GRACE et sim ont
    des références absolues différentes) → on compare les VARIATIONS.

        a_sim = storage_month − sim_baseline   (mm)
        a_obs = grace_month  − grace_baseline  (mm, anomalie GRACE)
        L = mean( (a_sim − a_obs)² [ / σ² ] )

    ``sim_baseline`` = moyenne long-terme du stockage simulé (mise à jour en
    running par époque, détachée). ``grace_baseline`` = moyenne GRACE sur les
    mois communs (constante précalculée).
    """
    if storage_month.numel() == 0:
        return torch.zeros((), device=storage_month.device)
    a_sim = storage_month - sim_baseline
    a_obs = grace_month - grace_baseline
    resid = a_sim - a_obs
    if sigma is not None:
        return (resid.pow(2) / (sigma ** 2 + 1.0)).mean()  # sigma: float ou tenseur
    return resid.pow(2).mean()


def peak_weighted_mse_loss(
    q_obs: Tensor, q_sim: Tensor, q_threshold: Tensor,
    station_var: Tensor | None = None,
) -> Tensor:
    """MSE restreinte aux pas de temps où Q_obs > seuil climato par station,
    normalisée par la variance Q par station pour échelle comparable au (1-NSE).

    Cible la pathologie « peak-shaving » diagnostiquée dans le PIT val 2019-2021
    (surplus u > 0.75) : donne du gradient explicite aux résidus en régime de
    pic, où le backbone calibré sur MSE+log_NSE rabote systématiquement.

    Parameters
    ----------
    q_obs : (T, S) débits observés
    q_sim : (T, S) débits simulés
    q_threshold : (S,) seuil par station (typiquement Q_p75 de la période d'entraînement)
    station_var : (S,) variance Q par station. Si fourni, normalise SE/var_s pour
                  que la magnitude finale soit O(0.1-1), comparable à (1-NSE).
                  Sans normalisation, la loss est en m³/s² brut et domine.

    Returns
    -------
    L_peak : MSE normalisée pooled sur les couples (t, s) tels que Q_obs[t,s] > q_threshold[s].
             Renvoie 0 (sans gradient) si aucun pic dans le chunk.

    Notes
    -----
    Chunk-safe : un chunk sans pic contribue 0, les chunks avec pics dominent.
    La pondération est binaire (seuil dur) — alternative continue : Q_obs^p.
    """
    if q_obs.numel() == 0:
        return torch.zeros((), device=q_sim.device)
    # Broadcast seuil (S,) sur (T, S)
    mask_peak = q_obs > q_threshold.unsqueeze(0)
    valid = mask_peak & ~torch.isnan(q_obs) & ~torch.isnan(q_sim)
    if not valid.any():
        return torch.zeros((), device=q_sim.device)
    # IMPORTANT : filtrer AVANT de calculer (q_obs - q_sim) ** 2. Sinon les
    # positions NaN produisent NaN dans le forward, et même si on les masque
    # avec [valid].mean(), le backward propage NaN × 0 = NaN à q_sim → tous
    # les grads zéroés en aval. Same pattern que gaussian_nll_loss.
    qo = q_obs[valid]
    qs = q_sim[valid]
    sq_err = (qo - qs) ** 2
    if station_var is not None:
        # Index station pour chaque (t,s) valide
        S = q_obs.shape[1]
        s_idx = torch.arange(S, device=q_obs.device).unsqueeze(0).expand_as(q_obs)
        var_at_valid = station_var[s_idx[valid]] + 1e-8
        sq_err = sq_err / var_at_valid
    return sq_err.mean()


def differentiable_mse_loss(
    q_obs: Tensor, q_sim: Tensor, var: Tensor | None = None
) -> Tensor:
    """Mean squared error, optionally normalized by precomputed variance.

    When ``var`` is provided, computes MSE/var — equivalent to 1-NSE but with
    a fixed denominator, so it remains additive across temporal chunks.
    """
    mse = ((q_obs - q_sim) ** 2).mean()
    if var is not None:
        return mse / (var + 1e-8)
    return mse


def timing_tolerant_mse(
    q_o: Tensor, q_s: Tensor, tol: int = 1, var: Tensor | None = None,
    w: Tensor | None = None,
) -> Tensor:
    """MSE tolérante au décalage temporel ±``tol`` jours, SYMÉTRIQUE, chunk-safe.

    Motivation (revue 2026-07-01) : la MSE point-à-point paie ~7× moins cher un
    pic APLATI au bon jour qu'un pic PARFAIT décalé d'un jour. Face à un lag
    résiduel (bruit convectif CaSR ±1j, irréductible), l'optimum de la loss est
    donc le lissage — le modèle re-lisse après avoir trouvé le bon régime. Cette
    perte cesse de punir un pic bien formé mais décalé de ≤ tol jours : l'erreur
    à t est le MIN de l'écart quadratique sur une fenêtre glissante ±tol.

    SYMÉTRIQUE pour rester honnête : terme (a) « le sim est-il expliqué par une
    obs voisine ? » + terme (b) « chaque obs est-elle expliquée par un sim
    voisin ? ». Sans (b), un sim plat près du baseline matcherait le creux
    pré-pic de l'obs dans la fenêtre et échapperait à la pénalité du pic manqué.

    q_o, q_s : (T, S), NaN autorisés (obs manquantes). Retourne une perte scalaire
    (moyenne par station sur le temps, pondérée par ``w`` si fourni).
    """
    T = q_o.shape[0]
    BIG = 1e12
    row = torch.arange(T, device=q_o.device).unsqueeze(1)  # (T,1) bords enroulés
    # ASSAINIR les NaN AVANT toute arithmétique : sinon (a - NaN)² = NaN, et même
    # masqué par torch.where, le backward fait 0×NaN = NaN (piège classique). On
    # remplace les NaN par 0 et on porte la validité dans des masques float.
    o_valid = (~torch.isnan(q_o)).to(q_o.dtype)
    s_valid = (~torch.isnan(q_s)).to(q_s.dtype)
    o_f = torch.nan_to_num(q_o, nan=0.0)
    s_f = torch.nan_to_num(q_s, nan=0.0)

    def windowed(a_f: Tensor, a_valid: Tensor, b_f: Tensor, b_valid: Tensor):
        # pour chaque t : min_d (a(t) - b(t+d))^2 sur d ∈ [-tol, tol], arithmétique
        # 100% finie (les candidats invalides sont poussés à BIG additivement).
        cands = []
        for d in range(-tol, tol + 1):
            bs = torch.roll(b_f, -d, dims=0)            # b(t+d) aligné sur t
            bv = torch.roll(b_valid, -d, dims=0)
            if d > 0:
                edge = (row >= (T - d)).to(a_f.dtype)
            elif d < 0:
                edge = (row < (-d)).to(a_f.dtype)
            else:
                edge = torch.zeros_like(a_f)
            usable = bv * (1.0 - edge)                  # 1 si candidat exploitable
            e = (a_f - bs) ** 2
            e = e * usable + BIG * (1.0 - usable)       # invalides -> BIG (fini)
            cands.append(e)
        m = torch.stack(cands, 0).amin(dim=0)           # (T,S)
        any_valid = (m < BIG * 0.5).to(a_f.dtype)
        used = a_valid * any_valid                       # compte : a valide ET voisin dispo
        return m * used, used                            # zéro (fini) là où inutilisé

    m_s, u_s = windowed(s_f, s_valid, o_f, o_valid)   # sim expliqué par obs voisines
    m_o, u_o = windowed(o_f, o_valid, s_f, s_valid)   # obs expliquée par sim voisins
    per_s = m_s.sum(dim=0) / (u_s.sum(dim=0) + 1e-8)  # moyenne masquée par station
    per_o = m_o.sum(dim=0) / (u_o.sum(dim=0) + 1e-8)
    per = 0.5 * (per_s + per_o)                        # (S,)
    if var is not None:
        per = per / (var + 1e-8)
    if w is not None:
        return (per * w).sum()
    return per.mean()


def differentiable_fdc_loss(q_obs: Tensor, q_sim: Tensor, quantiles: list[float] = None) -> Tensor:
    """Flow Duration Curve loss - matches flow quantiles, especially important for low flows.

    Computes MSE between observed and simulated flow quantiles.
    Default quantiles focus on low to medium flows for water shortage analysis.

    Args:
        q_obs: Observed discharge
        q_sim: Simulated discharge
        quantiles: List of quantiles to match (default: focus on low flows)

    Returns:
        FDC loss (lower is better)
    """
    if quantiles is None:
        # Focus on low to medium flows (Q95, Q90, Q75, Q50, Q25, Q10)
        quantiles = [0.95, 0.90, 0.75, 0.50, 0.25, 0.10]

    # Sort flows to get flow duration curves
    q_obs_sorted = torch.sort(q_obs, descending=True)[0]
    q_sim_sorted = torch.sort(q_sim, descending=True)[0]

    n = q_obs.shape[0]
    losses = []

    for q in quantiles:
        idx = int(q * n)
        idx = min(idx, n - 1)  # Ensure valid index

        # Get flow at this exceedance probability
        obs_q = q_obs_sorted[idx]
        sim_q = q_sim_sorted[idx]

        # Relative error weighted by 1/obs to emphasize low flows
        # Use log-space for better low-flow sensitivity
        weight = 1.0 / (obs_q + 1.0)  # +1 to avoid division by zero
        loss_q = weight * (torch.log(sim_q + 1.0) - torch.log(obs_q + 1.0)) ** 2
        losses.append(loss_q)

    return torch.stack(losses).mean()


def differentiable_nrmse_loss(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Normalized RMSE: RMSE / mean(obs). Dimensionless, lower is better."""
    mse = ((q_obs - q_sim) ** 2).mean()
    return torch.sqrt(mse + 1e-8) / (q_obs.mean() + 1e-8)


def differentiable_log_mse_loss(
    q_obs: Tensor, q_sim: Tensor, eps: float = 1.0
) -> Tensor:
    """MSE on log-transformed flows. Scale-invariant AND chunk-safe.

    Unlike log-NSE, this has no observation-variance denominator, so it remains
    additive across temporal chunks. The log transform naturally equalizes the
    gradient contribution of low-flow and high-flow periods.
    """
    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    return ((log_obs - log_sim) ** 2).mean()


def differentiable_log_nse_loss(
    q_obs: Tensor, q_sim: Tensor, eps: float = 0.01
) -> Tensor:
    """1 - NSE on log-transformed flows (emphasises low-flow periods).

    Using log(Q + eps) puts more weight on getting baseflow right,
    which improves KGE's variability ratio component.
    """
    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    num = ((log_obs - log_sim) ** 2).sum()
    denom = ((log_obs - log_obs.mean()) ** 2).sum()
    return num / (denom + 1e-8)


class CompositeKGELoss(nn.Module):
    """Composite KGE loss: alpha*(1-KGE) + (1-alpha)*(1-KGE_log).

    Per-station computation with optional station weights.
    Returns KGE components (r, beta, gamma) for diagnostic logging.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        eps: float = 1.0,
        per_station: bool = True,
        station_weights: Tensor | None = None,
        w_physics: float = 0.01,
        w_residual: float = 0.001,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.per_station = per_station
        self.w_physics = w_physics
        self.w_residual = w_residual
        if station_weights is not None:
            self.register_buffer("station_weights", station_weights)
        else:
            self.station_weights: Tensor | None = None

    def forward(
        self,
        q_obs: Tensor,
        q_sim: Tensor,
        station_mask: Tensor,
        swe_obs: Tensor | None = None,
        swe_sim: Tensor | None = None,
        et_obs: Tensor | None = None,
        et_sim: Tensor | None = None,
        water_balance_residual: Tensor | None = None,
        residual_gate_logits: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        q_sim_at_stations = q_sim[:, station_mask]  # (T, n_stations)
        dev = q_sim.device
        zero = torch.tensor(0.0, device=dev)

        if self.per_station:
            n_stations = q_sim_at_stations.shape[1]
            valid = ~torch.isnan(q_obs) & ~torch.isnan(q_sim_at_stations)
            valid_counts = valid.sum(dim=0)
            keep = valid_counts >= 30

            n_keep = keep.sum().item()
            if n_keep == 0:
                loss = zero
                components = {
                    "r": zero, "beta": zero, "gamma": zero, "kge": zero,
                    "r_log": zero, "beta_log": zero, "gamma_log": zero, "kge_log": zero,
                }
            else:
                if self.station_weights is not None and len(self.station_weights) == n_stations:
                    w = self.station_weights[keep]
                    w = w / w.sum()
                else:
                    w = torch.full((n_keep,), 1.0 / n_keep, device=dev)

                keep_idx = keep.nonzero(as_tuple=True)[0]
                losses, r_vals, beta_vals, gamma_vals, kge_vals = [], [], [], [], []
                r_log_vals, beta_log_vals, gamma_log_vals, kge_log_vals = [], [], [], []

                for si in keep_idx:
                    v = valid[:, si]
                    q_o_v = q_obs[v, si]
                    q_s_v = q_sim_at_stations[v, si]
                    l, info = differentiable_composite_kge_loss(
                        q_o_v, q_s_v, alpha=self.alpha, eps=self.eps,
                    )
                    losses.append(l)
                    r_vals.append(info["r"])
                    beta_vals.append(info["beta"])
                    gamma_vals.append(info["gamma"])
                    kge_vals.append(info["kge"])
                    r_log_vals.append(info["r_log"])
                    beta_log_vals.append(info["beta_log"])
                    gamma_log_vals.append(info["gamma_log"])
                    kge_log_vals.append(info["kge_log"])

                loss = (torch.stack(losses) * w).sum()
                components = {
                    "r": (torch.stack(r_vals) * w).sum(),
                    "beta": (torch.stack(beta_vals) * w).sum(),
                    "gamma": (torch.stack(gamma_vals) * w).sum(),
                    "kge": (torch.stack(kge_vals) * w).sum(),
                    "r_log": (torch.stack(r_log_vals) * w).sum(),
                    "beta_log": (torch.stack(beta_log_vals) * w).sum(),
                    "gamma_log": (torch.stack(gamma_log_vals) * w).sum(),
                    "kge_log": (torch.stack(kge_log_vals) * w).sum(),
                }
        else:
            q_o = q_obs.reshape(-1)
            q_s = q_sim_at_stations.reshape(-1)
            valid_mask = ~torch.isnan(q_o) & ~torch.isnan(q_s)
            q_o, q_s = q_o[valid_mask], q_s[valid_mask]
            loss, components = differentiable_composite_kge_loss(
                q_o, q_s, alpha=self.alpha, eps=self.eps,
            )

        # Regularization terms
        if self.w_physics > 0 and water_balance_residual is not None:
            valid_wb = ~torch.isnan(water_balance_residual)
            if valid_wb.any():
                L_phys = (water_balance_residual[valid_wb] ** 2).mean()
                loss = loss + self.w_physics * L_phys
                components["physics_loss"] = L_phys

        if self.w_residual > 0 and residual_gate_logits is not None:
            L_reg = (torch.sigmoid(residual_gate_logits) ** 2).mean()
            loss = loss + self.w_residual * L_reg
            components["residual_reg"] = L_reg

        components["total"] = loss
        return loss, components


class HydroLoss(nn.Module):
    """Multi-objective loss function.

    Parameters
    ----------
    w_nse, w_pbias, w_kge, w_mse, w_nrmse, w_log_nse : float
        Weights for the streamflow skill scores.
        w_mse is recommended for chunked training (additive across chunks).
    w_snow : float
        Weight for SWE reconstruction loss (0 if no SWE observations).
    w_et : float
        Weight for ET loss (0 if no flux-tower data).
    w_physics : float
        Weight for water balance closure penalty.
    w_residual : float
        L2 penalty on residual corrector gate values.
    per_station : bool
        If True, compute NSE/KGE/PBIAS per station and average (equal weight
        per station).  If False (default), pool all stations into one vector
        (dominated by the largest-flow station).
    """

    def __init__(
        self,
        w_nse: float = 1.0,
        w_pbias: float = 0.1,
        w_kge: float = 0.5,
        w_mse: float = 0.0,
        w_nrmse: float = 0.0,
        w_log_nse: float = 0.0,
        w_log_mse: float = 0.0,
        w_tol_mse: float = 0.0,
        tol_days: int = 1,
        w_nll: float = 0.0,
        w_nll_et: float = 0.0,
        w_nll_swe: float = 0.0,
        w_flatness: float = 0.0,
        w_snow: float = 0.0,
        w_et: float = 0.0,
        w_tws: float = 0.0,
        w_quantile: float = 0.0,
        w_mixture: float = 0.0,
        w_peak: float = 0.0,
        w_physics: float = 0.01,
        w_residual: float = 0.001,
        per_station: bool = False,
        station_weights: Tensor | None = None,
        station_var: Tensor | None = None,
        peak_threshold: Tensor | None = None,
        nll_lambda: float = 1.0,
        nll_distribution: str = "normal",   # "normal" | "box-cox" | "log-normal" | "student-t"
        flatness_n_bins: int = 21,
        flatness_bandwidth: float = 0.02,
    ) -> None:
        super().__init__()
        self.w_nse = w_nse
        self.w_pbias = w_pbias
        self.w_kge = w_kge
        self.w_mse = w_mse
        self.w_nrmse = w_nrmse
        self.w_log_nse = w_log_nse
        self.w_log_mse = w_log_mse
        self.w_tol_mse = w_tol_mse
        self.tol_days = int(tol_days)
        self.w_nll = w_nll
        self.w_nll_et = w_nll_et
        self.w_nll_swe = w_nll_swe
        self.w_flatness = w_flatness
        self.nll_lambda = nll_lambda
        self.nll_distribution = nll_distribution.lower()
        self.flatness_n_bins = flatness_n_bins
        self.flatness_bandwidth = flatness_bandwidth
        self.w_snow = w_snow
        self.w_et = w_et
        self.w_tws = w_tws  # GRACE TWS (calculé dans le trainer, lu via loss_fn.w_tws)
        self.w_quantile = w_quantile  # Régression quantile (calculée dans le trainer)
        self.w_mixture = w_mixture    # MDN (option 2b — calculée dans le trainer)
        self.w_peak = w_peak  # Pondération pics (seuil climato Q_p75 par station)
        self.w_physics = w_physics
        self.w_residual = w_residual
        self.per_station = per_station
        if station_weights is not None:
            self.register_buffer("station_weights", station_weights)
        else:
            self.station_weights: Tensor | None = None
        if station_var is not None:
            self.register_buffer("station_var", station_var)
        else:
            self.station_var: Tensor | None = None
        if peak_threshold is not None:
            self.register_buffer("peak_threshold", peak_threshold)
        else:
            self.peak_threshold: Tensor | None = None

    def forward(
        self,
        q_obs: Tensor,
        q_sim: Tensor,
        station_mask: Tensor,
        log_sigma_sim: Tensor | None = None,
        swe_obs: Tensor | None = None,
        swe_sim: Tensor | None = None,
        log_sigma_swe_sim: Tensor | None = None,
        et_obs: Tensor | None = None,
        et_sim: Tensor | None = None,
        log_sigma_et_sim: Tensor | None = None,
        water_balance_residual: Tensor | None = None,
        residual_gate_logits: Tensor | None = None,
        log_df: Tensor | None = None,   # ν Student-t (lu depuis noise_head.log_df par le trainer)
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Args:
            q_obs:  (n_timesteps, n_stations) observed streamflow
            q_sim:  (n_timesteps, n_nodes) simulated; masked via station_mask
            station_mask: (n_stations,) bool — which nodes have observations
            log_sigma_sim: (n_timesteps, n_nodes) log σ per timestep/node, used
                by the heteroscedastic Gaussian NLL term when ``w_nll > 0``.
            log_sigma_et_sim, log_sigma_swe_sim: idem pour ET et SWE — utilisés
                par les NLL multi-objectif (MOD16 ETR, NDSI snow) quand les
                weights ``w_nll_et`` / ``w_nll_swe`` sont > 0.
            ... optional auxiliary observations

        Returns:
            loss:       scalar total loss
            components: dict of named loss terms for logging
        """
        q_sim_at_stations = q_sim[:, station_mask]  # (T, n_stations)

        if self.per_station:
            n_stations = q_sim_at_stations.shape[1]
            dev = q_sim.device
            zero = torch.tensor(0.0, device=dev)

            # Validity mask: (T, S) — True where both obs and sim are valid
            valid = ~torch.isnan(q_obs) & ~torch.isnan(q_sim_at_stations)
            valid_counts = valid.sum(dim=0)  # (S,)
            keep = valid_counts >= 30  # stations with enough data
            n_keep = keep.sum().item()

            if n_keep == 0:
                logger.warning(
                    "No stations have >= 30 valid observations in this chunk "
                    "(total stations: %d, max valid count: %s). "
                    "Loss will be zero with no gradient.",
                    n_stations,
                    int(valid_counts.max().item()) if n_stations > 0 else "N/A",
                )
                L_nse = L_pbias = L_kge = L_mse = L_nrmse = L_log_nse = L_log_mse = zero
                L_tol_mse = zero
            else:
                # Masked obs/sim: set invalid to NaN for nanmean
                q_o = q_obs[:, keep].clone()                    # (T, S_keep)
                q_s = q_sim_at_stations[:, keep].clone()        # (T, S_keep)
                inv = ~valid[:, keep]
                q_o[inv] = float("nan")
                q_s[inv] = float("nan")

                # Weights for kept stations
                if self.station_weights is not None and len(self.station_weights) == n_stations:
                    w = self.station_weights[keep]
                    w = w / w.sum()
                else:
                    w = torch.full((n_keep,), 1.0 / n_keep, device=dev)

                # ── Vectorized MSE (chunk-safe) ──────────────────────────
                if self.w_mse > 0:
                    sq_err = (q_o - q_s) ** 2                   # (T, S_keep)
                    mse_per = torch.nanmean(sq_err, dim=0)      # (S_keep,)
                    if self.station_var is not None:
                        mse_per = mse_per / (self.station_var[keep] + 1e-8)
                    L_mse = (mse_per * w).sum()
                else:
                    L_mse = zero

                # ── Vectorized PBIAS ─────────────────────────────────────
                if self.w_pbias > 0:
                    diff_sum = torch.nansum(q_s - q_o, dim=0)   # (S_keep,)
                    obs_sum = torch.nansum(q_o, dim=0)           # (S_keep,)
                    pbias_per = (diff_sum / (obs_sum + 1e-8)).abs()
                    L_pbias = (pbias_per * w).sum()
                else:
                    L_pbias = zero

                # ── Vectorized log-MSE ───────────────────────────────────
                if self.w_log_mse > 0:
                    log_sq = (torch.log(q_o + 1.0) - torch.log(q_s.clamp(min=0.0) + 1.0)) ** 2
                    L_log_mse = (torch.nanmean(log_sq, dim=0) * w).sum()
                else:
                    L_log_mse = zero

                # ── MSE tolérante au timing ±tol_days (chunk-safe) ───────
                # Ne punit plus un pic bien formé décalé de ≤ tol jours (bruit
                # convectif CaSR irréductible) : casse le couplage lag→lissage
                # qui faisait re-lisser le modèle après le bon régime.
                if self.w_tol_mse > 0:
                    _var = self.station_var[keep] if self.station_var is not None else None
                    L_tol_mse = timing_tolerant_mse(q_o, q_s, tol=self.tol_days, var=_var, w=w)
                else:
                    L_tol_mse = zero

                # ── Per-station loop only for metrics that need it ───────
                # NSE, KGE, NRMSE, log-NSE require per-station variance
                # or correlation — only compute if weight > 0
                L_nse = L_kge = L_nrmse = L_log_nse = zero
                need_loop = (self.w_nse > 0 or self.w_kge > 0
                             or self.w_nrmse > 0 or self.w_log_nse > 0)
                if need_loop:
                    nse_v, kge_v, nrmse_v, lnse_v = [], [], [], []
                    keep_idx = keep.nonzero(as_tuple=True)[0]
                    for j, si in enumerate(keep_idx):
                        v = valid[:, si]
                        q_o_v = q_obs[v, si]
                        q_s_v = q_sim_at_stations[v, si]
                        if self.w_nse > 0:
                            nse_v.append(differentiable_nse_loss(q_o_v, q_s_v))
                        if self.w_kge > 0:
                            kge_v.append(differentiable_kge_loss(q_o_v, q_s_v))
                        if self.w_nrmse > 0:
                            nrmse_v.append(differentiable_nrmse_loss(q_o_v, q_s_v))
                        if self.w_log_nse > 0:
                            lnse_v.append(differentiable_log_nse_loss(q_o_v, q_s_v))
                    if self.w_nse > 0 and nse_v:
                        L_nse = (torch.stack(nse_v) * w).sum()
                    if self.w_kge > 0 and kge_v:
                        L_kge = (torch.stack(kge_v) * w).sum()
                    if self.w_nrmse > 0 and nrmse_v:
                        L_nrmse = (torch.stack(nrmse_v) * w).sum()
                    if self.w_log_nse > 0 and lnse_v:
                        L_log_nse = (torch.stack(lnse_v) * w).sum()
        else:
            # Pooled metrics: flatten time x station (dominated by largest station)
            q_o = q_obs.reshape(-1)
            q_s = q_sim_at_stations.reshape(-1)

            # Mask NaN in both obs and sim (NaN sim comes from missing forcing or
            # extreme parameter values early in training)
            valid = ~torch.isnan(q_o) & ~torch.isnan(q_s)
            q_o, q_s = q_o[valid], q_s[valid]

            L_nse = differentiable_nse_loss(q_o, q_s)
            L_pbias = differentiable_pbias_loss(q_o, q_s)
            L_kge = differentiable_kge_loss(q_o, q_s)
            L_mse = differentiable_mse_loss(q_o, q_s)
            L_nrmse = differentiable_nrmse_loss(q_o, q_s)
            _zero = torch.tensor(0.0, device=q_s.device)
            L_log_nse = differentiable_log_nse_loss(q_o, q_s) if self.w_log_nse > 0 else _zero
            L_log_mse = differentiable_log_mse_loss(q_o, q_s) if self.w_log_mse > 0 else _zero
            L_tol_mse = _zero   # tolérance timing : chemin per_station uniquement

        # Heteroscedastic Gaussian NLL (probabilistic loss replacing the
        # ensemble UQ stack). Aligns log_sigma to q_sim at station nodes.
        if self.w_nll > 0 and log_sigma_sim is not None:
            log_sigma_at_stations = log_sigma_sim[:, station_mask]
            ls_pool = log_sigma_at_stations.reshape(-1)
            qs_pool = q_sim_at_stations.reshape(-1)
            qo_pool = q_obs.reshape(-1)
            valid_pool = ~torch.isnan(qo_pool) & ~torch.isnan(qs_pool)
            if self.nll_distribution == "student-t" and log_df is not None:
                # Queues lourdes : ν = exp(log_df), apprenable depuis noise_head.
                L_nll = student_t_nll_loss(
                    qo_pool[valid_pool], qs_pool[valid_pool], ls_pool[valid_pool],
                    log_df, lam=self.nll_lambda,
                )
            else:
                L_nll = gaussian_nll_loss(
                    qo_pool[valid_pool], qs_pool[valid_pool], ls_pool[valid_pool],
                    lam=self.nll_lambda,
                )
        else:
            L_nll = torch.tensor(0.0, device=q_sim.device)

        # Flatness loss : pénalité directe sur la déviation du PIT histogram
        # par rapport à uniforme. Cible exactement la "vague" Talagrand,
        # contrairement au NLL qui peut accepter une σ gonflée.
        if self.w_flatness > 0 and log_sigma_sim is not None:
            log_sigma_at_stations = log_sigma_sim[:, station_mask]
            ls_pool_f = log_sigma_at_stations.reshape(-1)
            qs_pool_f = q_sim_at_stations.reshape(-1)
            qo_pool_f = q_obs.reshape(-1)
            valid_f = ~torch.isnan(qo_pool_f) & ~torch.isnan(qs_pool_f)
            L_flatness = flatness_loss(
                qo_pool_f[valid_f], qs_pool_f[valid_f], ls_pool_f[valid_f],
                lam=self.nll_lambda,
                n_bins=self.flatness_n_bins,
                bandwidth=self.flatness_bandwidth,
            )
        else:
            L_flatness = torch.tensor(0.0, device=q_sim.device)

        # Pondération pics (seuil climato Q_p75 par station) : MSE restreinte
        # aux pas de temps où Q_obs dépasse le seuil. Vise la queue haute
        # diagnostiquée comme sous-dispersée dans le PIT val.
        if self.w_peak > 0 and self.peak_threshold is not None:
            L_peak = peak_weighted_mse_loss(
                q_obs, q_sim_at_stations, self.peak_threshold,
                station_var=self.station_var,
            )
        else:
            L_peak = torch.tensor(0.0, device=q_sim.device)

        loss = (self.w_nse * L_nse + self.w_pbias * L_pbias
                + self.w_kge * L_kge + self.w_mse * L_mse
                + self.w_nrmse * L_nrmse
                + self.w_log_nse * L_log_nse
                + self.w_log_mse * L_log_mse
                + self.w_tol_mse * L_tol_mse
                + self.w_nll * L_nll
                + self.w_flatness * L_flatness
                + self.w_peak * L_peak)
        components = {"nse_loss": L_nse, "pbias_loss": L_pbias,
                      "kge_loss": L_kge, "mse_loss": L_mse,
                      "nrmse_loss": L_nrmse,
                      "log_nse_loss": L_log_nse,
                      "log_mse_loss": L_log_mse,
                      "tol_mse_loss": L_tol_mse,
                      "flatness_loss": L_flatness,
                      "nll_loss": L_nll,
                      "peak_loss": L_peak}

        if self.w_snow > 0 and swe_obs is not None and swe_sim is not None:
            valid = ~torch.isnan(swe_obs) & ~torch.isnan(swe_sim)
            if valid.any():
                L_snow = ((swe_obs[valid] - swe_sim[valid]) ** 2).mean()
                loss = loss + self.w_snow * L_snow
                components["snow_loss"] = L_snow

        if self.w_et > 0 and et_obs is not None and et_sim is not None:
            valid = ~torch.isnan(et_obs) & ~torch.isnan(et_sim)
            if valid.any():
                L_et = ((et_obs[valid] - et_sim[valid]) ** 2).mean()
                loss = loss + self.w_et * L_et
                components["et_loss"] = L_et

        # Heteroscedastic NLL on ET (vs MODIS MOD16 par ex.). Identifie K_c.
        if (self.w_nll_et > 0 and et_obs is not None and et_sim is not None
                and log_sigma_et_sim is not None):
            v = ~torch.isnan(et_obs) & ~torch.isnan(et_sim)
            if v.any():
                L_nll_et = gaussian_nll_loss(
                    et_obs[v], et_sim[v], log_sigma_et_sim[v],
                )
                loss = loss + self.w_nll_et * L_nll_et
                components["nll_et_loss"] = L_nll_et

        # Heteroscedastic NLL on SWE (vs MODIS NDSI / SNODAS). Identifie C_f.
        if (self.w_nll_swe > 0 and swe_obs is not None and swe_sim is not None
                and log_sigma_swe_sim is not None):
            v = ~torch.isnan(swe_obs) & ~torch.isnan(swe_sim)
            if v.any():
                L_nll_swe = gaussian_nll_loss(
                    swe_obs[v], swe_sim[v], log_sigma_swe_sim[v],
                )
                loss = loss + self.w_nll_swe * L_nll_swe
                components["nll_swe_loss"] = L_nll_swe

        if self.w_physics > 0 and water_balance_residual is not None:
            valid = ~torch.isnan(water_balance_residual)
            if valid.any():
                L_phys = (water_balance_residual[valid] ** 2).mean()
                loss = loss + self.w_physics * L_phys
                components["physics_loss"] = L_phys

        if self.w_residual > 0 and residual_gate_logits is not None:
            L_reg = (torch.sigmoid(residual_gate_logits) ** 2).mean()
            loss = loss + self.w_residual * L_reg
            components["residual_reg"] = L_reg

        components["total"] = loss
        return loss, components


# ---------------------------------------------------------------------------
# CRPS loss for ensemble calibration
# ---------------------------------------------------------------------------

def crps_loss(ensemble_Q: Tensor, observed_Q: Tensor) -> Tensor:
    """Continuous Ranked Probability Score — proper scoring rule for ensembles.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the ensemble distribution and y is
    the observation.  A perfectly calibrated ensemble minimises CRPS.

    Parameters
    ----------
    ensemble_Q : (n_members, n_timesteps, n_nodes) or (n_members, n_valid)
        Ensemble of simulated streamflow values.
    observed_Q : (n_timesteps, n_nodes) or (n_valid,)
        Observed streamflow (NaN values are automatically excluded).

    Returns
    -------
    crps : scalar mean CRPS across all valid (timestep, node) pairs.
    """
    n_members = ensemble_Q.shape[0]
    E = ensemble_Q.reshape(n_members, -1)   # (M, S)
    y = observed_Q.reshape(-1)               # (S,)

    # Mask NaN observations
    valid = ~torch.isnan(y)
    E, y = E[:, valid], y[valid]

    # E|X - y| averaged over members
    mae = (E - y.unsqueeze(0)).abs().mean(dim=0)    # (S,)

    # E|X - X'| via the energy-score Gini-mean difference identity
    E_sorted, _ = E.sort(dim=0)
    k = torch.arange(1, n_members + 1, dtype=E.dtype, device=E.device)
    spread = (
        (2.0 * k - n_members - 1).unsqueeze(-1) * E_sorted
    ).sum(dim=0) / (n_members * (n_members - 1) + 1e-8)

    return (mae - spread).mean()
