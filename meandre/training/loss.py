"""Multi-objective differentiable loss for hydrological calibration.

All terms are smooth functions of Q_sim so gradients flow back through the
full model. Use meandre.utils.metrics for non-differentiable evaluation.

L = w1*(1-NSE) + w2*|PBIAS|/100 + w3*(1-KGE)
  + w4*L_snow  + w5*L_ET
  + w6*L_physics  + w7*L_residual_reg
"""

import torch
import torch.nn as nn
from torch import Tensor


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
        w_snow: float = 0.0,
        w_et: float = 0.0,
        w_physics: float = 0.01,
        w_residual: float = 0.001,
        per_station: bool = False,
        station_weights: Tensor | None = None,
        station_var: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.w_nse = w_nse
        self.w_pbias = w_pbias
        self.w_kge = w_kge
        self.w_mse = w_mse
        self.w_nrmse = w_nrmse
        self.w_log_nse = w_log_nse
        self.w_log_mse = w_log_mse
        self.w_snow = w_snow
        self.w_et = w_et
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
        """
        Args:
            q_obs:  (n_timesteps, n_stations) observed streamflow
            q_sim:  (n_timesteps, n_nodes) simulated; masked via station_mask
            station_mask: (n_stations,) bool — which nodes have observations
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
                L_nse = L_pbias = L_kge = L_mse = L_nrmse = L_log_nse = L_log_mse = zero
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

        loss = (self.w_nse * L_nse + self.w_pbias * L_pbias
                + self.w_kge * L_kge + self.w_mse * L_mse
                + self.w_nrmse * L_nrmse
                + self.w_log_nse * L_log_nse
                + self.w_log_mse * L_log_mse)
        components = {"nse_loss": L_nse, "pbias_loss": L_pbias,
                      "kge_loss": L_kge, "mse_loss": L_mse,
                      "nrmse_loss": L_nrmse,
                      "log_nse_loss": L_log_nse,
                      "log_mse_loss": L_log_mse}

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
