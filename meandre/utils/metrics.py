"""Evaluation metrics (non-differentiable, for reporting only).

Use meandre.training.loss for differentiable versions used during training.
"""

import torch
from torch import Tensor


def nse(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Nash-Sutcliffe Efficiency. Perfect = 1.0, no-skill = 0.0."""
    num = ((q_obs - q_sim) ** 2).sum()
    denom = ((q_obs - q_obs.mean()) ** 2).sum()
    return 1.0 - num / (denom + 1e-8)


def pbias(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Percent Bias (%). Perfect = 0, positive = overestimate."""
    return 100.0 * (q_sim - q_obs).sum() / (q_obs.sum() + 1e-8)


def kge(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Kling-Gupta Efficiency. Perfect = 1.0."""
    r = torch.corrcoef(torch.stack([q_obs, q_sim]))[0, 1]
    r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    mu_obs, mu_sim = q_obs.mean(), q_sim.mean()
    sig_obs, sig_sim = q_obs.std(), q_sim.std()
    beta = mu_sim / (mu_obs + 1e-8)
    gamma = (sig_sim / (sig_obs + 1e-8)) / (mu_sim / (mu_obs + 1e-8) + 1e-8)
    sq = (r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2
    return 1.0 - torch.sqrt(sq + 1e-8)



def rmse(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Root Mean Squared Error (same units as input)."""
    return torch.sqrt(((q_obs - q_sim) ** 2).mean())


def nrmse(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Normalized RMSE: RMSE / mean(obs). Dimensionless, lower is better."""
    return rmse(q_obs, q_sim) / (q_obs.mean() + 1e-8)


def mae(q_obs: Tensor, q_sim: Tensor) -> Tensor:
    """Mean Absolute Error (same units as input)."""
    return (q_obs - q_sim).abs().mean()


def log_nse(q_obs: Tensor, q_sim: Tensor, eps: float = 0.01) -> Tensor:
    """NSE on log-transformed flows (emphasises low-flow periods)."""
    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    return nse(log_obs, log_sim)


def kge_components(q_obs: Tensor, q_sim: Tensor) -> dict[str, Tensor]:
    """KGE decomposed into r, beta, gamma. For diagnostic reporting."""
    r = torch.corrcoef(torch.stack([q_obs, q_sim]))[0, 1]
    r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    mu_obs, mu_sim = q_obs.mean(), q_sim.mean()
    sig_obs, sig_sim = q_obs.std(), q_sim.std()
    beta = mu_sim / (mu_obs + 1e-8)
    gamma = (sig_sim / (sig_obs + 1e-8)) / (mu_sim / (mu_obs + 1e-8) + 1e-8)
    sq = (r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2
    kge_val = 1.0 - torch.sqrt(sq + 1e-8)

    # Log-space KGE
    eps = 1.0
    log_obs = torch.log(q_obs + eps)
    log_sim = torch.log(q_sim.clamp(min=0.0) + eps)
    r_log = torch.corrcoef(torch.stack([log_obs, log_sim]))[0, 1]
    r_log = torch.nan_to_num(r_log, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    mu_lo, mu_ls = log_obs.mean(), log_sim.mean()
    sig_lo, sig_ls = log_obs.std(), log_sim.std()
    beta_log = mu_ls / (mu_lo + 1e-8)
    gamma_log = (sig_ls / (sig_lo + 1e-8)) / (mu_ls / (mu_lo + 1e-8) + 1e-8)
    sq_log = (r_log - 1) ** 2 + (beta_log - 1) ** 2 + (gamma_log - 1) ** 2
    kge_log = 1.0 - torch.sqrt(sq_log + 1e-8)

    return {
        "r": r, "beta": beta, "gamma": gamma, "kge": kge_val,
        "r_log": r_log, "beta_log": beta_log, "gamma_log": gamma_log, "kge_log": kge_log,
    }
