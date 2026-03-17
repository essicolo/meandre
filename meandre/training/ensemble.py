"""Full ensemble generator and Sobol-style variance decomposition.

Ensemble grid structure
-----------------------
    Meteo member  (m)  x  Dropout config (d)  x  Noise trajectory (s)

This mirrors the traditional hydrology ensemble framework:
    - Meteo members      = perturbed GEPS/ECCC NWP forecasts (same as Hydrotel)
    - Dropout configs    = different "model configurations" (like Hydrotel vs Raven)
    - Noise trajectories = aleatoric uncertainty (not available in classical models)

All trajectories are temporally coherent because:
    - Dropout masks are frozen for the entire trajectory (via frozen_dropout)
    - State noise is AR(1) correlated (via CorrelatedStateNoise)
    - Both perturbations pass through the full physics stack
"""

from __future__ import annotations

import torch
from torch import Tensor

from meandre.training.uncertainty import frozen_dropout


def generate_full_ensemble(
    model,
    forcing_members: list[Tensor],
    initial_state,
    n_dropout_members: int = 5,
    n_noise_trajectories: int = 3,
    **simulate_kwargs,
) -> dict:
    """Generate the complete ensemble grid.

    Parameters
    ----------
    model : YHydro
        Trained model with optional ``CorrelatedStateNoise`` module.
    forcing_members : list of Tensor, each (n_timesteps, n_nodes, n_forcing)
        One tensor per meteorological ensemble member.
    initial_state : HydroState
        Common initial state for all members.
    n_dropout_members : int
        Number of frozen-dropout configurations to sample.
    n_noise_trajectories : int
        Number of AR(1) state-noise trajectories per dropout configuration.
        Only meaningful when model has a ``state_noise`` module.
    **simulate_kwargs
        Extra keyword arguments forwarded to ``model.simulate()``.

    Returns
    -------
    dict with keys:
        ``ensemble``         : (n_meteo, n_dropout, n_noise, n_timesteps, n_nodes)
        ``shape_description``: human-readable axis labels
        ``n_total_members``  : total number of simulation runs
        ``dimensions``       : dict of individual axis sizes
    """
    n_meteo = len(forcing_members)
    all_Q: list[Tensor] = []

    for m, forcing in enumerate(forcing_members):
        dropout_Q: list[Tensor] = []
        for d in range(n_dropout_members):
            noise_Q: list[Tensor] = []
            for s in range(n_noise_trajectories):
                # Use a unique but reproducible seed for each (m, d, s) triplet
                noise_seed = m * 10_000 + d * 100 + s
                torch.manual_seed(noise_seed)

                with frozen_dropout(model, seed=d):
                    with torch.no_grad():
                        Q, _ = model.simulate(
                            forcing,
                            initial_state,
                            inject_noise=True,
                            **simulate_kwargs,
                        )
                noise_Q.append(Q.detach())

            dropout_Q.append(torch.stack(noise_Q))   # (n_noise, T, N)
        all_Q.append(torch.stack(dropout_Q))          # (n_dropout, n_noise, T, N)

    ensemble = torch.stack(all_Q)  # (n_meteo, n_dropout, n_noise, T, N)

    return {
        "ensemble": ensemble,
        "shape_description": "(n_meteo, n_dropout, n_noise, n_timesteps, n_nodes)",
        "n_total_members": n_meteo * n_dropout_members * n_noise_trajectories,
        "dimensions": {
            "meteo": n_meteo,
            "dropout": n_dropout_members,
            "noise": n_noise_trajectories,
        },
    }


def variance_decomposition(ensemble: Tensor) -> dict[str, Tensor]:
    """Decompose total ensemble variance into meteorological, parametric, and
    aleatoric components.

    Uses an ANOVA-style decomposition:
        var_total  = var_meteo + var_parametric + var_aleatoric  (approx)

    Parameters
    ----------
    ensemble : (n_meteo, n_dropout, n_noise, n_timesteps, n_nodes)

    Returns
    -------
    dict with per-(timestep, node) variances and fractions for each source.
    Typically:
        meteo       dominates for flood forecasting horizons
        parametric  dominates for counterfactual / naturalization analysis
        aleatoric   dominates during low-flow periods
    """
    # Total variance across all members
    flat = ensemble.reshape(-1, *ensemble.shape[-2:])  # (M*D*S, T, N)
    var_total = flat.var(dim=0)                         # (T, N)

    # Meteo variance: variance of the (dropout x noise)-means across meteo members
    meteo_mean = ensemble.mean(dim=(1, 2))             # (n_meteo, T, N)
    var_meteo = meteo_mean.var(dim=0)                  # (T, N)

    # Parametric (dropout) variance: average over meteo of dropout-means variance
    dropout_mean = ensemble.mean(dim=2)                # (n_meteo, n_dropout, T, N)
    var_dropout = dropout_mean.var(dim=1).mean(dim=0)  # (T, N)

    # Aleatoric: residual after meteo and parametric are accounted for
    var_aleatoric = (var_total - var_meteo - var_dropout).clamp(min=0.0)

    eps = 1e-10
    return {
        "total": var_total,
        "meteo": var_meteo,
        "parametric": var_dropout,
        "aleatoric": var_aleatoric,
        "fraction_meteo": var_meteo / (var_total + eps),
        "fraction_parametric": var_dropout / (var_total + eps),
        "fraction_aleatoric": var_aleatoric / (var_total + eps),
    }
