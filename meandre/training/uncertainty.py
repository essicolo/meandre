"""Uncertainty quantification via frozen MC Dropout and ensemble methods.

Design principle
----------------
Stochasticity must be injected UPSTREAM of the temporal dynamics, not at the
output.  When noise passes through the physics (soil balance, routing, snow),
it is filtered by the watershed's inertia — exactly as in a real catchment.
Independent per-timestep sampling at the output produces jagged,
non-physical trajectories that violate the autocorrelation structure of
observed streamflow.

Two sources of parametric uncertainty are implemented here:

frozen_dropout
    Fixes all dropout masks for one entire simulation trajectory.  Different
    seeds give different "model configurations", analogous to different
    Hydrotel/Raven parameter sets.

ensemble_predict
    Average over independently-trained models (classical deep ensemble).
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Frozen dropout context manager
# ---------------------------------------------------------------------------

@contextmanager
def frozen_dropout(
    model: nn.Module,
    seed: int,
) -> Generator[nn.Module, None, None]:
    """Fix all ``nn.Dropout`` masks for the duration of a ``with`` block.

    Usage::

        with frozen_dropout(model, seed=42):
            Q, _ = model.simulate(forcing, state, ...)
        # Q is a temporally coherent trajectory.

        with frozen_dropout(model, seed=99):
            Q2, _ = model.simulate(forcing, state, ...)
        # Q2 uses a different fixed configuration.

    Different seeds correspond to different "model configurations" (epistemic
    uncertainty), analogous to different hydrological model parameter sets.

    Parameters
    ----------
    model : nn.Module
        Model with ``nn.Dropout`` layers (e.g. ``SpatialFieldNetwork``).
    seed : int
        Random seed that determines the dropout mask.  Same seed → identical
        trajectory; different seeds → different trajectories.

    Yields
    ------
    model
        The same model in training mode with frozen dropout masks.
    """
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    originals: dict[str, types.MethodType] = {}
    mask_cache: dict[tuple, Tensor] = {}

    from meandre.spatial.concrete_dropout import ConcreteDropout

    for name, module in model.named_modules():
        # Support both standard nn.Dropout and ConcreteDropout
        if isinstance(module, ConcreteDropout):
            originals[name] = module.forward
            drop_p = module.p.item()
        elif isinstance(module, nn.Dropout) and module.p > 0.0:
            originals[name] = module.forward
            drop_p = module.p
        else:
            continue

        def _make_forward(mod_name: str, drop_p: float):
            def _frozen_forward(x: Tensor) -> Tensor:
                key = (mod_name, tuple(x.shape))
                if key not in mask_cache:
                    # Bernoulli draw scaled for inverted dropout
                    mask_cache[key] = torch.bernoulli(
                        torch.full(x.shape, 1.0 - drop_p, device=x.device),
                        generator=generator,
                    ) / (1.0 - drop_p + 1e-8)
                return x * mask_cache[key]
            return _frozen_forward

        module.forward = _make_forward(name, drop_p)  # type: ignore[method-assign]

    model.train()   # Dropout layers must be in training mode to activate
    try:
        yield model
    finally:
        for name, module in model.named_modules():
            if name in originals:
                module.forward = originals[name]  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Frozen ParamNoise — Position B (recommended)
# ---------------------------------------------------------------------------

@contextmanager
def frozen_param_noise(
    model: nn.Module,
    seed: int,
) -> Generator[nn.Module, None, None]:
    """Freeze a single ParamNoise realisation across an entire trajectory.

    Each ensemble member draws ONE ε ~ N(0, I) at entry, then uses the
    SAME ε for every simulate() call inside the with-block.  This produces
    a temporally-coherent member analogous to running the physics with one
    specific (deterministic) parameter set.

    Mass conservation is preserved because the noise is injected on raw
    fc_out logits and then `_apply_constraints` maps them into physical
    bounds before the vertical column sees them.

    Usage::

        with frozen_param_noise(model, seed=k):
            Q_k, _ = model.simulate(forcing, initial_state, ..., perturb_params=True)

    Different seeds → different members.

    Notes
    -----
    * The model must have been constructed with ``param_noise=True`` and
      have a ``spatial_encoder.param_log_sigma`` Parameter.  If absent,
      this context manager is a no-op.
    """
    spatial = getattr(model, "spatial_encoder", None)
    if spatial is None or not getattr(spatial, "param_noise", False):
        yield model
        return

    device = next(model.parameters()).device
    n_params = spatial.param_log_sigma.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    eps = torch.randn(n_params, device=device, generator=gen)

    # Monkey-patch forward to lock in this ε
    original_forward = spatial.forward

    def frozen_forward(coords, territorial, perturb_params=False, param_noise_eps=None):
        # Force perturb_params=True with our frozen ε regardless of caller
        return original_forward(
            coords, territorial,
            perturb_params=True,
            param_noise_eps=eps,
        )

    spatial.forward = frozen_forward  # type: ignore[method-assign]
    try:
        yield model
    finally:
        spatial.forward = original_forward  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# MC Dropout ensemble generator
# ---------------------------------------------------------------------------

def generate_ensemble_mc(
    model: nn.Module,
    forcing: Tensor,
    initial_state,
    n_members: int = 20,
    **simulate_kwargs,
) -> Tensor:
    """Generate an ensemble via frozen MC Dropout.

    Each member uses a different frozen dropout mask = a different model
    configuration.  All members receive the same forcing.

    Parameters
    ----------
    model : nn.Module
        Trained HydroModel model with dropout layers.
    forcing : Tensor
        (n_timesteps, n_nodes, n_forcing)
    initial_state : HydroState
    n_members : int
        Number of ensemble members.
    **simulate_kwargs
        Forwarded to ``model.simulate()``.

    Returns
    -------
    ensemble : (n_members, n_timesteps, n_nodes)
    """
    trajectories: list[Tensor] = []
    for i in range(n_members):
        with frozen_dropout(model, seed=i):
            with torch.no_grad():
                Q, _ = model.simulate(forcing, initial_state, **simulate_kwargs)
        trajectories.append(Q.detach())
    return torch.stack(trajectories)


# ---------------------------------------------------------------------------
# Classical deep ensemble
# ---------------------------------------------------------------------------

def ensemble_predict(
    models: list[nn.Module],
    forcing: Tensor,
    initial_state,
    **simulate_kwargs,
) -> tuple[Tensor, Tensor]:
    """Average predictions across an ensemble of independently-trained models.

    Parameters
    ----------
    models : list of trained HydroModel instances.
    forcing, initial_state : passed to each ``model.simulate()``.

    Returns
    -------
    Q_mean, Q_std : (n_timesteps, n_nodes)
    """
    samples: list[Tensor] = []
    for model in models:
        model.eval()
        with torch.no_grad():
            Q_sim, _ = model.simulate(forcing, initial_state, **simulate_kwargs)
        samples.append(Q_sim)
    stacked = torch.stack(samples, dim=0)  # (K, T, N)
    return stacked.mean(dim=0), stacked.std(dim=0)
