"""Le terme sur la vitesse de vidange identifie la partition entre les deux chemins lents.

La colonne a deux chemins lents, l'écoulement hypodermique et la nappe, et aucune observation
ne les séparait. Mesuré le 2026-09-21 : à volume lent constant, basculer tout ce volume d'un
chemin à l'autre déplace l'indice d'Eckhardt de six millièmes, parce que ce filtre ajuste sa
constante de récession et absorbe l'information. La même bascule déplace cette constante de
0,966 à 0,735. C'est donc la forme de la vidange qui porte la partition.
"""
import numpy as np
import pytest
import torch

from meandre.training.loss import HydroLoss, differentiable_recession_loss


def _reservoir(apport, tau):
    s, out = 0.0, np.empty(len(apport))
    for k, x in enumerate(apport):
        s += x
        q = s / tau
        s -= q
        out[k] = q
    return out


def _hydrogramme(part_fast, n=3000, seed=0):
    """Même volume lent, réparti entre un chemin de 3 jours et un de 45 jours."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    apport = np.clip((1.0 + 0.9 * np.sin(2 * np.pi * (t - 100) / 365.25))
                     * rng.gamma(0.6, 3.0, n), 0, None)
    surface = _reservoir(apport * 0.30, 1.5)
    slow = apport * 0.70
    q = surface + _reservoir(slow * part_fast, 3.0) + _reservoir(slow * (1 - part_fast), 45.0)
    return torch.tensor(q, dtype=torch.float32)


def test_truth_is_zero():
    q = _hydrogramme(0.5)
    assert float(differentiable_recession_loss(q, q)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("part", [0.0, 0.25, 0.75, 1.0])
def test_term_sees_partition_between_both_slow_paths(part):
    """Ce que l'indice d'écoulement de base d'Eckhardt ne voit pas."""
    truth = _hydrogramme(0.5)
    other = _hydrogramme(part)
    assert float(differentiable_recession_loss(truth, other)) > 0.02, (
        f"part {part} : le terme ne distingue pas cette partition de la verite")


def test_response_grows_with_partition_gap():
    truth = _hydrogramme(0.5)
    near = float(differentiable_recession_loss(truth, _hydrogramme(0.6)))
    far = float(differentiable_recession_loss(truth, _hydrogramme(1.0)))
    assert far > near


def test_term_responds_at_FIRST_order():
    """Un carré répondrait au second ordre et serait aveugle aux petits écarts."""
    truth = _hydrogramme(0.5)
    a = float(differentiable_recession_loss(truth, _hydrogramme(0.55)))
    b = float(differentiable_recession_loss(truth, _hydrogramme(0.60)))
    assert b / max(a, 1e-9) == pytest.approx(2.0, rel=0.35)


def test_mask_comes_from_observation():
    """Un modèle plat ne doit pas pouvoir déplacer ses propres jours de décrue."""
    truth = _hydrogramme(0.5)
    flat = torch.full_like(truth, float(truth.mean()))
    # Sans masque observe, une serie plate n'aurait aucun jour de decrue et le terme serait
    # nul ; avec le masque, elle est penalisee.
    assert float(differentiable_recession_loss(truth, flat)) > 0.05


def test_term_is_differentiable():
    truth = _hydrogramme(0.5)
    sim = _hydrogramme(0.8).clone().requires_grad_(True)
    loss = differentiable_recession_loss(truth, sim)
    loss.backward()
    assert sim.grad is not None and torch.isfinite(sim.grad).all()
    assert float(sim.grad.abs().sum()) > 0.0


def test_weight_is_zero_by_default():
    assert HydroLoss(per_station=True).w_recession == 0.0


def test_enters_the_weighted_sum():
    q = _hydrogramme(0.5).unsqueeze(1)
    sim = _hydrogramme(1.0).unsqueeze(1)
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0, w_log_nse=0.0,
                w_log_mse=0.0, w_dq=0.0, w_fdc_low=0.0, w_dq_log=0.0, w_peak=0.0,
                per_station=True)
    mk = torch.ones(1, dtype=torch.bool)
    without = HydroLoss(**base)(q_obs=q, q_sim=sim, station_mask=mk)
    avec = HydroLoss(**{**base, "w_recession": 1.0})(q_obs=q, q_sim=sim, station_mask=mk)
    without = without[0] if isinstance(without, tuple) else without
    avec = avec[0] if isinstance(avec, tuple) else avec
    assert float(avec) > float(without)
