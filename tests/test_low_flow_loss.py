"""Le terme d'étiage direct vaut zéro quand la simulation colle, et le logarithme de l'erreur relative sur les seuls jours d'étiage observés."""
import math

import torch

from meandre.training.loss import HydroLoss, differentiable_low_flow_loss


def test_zero_if_identical_else_relative_error():
    torch.manual_seed(0)
    qo = torch.rand(365) * 10.0 + 0.5
    assert float(differentiable_low_flow_loss(qo, qo.clone())) == 0.0
    assert abs(float(differentiable_low_flow_loss(qo, qo * 1.5)) - math.log(1.5)) < 1e-3


def test_gradient_only_on_low_flow_days():
    torch.manual_seed(1)
    qo = torch.rand(300) * 10.0 + 0.5
    qs = (qo * 1.3).requires_grad_(True)
    differentiable_low_flow_loss(qo, qs).backward()
    bas = qo <= torch.quantile(qo, 0.3)
    assert bool((qs.grad[~bas] == 0).all()) and bool((qs.grad[bas] != 0).all())


def test_weight_is_set_on_the_loss():
    assert HydroLoss(w_low_flow=0.7).w_low_flow == 0.7
