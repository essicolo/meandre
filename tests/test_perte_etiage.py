"""Le terme d'étiage direct vaut zéro quand la simulation colle, et le logarithme de l'erreur relative sur les seuls jours d'étiage observés."""
import math

import torch

from meandre.training.loss import HydroLoss, differentiable_etiage_loss


def test_zero_si_identique_et_erreur_relative_sinon():
    torch.manual_seed(0)
    qo = torch.rand(365) * 10.0 + 0.5
    assert float(differentiable_etiage_loss(qo, qo.clone())) == 0.0
    assert abs(float(differentiable_etiage_loss(qo, qo * 1.5)) - math.log(1.5)) < 1e-3


def test_le_gradient_ne_porte_que_sur_les_jours_d_etiage():
    torch.manual_seed(1)
    qo = torch.rand(300) * 10.0 + 0.5
    qs = (qo * 1.3).requires_grad_(True)
    differentiable_etiage_loss(qo, qs).backward()
    bas = qo <= torch.quantile(qo, 0.3)
    assert bool((qs.grad[~bas] == 0).all()) and bool((qs.grad[bas] != 0).all())


def test_le_poids_est_pose_sur_la_perte():
    assert HydroLoss(w_etiage=0.7).w_etiage == 0.7
