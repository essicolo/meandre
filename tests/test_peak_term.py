"""Le terme de pics ne doit pas payer le modèle pour aplatir.

Mesuré le 2026-09-20 sur 32 stations : le terme de pics en vigueur, un écart quadratique
restreint aux hauts débits, PRÉFÈRE un hydrogramme lissé sur sept jours à un hydrogramme net
décalé d'un jour, sur 78 % des stations. C'est le terme ajouté pour protéger les pics qui les
détruit, parce qu'il regarde l'écart jour par jour là où un décalage coûte le plus cher.

Le terme par RAPPORT des magnitudes ne peut pas être dupé : lisser abaisse la pointe simulée.
Ce fichier épingle la propriété, qui est la raison d'être du terme.
"""
import numpy as np
import pytest
import torch

from meandre.training.loss import differentiable_peak_ratio_loss


def _series(n=1500, graine=5):
    g = torch.Generator().manual_seed(graine)
    t = torch.arange(n, dtype=torch.float32)
    base = 10.0 + 6.0 * torch.sin(t / 58.0) + 3.0 * torch.sin(t / 11.0)
    return torch.clamp(base + torch.rand(n, generator=g) * 2.0, min=0.5)


def _shifted(q, k=1):
    return torch.cat([q[:k], q[:-k]])


def _smoothed(q, n):
    pad = torch.nn.functional.pad(q.view(1, 1, -1), (n, n), mode="replicate")
    return torch.nn.functional.avg_pool1d(pad, n, 1).view(-1)[:len(q)]


@pytest.mark.parametrize("fenetre", [7, 15, 30])
@pytest.mark.parametrize("quantile", [0.75, 0.90, 0.95])
def test_smoothing_always_costs_more_than_lag(fenetre, quantile):
    """Le choix réel du modèle : rester net et en retard, ou lisser."""
    q = _series()
    seuil = torch.quantile(q, quantile)
    tard = _shifted(q)
    liss = _smoothed(tard, fenetre)
    perte_tard = float(differentiable_peak_ratio_loss(q, tard, seuil))
    perte_liss = float(differentiable_peak_ratio_loss(q, liss, seuil))
    assert perte_liss > perte_tard, (
        f"lissage {fenetre} j au seuil {quantile} : le terme prefere le lissage, "
        f"{perte_liss:.5f} contre {perte_tard:.5f}")


def test_truth_is_zero():
    q = _series()
    assert float(differentiable_peak_ratio_loss(q, q, torch.quantile(q, 0.75))) == pytest.approx(0.0, abs=1e-9)


def test_penalty_is_symmetric_in_log():
    """Une pointe deux fois trop forte coûte autant qu'une pointe deux fois trop faible."""
    q = _series()
    seuil = torch.quantile(q, 0.75)
    forte = float(differentiable_peak_ratio_loss(q, q * 2.0, seuil))
    faible = float(differentiable_peak_ratio_loss(q, q * 0.5, seuil))
    assert forte == pytest.approx(faible, rel=1e-5)


def test_term_is_differentiable():
    q = _series()
    sim = (q * 0.8).clone().requires_grad_(True)
    perte = differentiable_peak_ratio_loss(q, sim, torch.quantile(q, 0.75))
    perte.backward()
    assert sim.grad is not None and torch.isfinite(sim.grad).all()
    # Le gradient ne porte QUE sur les jours au-dessus du seuil observe.
    assert torch.all(sim.grad[q < torch.quantile(q, 0.75)] == 0.0)


def test_too_few_peaks_returns_zero_without_crashing():
    q = _series(n=20)
    assert float(differentiable_peak_ratio_loss(q, q * 0.5, torch.quantile(q, 0.99))) == 0.0


def test_shape_terms_respond_at_FIRST_order():
    """Un carré répondrait au second ordre et rendrait le terme aveugle aux petites erreurs.

    Défaut trouvé trois fois le 2026-09-20 et le 2026-09-21 : sur les facteurs du KGE, sur le
    terme d'étiage, et sur ce terme-ci que j'avais écrit moi-même. Pour une erreur relative e
    petite, une forme du premier ordre varie comme e et une forme du second comme e carré. Le
    test le vérifie en comparant deux tailles d'erreur : si le rapport des pertes suit le
    rapport des erreurs, la forme est du premier ordre.
    """
    q = _series()
    seuil = torch.quantile(q, 0.75)
    haut = q >= seuil
    petite, grande = 1.02, 1.04
    a = q.clone()
    a[haut] = a[haut] * petite
    b = q.clone()
    b[haut] = b[haut] * grande
    pa = float(differentiable_peak_ratio_loss(q, a, seuil))
    pb = float(differentiable_peak_ratio_loss(q, b, seuil))
    # Erreur doublee : une forme du premier ordre double, une du second quadruple.
    assert pb / pa == pytest.approx(2.0, rel=0.05), (
        f"rapport {pb / pa:.2f} : la forme n'est pas du premier ordre")
