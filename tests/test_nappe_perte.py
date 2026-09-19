"""Terme de perte sur les niveaux de nappe mesurés.

Le terme vise la FORME de la variation et non son amplitude, parce que le balayage du
2026-09-18 montre que la mesure identifie la phase et la présence des mécanismes, non la
valeur des paramètres. Ces tests fixent ce contrat : insensibilité à l'échelle, valeurs
connues aux trois cas de corrélation, écart des puits trop courts, et garde-fou physique.
"""
import torch

from meandre.training.loss import nappe_anomaly_loss


def _series(graine=0, mois=120, puits=8):
    g = torch.Generator().manual_seed(graine)
    return torch.randn(mois, puits, generator=g, dtype=torch.float64)


def test_valeurs_aux_trois_correlations():
    z = _series()
    assert float(nappe_anomaly_loss(z, z)) < 1e-10
    assert abs(float(nappe_anomaly_loss(z, -z)) - 4.0) < 0.2
    assert abs(float(nappe_anomaly_loss(z, _series(1))) - 2.0) < 0.2


def test_insensible_a_l_echelle_et_au_decalage():
    """Multiplier une série par un facteur ou la décaler ne change pas la perte."""
    z = _series()
    o = _series(2)
    base = float(nappe_anomaly_loss(z, o))
    assert abs(float(nappe_anomaly_loss(z * 7.0 + 13.0, o)) - base) < 1e-9
    assert abs(float(nappe_anomaly_loss(z, o * 0.1 - 5.0)) - base) < 1e-9


def test_puits_trop_court_ecarte():
    """Moins de 24 mois observés : le puits ne compte pas, et zéro puits donne zéro."""
    z = _series(mois=120, puits=2)
    m = torch.zeros_like(z, dtype=torch.bool)
    m[:10] = True
    assert float(nappe_anomaly_loss(z, -z, masque=m)) == 0.0
    m[:30] = True
    assert float(nappe_anomaly_loss(z, -z, masque=m)) > 1.0


def test_vide_rend_zero():
    v = torch.zeros(0, 0)
    assert float(nappe_anomaly_loss(v, v)) == 0.0


def test_garde_fou_porosite_nul_dans_la_plage():
    """Le garde-fou ne coûte rien tant que la porosité impliquée est physique."""
    z = _series()
    amp_s = torch.full((z.shape[1],), 2.0, dtype=torch.float64)
    amp_o = torch.full((z.shape[1],), 1.0, dtype=torch.float64)
    base = float(nappe_anomaly_loss(z, _series(3)))
    dedans = float(nappe_anomaly_loss(z, _series(3), sy=0.05, amplitude_sim=amp_s,
                                      amplitude_obs=amp_o, poids_porosite=1.0))
    dehors = float(nappe_anomaly_loss(z, _series(3), sy=0.05, amplitude_sim=amp_s * 20.0,
                                      amplitude_obs=amp_o, poids_porosite=1.0))
    assert abs(dedans - base) < 1e-9
    assert dehors > base + 0.1


def test_gradient_remonte_vers_la_simulation():
    z = _series().requires_grad_(True)
    nappe_anomaly_loss(z, _series(4)).backward()
    assert torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0.0
