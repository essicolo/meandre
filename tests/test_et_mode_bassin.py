"""Le terme d'évapotranspiration peut porter sur la moyenne du territoire.

Motif mesuré le 2026-09-22 : sur 3150 nœuds au pas de huit jours, MOD16 porte UNE direction
indépendante et un rapport de participation de 1,1, soit une seule courbe saisonnière. Et son
apport propre, mesuré comme le rang de l'ensemble des observations moins celui de l'ensemble
privé d'elle, vaut zéro. En faire une contrainte par tronçon est donc une prétention, et elle
coûte 3150 séries en mémoire.
"""
import pytest
import torch

from meandre.training.loss import HydroLoss


def _perte(mode):
    return HydroLoss(w_kge=0.0, w_mse=0.0, w_et=1.0, et_mode=mode, per_station=False)


def test_un_mode_inconnu_est_refuse():
    with pytest.raises(ValueError, match="et_mode inconnu"):
        _perte("mediane")


def test_les_trois_modes_sont_acceptes():
    for m in ("level", "anomaly", "bassin"):
        assert _perte(m).et_mode == m


def test_le_mode_bassin_ignore_la_repartition_entre_nœuds():
    """Deux champs de même moyenne par pas de temps doivent coûter la même chose."""
    obs = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    egal = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    inverse = torch.tensor([[3.0, 1.0], [2.0, 2.0]])
    p = _perte("bassin")
    a = _cout(p, obs, egal)
    b = _cout(p, obs, inverse)
    assert a == pytest.approx(b), "le mode bassin ne doit voir que la moyenne"


def test_le_mode_niveau_voit_la_repartition():
    obs = torch.tensor([[1.0, 3.0], [2.0, 2.0]])
    egal = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    p = _perte("level")
    assert _cout(p, obs, egal) > 0.0


def test_un_nœud_sans_observation_ne_tire_pas_la_moyenne():
    obs = torch.tensor([[1.0, float("nan")], [2.0, 2.0]])
    sim = torch.tensor([[1.0, 99.0], [2.0, 2.0]])
    assert _cout(_perte("bassin"), obs, sim) == pytest.approx(0.0)


def _cout(p, obs, sim):
    """Terme d'évapotranspiration seul, fenêtre glissante neutralisée."""
    p.et_window = 1
    q = torch.ones(obs.shape[0], 1)
    _l, comps = p(q_obs=q, q_sim=q, station_mask=torch.ones(1, dtype=torch.bool),
                  et_obs=obs, et_sim=sim)
    return float(comps["et_loss"])
