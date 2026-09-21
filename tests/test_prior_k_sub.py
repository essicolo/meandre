"""Le plafond de percolation : moyenne ancrée et variation RÉTRÉCIE.

Mesuré le 2026-09-21 : laissé libre, ce plafond apprend un motif positionnel, sa variation
étant expliquée à 0,67 par les seules coordonnées et à 0,30 par la texture. La granulométrie
ne gouverne pas la conductivité d'un till, la compaction le fait, et elle n'est cartographiée
nulle part. Avec 3412 tronçons pour 16 stations et 27 puits sur l'Outaouais, une variation
libre par nœud n'est pas identifiable : elle est ajustée.

Le prior traite donc ce paramètre autrement que krec et k_gw, dont seule la moyenne est
ancrée. Il ancre la moyenne ET fait coûter la dispersion, ce qui est un rétrécissement et non
un gel : la variation reste possible, elle n'est plus gratuite.
"""
import math

import pytest
import torch

from meandre.spatial.field_network import KSUB_REF, SpatialFieldNetwork, SpatialParams


def _reseau(prior=True, poids=0.3):
    torch.manual_seed(0)
    net = SpatialFieldNetwork(n_territorial=4)
    net.prior_on_k_sub = prior
    net.poids_k_sub = poids
    return net


def _params(net, k_sub):
    raw = net._literature_raw_vector().unsqueeze(0).expand(len(k_sub), -1).clone()
    p = net._apply_constraints(raw)
    champs = {f.name: getattr(p, f.name) for f in p.__dataclass_fields__.values()} \
        if hasattr(p, "__dataclass_fields__") else {}
    import dataclasses
    champs = {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
    champs["k_sub"] = k_sub
    return SpatialParams(**champs)


def test_sans_le_drapeau_rien_ne_change():
    net_sans, net_avec = _reseau(prior=False), _reseau(prior=True)
    k = torch.full((32,), KSUB_REF)
    a = float(net_sans.physical_prior_loss(_params(net_sans, k)))
    b = float(net_avec.physical_prior_loss(_params(net_avec, k)))
    assert b == pytest.approx(a, abs=1e-6), "a la reference, le prior ne doit rien couter"


def test_la_moyenne_est_ancree():
    net = _reseau()
    ref = float(net.physical_prior_loss(_params(net, torch.full((32,), KSUB_REF))))
    for facteur in (0.1, 10.0):
        loin = float(net.physical_prior_loss(_params(net, torch.full((32,), KSUB_REF * facteur))))
        assert loin > ref + 0.05, f"un plafond {facteur} fois la reference doit couter"


def test_la_dispersion_coute_meme_a_moyenne_juste():
    """Le point qui distingue ce prior de celui de krec."""
    net = _reseau()
    n = 64
    uniforme = torch.full((n,), KSUB_REF)
    # Meme moyenne EN LOG, mais dispersee d'un facteur trois de part et d'autre.
    disperse = torch.tensor([KSUB_REF * (3.0 if i % 2 else 1 / 3.0) for i in range(n)])
    a = float(net.physical_prior_loss(_params(net, uniforme)))
    b = float(net.physical_prior_loss(_params(net, disperse)))
    assert b > a + 0.1, "une variation gratuite doit desormais couter"


def test_le_poids_regle_la_force_du_retrecissement():
    n = 64
    disperse = torch.tensor([KSUB_REF * (3.0 if i % 2 else 1 / 3.0) for i in range(n)])
    faible = _reseau(poids=0.1)
    fort = _reseau(poids=1.0)
    a = float(faible.physical_prior_loss(_params(faible, disperse)))
    b = float(fort.physical_prior_loss(_params(fort, disperse)))
    assert b > a, "un poids plus grand doit retrecir davantage"


def test_le_prior_reste_derivable():
    net = _reseau()
    k = (torch.full((16,), KSUB_REF) * torch.linspace(0.5, 2.0, 16)).requires_grad_(True)
    perte = net.physical_prior_loss(_params(net, k))
    perte.backward()
    assert k.grad is not None and torch.isfinite(k.grad).all()
