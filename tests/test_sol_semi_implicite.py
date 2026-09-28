"""Schéma semi-implicite du sol : même simulation, dérivée bornée.

Le schéma explicite du clone amplifie une perturbation de l'humidité jusqu'à cent fois par
jour près de la saturation (registre, R197), si bien que le gradient explose avec l'horizon.
Le schéma linéairement implicite (MEANDRE_SOL_SEMI_IMPLICITE=1) doit rendre la même
simulation et un jacobien journalier de rayon spectral au plus un.
"""
import pytest
import torch

import hydrotel_clone.bv3c2 as bv

N = 40


def _params(texture):
    p = bv.make_params(texture, texture, texture)
    for k in list(p):
        if torch.is_tensor(p[k]) and p[k].ndim == 0:
            p[k] = p[k].expand(N).clone()
    return p


def _jour(t, p, semi):
    bv._SEMI_IMPLICITE = semi
    m = bv.BV3C2Clone(n_substep=16)
    out = m(t[0], t[1], t[2], torch.full((N,), 5.0), torch.full((N,), 2.0), torch.zeros(N), torch.zeros(N), p)
    return out[4]


@pytest.mark.parametrize("texture", ["clay", "loam", "sand"])
def test_precision_au_moins_celle_du_schema_explicite(texture):
    """Sur trente jours, l'ecart moyen a une reference a 1024 sous-pas ne doit pas depasser
    celui du schema explicite au meme nombre de sous-pas (mesure : egal sur loam et argile,
    0,016 contre 0,025 sur sable a 16 sous-pas)."""
    p = _params(texture)
    ths = float(p["thetas1"][0])

    def course(semi, nsub):
        bv._SEMI_IMPLICITE = semi
        m = bv.BV3C2Clone(n_substep=nsub)
        t = [torch.linspace(0.4, 0.99, N) * ths for _ in range(3)]
        for _ in range(30):
            t = list(m(t[0], t[1], t[2], torch.full((N,), 5.0), torch.full((N,), 2.0), torch.zeros(N), torch.zeros(N), p)[4])
        return torch.stack(t)

    ref = course(False, 1024)
    e_expl = float((course(False, 16) - ref).abs().mean())
    e_semi = float((course(True, 16) - ref).abs().mean())
    bv._SEMI_IMPLICITE = False
    assert e_semi <= e_expl * 1.05 + 1e-4


@pytest.mark.parametrize("texture", ["clay", "loam"])
def test_jacobien_journalier_contractant(texture):
    p = _params(texture)
    ths = float(p["thetas1"][0])
    t = [(torch.linspace(0.6, 0.999, N) * ths).requires_grad_(True) for _ in range(3)]
    out = _jour(t, p, True)
    J = torch.zeros(N, 3, 3)
    for i in range(3):
        g = torch.autograd.grad(out[i].sum(), t, retain_graph=True, allow_unused=True)
        for k in range(3):
            if g[k] is not None:
                J[:, i, k] = g[k]
    bv._SEMI_IMPLICITE = False
    rho = torch.linalg.eigvals(J).abs().max(dim=1).values
    assert float(rho.max()) <= 1.01
