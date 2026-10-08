"""Bornes de conductivite par troncon (texture) : la sortie du champ reste entre lo et hi de chaque noeud,
et une sortie brute nulle rend la valeur de texture (moyenne geometrique des bornes)."""
import numpy as np
import torch

from meandre.data.texture_bounds import bounds_from_texture
from meandre.spatial.field_network import SpatialFieldNetwork


def _champ(n):
    return SpatialFieldNetwork(n_nodes=n, n_territorial=4, use_latent_codes=False)


def test_sortie_entre_les_bornes_de_chaque_noeud():
    torch.manual_seed(0)
    n = 6
    f = _champ(n)
    ks = np.array([0.1, 0.5, 2.0, np.nan, 1.0, 0.05])
    lo, hi = bounds_from_texture(ks, 3.0, 3.355e-4, 54.6)
    f.set_node_bounds("K_sat_1", torch.tensor(lo), torch.tensor(hi))
    coords = torch.rand(n, 2); terr = torch.randn(n, 4)
    for _ in range(3):
        sp = f(coords, terr)
        k = sp.K_sat_1.detach().numpy()
        assert np.all(k >= lo - 1e-6) and np.all(k <= hi + 1e-6)
    # le noeud sans texture garde les bornes uniques
    assert lo[3] == np.float32(3.355e-4) and hi[3] == np.float32(54.6)


def test_brut_nul_rend_la_texture():
    n = 3
    f = _champ(n)
    ks = np.array([0.1, 0.5, 2.0])
    lo, hi = bounds_from_texture(ks, 3.0, 3.355e-4, 54.6)
    f.set_node_bounds("K_sat_2", torch.tensor(lo), torch.tensor(hi))
    sp = f._apply_constraints(torch.zeros(n, sp_dim(f)))
    assert np.allclose(sp.K_sat_2.detach().numpy(), ks, rtol=1e-4)


def sp_dim(f):
    from meandre.spatial.field_network import SpatialParams
    return SpatialParams.N_PARAMS
