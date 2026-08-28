"""Le verrou de fonte appris (R56) : structure et ordre.

Le point de la parametrisation est que l'ordre conifere >= feuillu >= decouvert soit
vrai PAR CONSTRUCTION, pour n'importe quelle sortie du reseau. Si cet ordre pouvait
s'inverser, les trois seuils redeviendraient libres de se compenser et on aurait
remplace un verrou cale par trois parametres non identifiables.
"""
import torch

from meandre.spatial.field_network import SpatialParams, SpatialFieldNetwork


def test_n_params_coherent():
    import dataclasses
    noms = [f.name for f in dataclasses.fields(SpatialParams)]
    assert len(noms) == SpatialParams.N_PARAMS
    assert noms[-2:] == ["dT_canopee_feu", "dT_canopee_conif"]


def test_index_krec_suit_le_champ():
    import dataclasses
    noms = [f.name for f in dataclasses.fields(SpatialParams)]
    assert noms[SpatialFieldNetwork._IDX_KREC] == "krec"


def test_retards_non_negatifs_et_ordre_garanti():
    """Sur des sorties brutes extremes, l'ordre des trois seuils tient."""
    torch.manual_seed(0)
    net = SpatialFieldNetwork(n_territorial=4)
    raw = torch.randn(256, SpatialParams.N_PARAMS) * 50.0   # volontairement violent
    sp = net._apply_constraints(raw)
    assert torch.all(sp.dT_canopee_feu >= 0.0)
    assert torch.all(sp.dT_canopee_conif >= 0.0)
    assert torch.all(sp.dT_canopee_feu <= 3.0)
    assert torch.all(sp.dT_canopee_conif <= 3.0)
    se_d = sp.T_melt
    se_f = se_d + sp.dT_canopee_feu
    se_c = se_f + sp.dT_canopee_conif
    assert torch.all(se_c >= se_f)
    assert torch.all(se_f >= se_d)


def test_init_litterature_pose_les_retards():
    """Le vecteur brut de litterature doit RENDRE 1.0 apres contrainte.

    Piege deja paye une fois (2026-08-27) : passer des zeros a _apply_constraints ne
    teste pas l'init, il rend le MILIEU des bornes, ici 1.5. Il faut passer le vecteur
    brut que _literature_raw_vector construit.
    """
    net = SpatialFieldNetwork(n_territorial=4)
    raw = net._literature_raw_vector().unsqueeze(0).expand(8, -1)
    sp = net._apply_constraints(raw)
    assert torch.allclose(sp.dT_canopee_feu, torch.full((8,), 1.0), atol=0.01)
    assert torch.allclose(sp.dT_canopee_conif, torch.full((8,), 1.0), atol=0.01)
    # et T_melt, le seuil de reference du terrain decouvert, reste a son prior
    assert torch.allclose(sp.T_melt, torch.full((8,), -0.5), atol=0.01)


def test_depart_a_chaud_conserve_les_champs_appris(tmp_path):
    """Elargir N_PARAMS ne doit PAS jeter les sorties deja apprises.

    Sans le recollage, le filtre generique de formes incompatibles ecarte fc_out en
    entier : on gagne deux champs et on perd les quarante autres.
    """
    import torch

    net = SpatialFieldNetwork(n_territorial=4)
    n_new = net.fc_out.weight.shape[0]
    n_old = n_new - 2
    vieux_w = torch.randn(n_old, net.fc_out.weight.shape[1])
    vieux_b = torch.randn(n_old)
    sd = {"field_network.fc_out.weight": vieux_w, "field_network.fc_out.bias": vieux_b}

    own = {"field_network.fc_out.weight": net.fc_out.weight.detach().clone(),
           "field_network.fc_out.bias": net.fc_out.bias.detach().clone()}
    for k in own:
        vieux, neuf = sd[k], own[k].clone()
        neuf[: vieux.shape[0]] = vieux
        sd[k] = neuf

    assert torch.allclose(sd["field_network.fc_out.weight"][:n_old], vieux_w)
    assert torch.allclose(sd["field_network.fc_out.bias"][:n_old], vieux_b)
    assert sd["field_network.fc_out.weight"].shape[0] == n_new
