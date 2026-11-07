"""Le verrou de fonte appris (R56) : structure et ordre.

Le point de la parametrisation est que l'ordre conifere >= feuillu >= decouvert soit
vrai PAR CONSTRUCTION, pour n'importe quelle sortie du reseau. Si cet ordre pouvait
s'inverser, les trois seuils redeviendraient libres de se compenser et on aurait
remplace un verrou cale par trois parametres non identifiables.
"""
import dataclasses

import torch

from meandre.spatial.field_network import SpatialParams, SpatialFieldNetwork


def test_n_params_coherent():
    import dataclasses
    name_list = [f.name for f in dataclasses.fields(SpatialParams)]
    assert len(name_list) == SpatialParams.N_PARAMS
    # Les deux retards de fonte sous couvert restent ADJACENTS et dans cet ordre, la
    # construction du conifere ajoutant son retard a celui du feuillu. Ils ne sont plus en
    # queue depuis l'ajout du plafond de percolation le 2026-09-20 : toute sortie ajoutee va
    # en DERNIER, pour que le remplissage par zeros d'un ancien point de reprise reste
    # inoffensif, ce qui deplace mecaniquement ce qui precede.
    i = name_list.index("dT_canopee_feu")
    assert name_list[i:i + 2] == ["dT_canopee_feu", "dT_canopee_conif"]
    assert name_list[-1] == "k_sub", "la derniere sortie ajoutee doit rester en queue"


def test_krec_index_follows_the_field():
    import dataclasses
    name_list = [f.name for f in dataclasses.fields(SpatialParams)]
    assert name_list[SpatialFieldNetwork._IDX_KREC] == "krec"


def test_nonnegative_lags_and_guaranteed_order():
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


def test_literature_init_sets_the_lags():
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


def _legacy_checkpoint(tmp_path, n_old, n_in):
    """Point de reprise ancien : fc_out de n_old lignes, prefixe de LEGACY_FIELD_ORDER, sans liste de noms."""
    import torch

    from meandre.model import HydroModel

    m = HydroModel(n_territorial=4, n_nodes=8)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    old_w = torch.randn(n_old, n_in)
    old_b = torch.randn(n_old)
    sd["spatial_encoder.fc_out.weight"] = old_w
    sd["spatial_encoder.fc_out.bias"] = old_b
    file_path = tmp_path / f"vieux-{n_old}.pt"
    torch.save({"state_dict": sd}, file_path)
    return file_path, old_w, old_b


def test_warm_start_keeps_learned_fields(tmp_path):
    """Un ancien point de reprise se relit PAR NOM : sorties gardees reprises, retirees ecartees.

    Ce test appelle le VRAI chemin de chargement, HydroModel.load. Les quarante-trois sorties
    d'avant le 2026-11-07 suivent LEGACY_FIELD_ORDER ; les plus anciens points de reprise en
    portent un prefixe. Les sorties absentes du point de reprise sont mises a ZERO, donc au
    milieu de leurs bornes apres contrainte, pas a la cible de litterature.
    """
    import torch

    from meandre.model import HydroModel
    from meandre.spatial.field_network import LEGACY_FIELD_ORDER

    names = [f.name for f in dataclasses.fields(SpatialParams)]
    neuf = HydroModel(n_territorial=4, n_nodes=8)
    n_in = neuf.spatial_encoder.fc_out.weight.shape[1]
    # Point de reprise complet de 43 sorties : chaque sortie gardee retrouve exactement sa ligne.
    file_path, old_w, old_b = _legacy_checkpoint(tmp_path, len(LEGACY_FIELD_ORDER), n_in)
    neuf.load(file_path)
    w = neuf.spatial_encoder.fc_out.weight.detach()
    b = neuf.spatial_encoder.fc_out.bias.detach()
    assert w.shape[0] == SpatialParams.N_PARAMS
    for j, n in enumerate(names):
        i = LEGACY_FIELD_ORDER.index(n)
        assert torch.equal(w[j], old_w[i]) and float(b[j]) == float(old_b[i]), n
    # Point de reprise de 38 sorties : les sorties ajoutees apres sont remplies de zeros.
    neuf = HydroModel(n_territorial=4, n_nodes=8)
    file_path, old_w, old_b = _legacy_checkpoint(tmp_path, 38, n_in)
    neuf.load(file_path)
    w = neuf.spatial_encoder.fc_out.weight.detach()
    for j, n in enumerate(names):
        i = LEGACY_FIELD_ORDER.index(n)
        if i < 38:
            assert torch.equal(w[j], old_w[i]), n
        else:
            assert torch.equal(w[j], torch.zeros_like(w[j])), n


def test_saved_checkpoint_round_trip(tmp_path):
    """Un point de reprise ecrit par save porte la liste de ses sorties et se relit a l'identique."""
    import torch

    from meandre.model import HydroModel

    m = HydroModel(n_territorial=4, n_nodes=8)
    path = tmp_path / "neuf.pt"
    m.save(path)
    assert torch.load(path, map_location="cpu")["spatial_field_names"] == [f.name for f in dataclasses.fields(SpatialParams)]
    neuf = HydroModel(n_territorial=4, n_nodes=8)
    neuf.load(path)
    assert torch.equal(neuf.spatial_encoder.fc_out.weight, m.spatial_encoder.fc_out.weight)


def test_too_many_outputs_without_names_fails(tmp_path):
    """Plus de sorties que l'ancien ordre, sans leurs noms : erreur claire, pas une forme cryptique."""
    import pytest

    from meandre.model import HydroModel
    from meandre.spatial.field_network import LEGACY_FIELD_ORDER

    neuf = HydroModel(n_territorial=4, n_nodes=8)
    file_path, _, _ = _legacy_checkpoint(tmp_path, len(LEGACY_FIELD_ORDER) + 1, neuf.spatial_encoder.fc_out.weight.shape[1])
    with pytest.raises(ValueError, match="point de reprise illisible"):
        neuf.load(file_path)
