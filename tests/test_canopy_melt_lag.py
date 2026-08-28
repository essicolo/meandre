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

    Ce test appelle le VRAI chemin de chargement, HydroModel.load. Sa premiere version
    reimplementait la logique a cote et passait donc quoi qu'il arrive : un test qui refait
    le travail du code ne teste rien. En l'ecrivant pour de bon on decouvre que le
    rembourrage existait DEJA dans model.py et fonctionnait ; le correctif que j'avais
    ajoute le 2026-08-28 etait du code mort, et a ete retire.

    Comportement verifie ici : les lignes apprises sont reprises telles quelles, et les
    nouvelles sont mises a ZERO -- donc au milieu de leurs bornes apres contrainte, pas a
    la cible de litterature.
    """
    import torch

    from meandre.model import HydroModel

    m = HydroModel(n_territorial=4, n_nodes=8)
    n_new = m.spatial_encoder.fc_out.weight.shape[0]
    n_in = m.spatial_encoder.fc_out.weight.shape[1]
    n_old = n_new - 2

    sd = {k: v.clone() for k, v in m.state_dict().items()}
    vieux_w = torch.randn(n_old, n_in)
    vieux_b = torch.randn(n_old)
    sd["spatial_encoder.fc_out.weight"] = vieux_w
    sd["spatial_encoder.fc_out.bias"] = vieux_b
    chemin = tmp_path / "vieux.pt"
    torch.save({"state_dict": sd}, chemin)

    neuf = HydroModel(n_territorial=4, n_nodes=8)
    avant = neuf.spatial_encoder.fc_out.weight.detach().clone()
    neuf.load(chemin)
    apres = neuf.spatial_encoder.fc_out.weight.detach()

    assert apres.shape[0] == n_new
    # les lignes apprises sont reprises
    assert torch.allclose(apres[:n_old], vieux_w, atol=1e-6)
    # les deux nouvelles sont mises a zero par le rembourrage historique
    assert torch.allclose(apres[n_old:], torch.zeros_like(apres[n_old:]), atol=1e-6)
    assert not torch.allclose(avant[n_old:], torch.zeros_like(avant[n_old:]))
    assert torch.allclose(neuf.spatial_encoder.fc_out.bias.detach()[:n_old], vieux_b, atol=1e-6)
