"""L'ancrage geometrique du temps de transfert Muskingum.

Mesure du 2026-09-14 : sur 2 212 troncons du Saguenay, le temps de transfert appris vaut
23,96 h avec un ecart-type de 0,3 h, et le coefficient de ponderation 0,202 a 0,006 pres.
Les deux sont constants : la sortie de routage du champ n'a pas appris, sa ligne de poids
ayant une norme quinze fois plus faible que celle de la conductivite a saturation. Or le
temps de parcours physique d'un troncon median de 7,1 km vaut 2,0 h. L'ancrage remplace la
constante par la geometrie ; ces tests fixent son comportement et sa neutralite quand il
est absent.
"""
import numpy as np
import pytest
import torch

from meandre.spatial.field_network import SpatialFieldNetwork


def _champ(n=64):
    torch.manual_seed(0)
    f = SpatialFieldNetwork(n_territorial=16, n_nodes=n)
    f.eval()
    return f


def _entrees(n=64):
    torch.manual_seed(1)
    return torch.randn(n, 2) * 2 + torch.tensor([-72.0, 50.0]), torch.randn(n, 16)


def test_sans_ancre_le_comportement_est_inchange():
    """Condition pour que l'ancrage soit un choix : sans lui, rien ne bouge."""
    f, (c, t) = _champ(), _entrees()
    with torch.no_grad():
        a = f(c, t).K_musk_hours.clone()
    f.set_routing_anchor(None)
    with torch.no_grad():
        b = f(c, t).K_musk_hours
    assert torch.equal(a, b)


def _identiques(n=2):
    """Memes coordonnees et memes attributs sur tous les noeuds : la modulation du reseau
    est alors identique et seule l'ancre distingue les troncons."""
    return (torch.tensor([[-72.0, 50.0]]).repeat(n, 1), torch.zeros(n, 16))


def test_l_ancre_impose_le_temps_de_parcours_geometrique():
    """Un troncon deux fois plus long doit prendre deux fois plus de temps."""
    f, (c, t) = _champ(2), _identiques(2)
    f.set_routing_anchor(torch.tensor([5.0, 10.0]))
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    assert float(k[1] / k[0]) == pytest.approx(2.0, rel=1e-4)


def test_la_pente_accelere_l_ecoulement():
    """Selon Manning, la vitesse suit la racine de la pente : quatre fois plus de pente
    donne deux fois moins de temps de parcours."""
    f, (c, t) = _champ(2), _identiques(2)
    f.set_routing_anchor(torch.tensor([7.0, 7.0]), slope_frac=torch.tensor([0.005, 0.020]))
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    assert float(k[0] / k[1]) == pytest.approx(2.0, rel=1e-3)


def test_l_ancre_donne_un_temps_physique_sur_un_troncon_median():
    """7,1 km a un metre par seconde : environ 1,2 h avec la celerite cinematique,
    soit vingt fois moins que l'initialisation constante de 24 h."""
    f, (c, t) = _champ(1), _entrees(1)
    f.set_routing_anchor(torch.tensor([7.1]))
    with torch.no_grad():
        k = float(f(c, t).K_musk_hours[0])
    assert 0.3 < k < 6.0, k


def test_le_reseau_module_autour_de_l_ancre_sans_la_quitter():
    """La modulation est bornee : elle ne peut pas rendre un temps aberrant, quel que
    soit ce que le reseau apprend."""
    f, (c, t) = _champ(), _entrees()
    L = torch.full((64,), 7.1)
    f.set_routing_anchor(L)
    with torch.no_grad():
        anc = f._k_musk_anchor
        for g in (-50.0, 50.0):
            with torch.no_grad():
                f.fc_out.bias.fill_(g)
            k = f(c, t).K_musk_hours
            assert torch.all(k >= 0.05) and torch.all(k <= 48.0)
            assert torch.all(k <= anc * np.e ** 1.5 + 1e-4)
            assert torch.all(k >= torch.clamp(anc * np.e ** -1.5, min=0.05) - 1e-4)


def test_l_ancre_varie_d_un_troncon_a_l_autre():
    """Le defaut corrige : un parametre constant sur toute la region."""
    f, (c, t) = _champ(), _entrees()
    torch.manual_seed(2)
    L = torch.rand(64) * 20 + 0.5
    f.set_routing_anchor(L)
    with torch.no_grad():
        k = f(c, t).K_musk_hours
    assert float(k.std() / k.mean()) > 0.3, float(k.std() / k.mean())


def test_le_gradient_passe_par_la_modulation():
    """Le reseau doit pouvoir corriger l'ancre, sinon elle est une contrainte figee."""
    f, (c, t) = _champ(8), _entrees(8)
    f.set_routing_anchor(torch.full((8,), 7.1))
    f(c, t).K_musk_hours.sum().backward()
    assert f.fc_out.weight.grad is not None
    assert torch.isfinite(f.fc_out.weight.grad).all()
    assert float(f.fc_out.weight.grad.abs().sum()) > 0
