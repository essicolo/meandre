"""La capacité au champ se déduit de la courbe de rétention, elle ne s'apprend pas à côté.

Le calage d'Hydrotel impose par nœud la courbe de Campbell, b et psi_s, mais ne fournit aucune
capacité au champ : celle-ci sortait du champ spatial, librement. Or ce n'est pas une
propriété indépendante, c'est la teneur en eau à une succion de référence SUR CETTE COURBE.
Une capacité au champ et une courbe qui ne s'accordent pas décrivent deux sols différents.

Le banc de sensibilité du 2026-09-22 a montré que la capacité au champ de la couche profonde
est le deuxième champ le plus sensible des quarante-trois : la laisser libre revenait à
ajuster un paramètre de premier ordre sans qu'aucune donnée ne le rattache à la texture.
"""
import pytest
import torch

from meandre.vertical.hydrotel_column import HydrotelColumn

DEDUIRE = HydrotelColumn.capacite_au_champ_de_la_courbe


def _sol(b=4.0, psis=0.3, thetas=0.45, n=3):
    return {"b3": torch.full((n,), b), "psis3": torch.full((n,), psis),
            "thetas3": torch.full((n,), thetas)}


def test_la_capacite_reste_sous_la_porosite():
    v = DEDUIRE(_sol(), 3)
    assert float(v.max()) < 0.45


def test_un_sol_plus_argileux_retient_davantage():
    """Un b plus grand est une courbe plus plate : plus d'eau retenue a la meme succion."""
    sable = float(DEDUIRE(_sol(b=2.5), 3)[0])
    argile = float(DEDUIRE(_sol(b=8.0), 3)[0])
    assert argile > sable


def test_une_porosite_plus_grande_donne_une_capacite_plus_grande():
    a = float(DEDUIRE(_sol(thetas=0.40), 3)[0])
    b = float(DEDUIRE(_sol(thetas=0.55), 3)[0])
    assert b > a


def test_la_valeur_est_dans_la_plage_pedologique():
    """Un loam de b = 4, psi_s = 0,3 m doit donner une capacite au champ plausible."""
    v = float(DEDUIRE(_sol(), 3)[0])
    assert 0.10 < v < 0.40, f"capacite au champ hors plage : {v}"


def test_sans_courbe_la_deduction_rend_rien():
    """Une recette sans courbe imposee doit retomber sur le champ spatial, pas sur zero."""
    assert DEDUIRE({"b3": torch.ones(2)}, 3) is None


def test_la_succion_de_reference_agit_dans_le_bon_sens():
    humide = float(DEDUIRE(_sol(), 3, psi_fc=1.0)[0])
    sec = float(DEDUIRE(_sol(), 3, psi_fc=10.0)[0])
    assert humide > sec, "une succion plus forte doit retenir moins d'eau"


def test_le_gradient_traverse_la_deduction():
    """La capacite reste derivable : elle depend de parametres que le champ peut porter."""
    thetas = torch.full((2,), 0.45, requires_grad=True)
    v = DEDUIRE({"b3": torch.full((2,), 4.0), "psis3": torch.full((2,), 0.3),
                 "thetas3": thetas}, 3)
    v.sum().backward()
    assert float(thetas.grad.abs().sum()) > 0
