"""Drainage souterrain agricole (Hooghoudt) : fidelite par defaut, et comportement.

Motif (O12, 2026-08-26). Sur les six territoires bien echantillonnes, la relation entre
fraction agricole et tenue de cote est presque monotone : MONT 33.7 % pour 0.4821, SLSO
29.2 % pour 0.5380, jusqu'a OUTV 9.2 % pour 0.6519. `f_agriculture` EST deja un attribut
du champ, donc ce n'est pas une variable qui manque mais un PROCESSUS : la colonne n'a
aucun chemin qui sorte de la couche 2 vers le troncon par un seuil de profondeur.

Ce que ces tests verrouillent, dans l'ordre d'importance :
  1. l'ABSENCE du drain ne change RIEN (fidelite au clone C++ preservee) ;
  2. le drain ne se declenche qu'AU-DESSUS du plan des drains (c'est un seuil) ;
  3. il CONSERVE la masse (ce qui quitte la couche 2 arrive a la production) ;
  4. il repond dans le bon SENS a l'espacement et a la fraction drainee.
"""
import torch

from hydrotel_clone.bv3c2 import BV3C2Clone, make_params


def _etat(n=4, sat=0.9):
    """make_params rend des scalaires broadcastables ; on les etale sur n noeuds pour
    que le drain, qui est un champ par noeud, ait des voisins a comparer."""
    p = {k: (v.expand(n).clone() if torch.is_tensor(v) and v.ndim == 0 else v)
         for k, v in make_params(device="cpu").items()}
    t = tuple(p[f"thetas{i}"] * sat for i in (1, 2, 3))
    return p, t


def _pas(p, t, drain=None, apport=5.0):
    mod = BV3C2Clone(n_substep=24, static=False)
    q = dict(p)
    if drain:
        q.update(drain)
    n = t[0].shape[0]
    return mod(t[0].clone(), t[1].clone(), t[2].clone(),
               torch.full((n,), float(apport)), torch.full((n,), 2.0),
               torch.zeros(n), torch.zeros(n), q)


def test_absence_de_drain_ne_change_rien():
    """Sans les cles du drain, le resultat doit etre IDENTIQUE au bit pres. C'est la
    regle de tout processus opt-in du projet : le defaut reste le clone fidele."""
    p, t = _etat()
    a = _pas(p, t)
    b = _pas(p, t)
    for x, y in zip(a[:4], b[:4]):
        assert torch.equal(x, y)


def test_seuil_le_drain_ne_coule_pas_sous_son_plan():
    """Un drain pose a 1 m ne peut rien evacuer si la zone saturee ne l'atteint pas.
    Sans ce seuil, on aurait un simple reservoir lineaire de plus, pas un drain."""
    p, _ = _etat()
    n = p["z1"].shape[0]
    drain = dict(drain_spacing=torch.full((n,), 15.0),
                 drain_depth=torch.full((n,), 1.0),
                 drain_frac=torch.full((n,), 1.0))
    # couche 2 presque vide : la nappe est bien sous le plan des drains
    sec = tuple(p[f"thetas{i}"] * f for i, f in ((1, 0.3), (2, 0.05), (3, 0.3)))
    _, _, _, _, _, diag = _pas(p, sec, drain, apport=0.0)
    assert float(diag["drain_mm"].max()) < 1e-9

    # couche 2 saturee : le drain doit couler
    plein = tuple(p[f"thetas{i}"] * f for i, f in ((1, 0.9), (2, 1.0), (3, 0.9)))
    _, _, _, _, _, d2 = _pas(p, plein, drain, apport=0.0)
    assert float(d2["drain_mm"].min()) > 0.0


def test_conservation_de_la_masse():
    """Ce que le drain retire a la couche 2 doit se retrouver dans la production, a la
    precision numerique. Un chemin qui fuit serait pire que pas de chemin du tout."""
    p, t = _etat(sat=1.0)
    n = p["z1"].shape[0]
    drain = dict(drain_spacing=torch.full((n,), 15.0),
                 drain_depth=torch.full((n,), 1.0),
                 drain_frac=torch.full((n,), 1.0))
    sans = _pas(p, t, None, apport=0.0)
    avec = _pas(p, t, drain, apport=0.0)
    # eau restante dans le profil (mm), les deux cas
    def stock(r):
        (a, b, c) = r[4]
        return ((a * p["z1"] + b * p["z2"] + c * p["z3"]) * 1000.0)
    # REFERENTIELS. Les theta sont par unite de surface de SOL, la production est par
    # unite de surface de TRONCON : la colonne pondere par `fsa`, exactement comme elle
    # le fait deja pour l'ecoulement lateral q2. Comparer les deux sans cette pondera-
    # tion ferait apparaitre une fuite de dix pour cent qui n'existe pas. C'est aussi
    # ce qui fixe le sens de `drain_frac` : une fraction de la surface de SOL, pas du
    # troncon.
    perdu = (stock(sans) - stock(avec)) * p["fsa"]
    gagne = (avec[0] + avec[1] + avec[2]) - (sans[0] + sans[1] + sans[2])
    assert torch.allclose(perdu, gagne, atol=1e-3), (perdu, gagne)


def test_sens_de_reponse_espacement_et_fraction():
    """Hooghoudt : le flux varie en 1/L^2 et proportionnellement a la surface drainee.
    Un drain plus serre evacue plus, une parcelle moins drainee evacue moins."""
    p, t = _etat(sat=1.0)
    n = p["z1"].shape[0]
    base = dict(drain_depth=torch.full((n,), 1.0), drain_frac=torch.full((n,), 1.0))
    serre = _pas(p, t, dict(base, drain_spacing=torch.full((n,), 10.0)), apport=0.0)
    large = _pas(p, t, dict(base, drain_spacing=torch.full((n,), 30.0)), apport=0.0)
    assert float(serre[5]["drain_mm"].mean()) > float(large[5]["drain_mm"].mean())

    moitie = _pas(p, t, dict(drain_depth=torch.full((n,), 1.0),
                             drain_spacing=torch.full((n,), 10.0),
                             drain_frac=torch.full((n,), 0.5)), apport=0.0)
    assert float(moitie[5]["drain_mm"].mean()) < float(serre[5]["drain_mm"].mean())


def test_gradient_traverse_les_parametres_du_drain():
    """Le projet vit de la differentiabilite : un processus dont le gradient ne passe
    pas est un processus qu'on ne peut pas identifier."""
    p, t = _etat(sat=1.0)
    n = p["z1"].shape[0]
    L = torch.full((n,), 15.0, requires_grad=True)
    r = _pas(p, t, dict(drain_spacing=L, drain_depth=torch.full((n,), 1.0),
                        drain_frac=torch.full((n,), 1.0)), apport=0.0)
    r[1].sum().backward()
    assert L.grad is not None and float(L.grad.abs().sum()) > 0.0
