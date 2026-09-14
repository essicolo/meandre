"""Le degre-jour integrant le cycle diurne, et sa neutralite quand il est desactive."""
import math

import pytest
import torch

from hydrotel_clone.snow import degre_jour_effectif


def test_journee_entierement_au_dessus_du_seuil():
    """Quand le minimum depasse deja le seuil, l'integrale vaut la moyenne moins le
    seuil : le cycle diurne ne change rien, la partie positive couvre la journee."""
    tmin = torch.tensor([5.0, 10.0])
    tmax = torch.tensor([15.0, 12.0])
    dj = degre_jour_effectif(tmin, tmax, 0.0)
    assert torch.allclose(dj, torch.tensor([10.0, 11.0]), atol=1e-4)


def test_journee_entierement_sous_le_seuil():
    tmin = torch.tensor([-20.0, -5.0])
    tmax = torch.tensor([-10.0, -0.1])
    dj = degre_jour_effectif(tmin, tmax, 0.0)
    assert torch.all(dj == 0.0)


def test_seuil_a_la_moyenne_donne_la_valeur_analytique():
    """Seuil exactement a la moyenne : la moitie du cycle est positive et l'integrale
    de la partie positive d'une sinusoide d'amplitude B vaut B / pi."""
    tmin, tmax = torch.tensor([-10.0]), torch.tensor([10.0])
    dj = degre_jour_effectif(tmin, tmax, 0.0)
    assert dj.item() == pytest.approx(10.0 / math.pi, rel=1e-3)


def test_le_redoux_efface_par_la_moyenne_est_retrouve():
    """Le cas qui motive le correctif : maximum au-dessus du seuil, moyenne en dessous.
    Le clone rend zero, la version diurne rend une fonte strictement positive."""
    tmin, tmax = torch.tensor([-8.0]), torch.tensor([6.0])
    moyenne = (tmin + tmax) / 2
    assert moyenne.item() < 0.5
    dj = degre_jour_effectif(tmin, tmax, 0.5)
    assert dj.item() > 0.3


def test_toujours_superieur_ou_egal_au_degre_jour_sur_la_moyenne():
    """L'integrale de la partie positive majore la partie positive de la moyenne, par
    convexite. Un correctif qui ferait FONDRE MOINS serait un bogue."""
    g = torch.Generator().manual_seed(0)
    tmoy = torch.randn(2000, generator=g) * 12.0
    demi = torch.rand(2000, generator=g) * 10.0
    dj = degre_jour_effectif(tmoy - demi, tmoy + demi, 0.5)
    ref = torch.clamp(tmoy - 0.5, min=0.0)
    assert torch.all(dj >= ref - 1e-4)


def test_croissance_avec_l_amplitude_diurne():
    """A moyenne fixee sous le seuil, elargir l'amplitude diurne augmente la fonte."""
    tmoy = torch.full((5,), -2.0)
    demi = torch.tensor([1.0, 3.0, 6.0, 9.0, 12.0])
    dj = degre_jour_effectif(tmoy - demi, tmoy + demi, 0.0)
    assert torch.all(torch.diff(dj) > 0)


def test_gradient_fini_partout():
    """Le seuil traverse le cycle : la derivee de l'arc cosinus diverge aux bornes et
    doit rester bornee par le calage de l'argument."""
    for lo, hi in [(-10.0, 10.0), (0.0, 0.0), (-1e-7, 1e-7), (0.5, 0.5), (-30.0, -20.0)]:
        tmin = torch.tensor([lo], requires_grad=True)
        tmax = torch.tensor([hi], requires_grad=True)
        degre_jour_effectif(tmin, tmax, 0.5).sum().backward()
        assert torch.isfinite(tmin.grad).all()
        assert torch.isfinite(tmax.grad).all()


def test_desactive_le_clone_est_rendu_a_l_identique():
    """calcule_fonte sans l'option doit donner exactement le meme resultat qu'avant :
    c'est la condition pour que l'ecart a Hydrotel reste un choix et non un accident."""
    from hydrotel_clone.snow import calcule_fonte
    n = 40
    g = torch.Generator().manual_seed(1)
    kw = dict(pluie_m=torch.rand(n, generator=g) * 0.01,
              neige_m=torch.rand(n, generator=g) * 0.01,
              indice_rad=torch.rand(n, generator=g),
              stock=torch.rand(n, generator=g) * 0.3,
              hauteur=torch.rand(n, generator=g) * 1.2 + 0.1,
              chaleur=-torch.rand(n, generator=g) * 1e6,
              eau_retenue=torch.rand(n, generator=g) * 0.01,
              albedo=torch.rand(n, generator=g) * 0.4 + 0.4,
              coeff_fonte=torch.full((n,), 0.004), seuil_fonte=torch.full((n,), 0.5),
              taux_fonte_geo=torch.full((n,), 0.0001), densite_max=torch.full((n,), 550.0),
              constante_tassement=torch.full((n,), 0.3))
    tmoy = torch.randn(n, generator=g) * 10.0
    tmin, tmax = tmoy - 5.0, tmoy + 5.0
    a = calcule_fonte(tmin, tmax, **kw)
    b = calcule_fonte(tmin, tmax, melt_diurnal=False, **kw)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def _cumul_fonte(jours, diurne, tmin, tmax):
    """Enchaine les pas de temps en reportant l'etat du manteau, et rend l'eau totale
    liberee. Un pas isole ne suffit pas : l'eau fondue rechauffe d'abord le manteau puis
    y est retenue, et la liberation ne commence qu'une fois celui-ci mur."""
    from hydrotel_clone.snow import calcule_fonte
    st = torch.tensor([0.30])
    ha = torch.tensor([0.90])
    ch = torch.tensor([-2.0e6])
    er = torch.zeros(1)
    alb = torch.tensor([0.55])
    total = 0.0
    for _ in range(jours):
        f, st, ha, ch, er, alb = calcule_fonte(
            tmin, tmax, torch.zeros(1), torch.zeros(1), torch.tensor([0.6]),
            st, ha, ch, er, alb, torch.tensor([0.004]), torch.tensor([0.5]),
            torch.tensor([1e-4]), torch.tensor([550.0]), torch.tensor([0.3]),
            melt_diurnal=diurne)
        total += float(f.sum())
    return total


def test_active_la_fonte_hivernale_apparait():
    """Le cas qui motive le correctif, sur une saison entiere : des jours dont le maximum
    depasse le seuil et dont la moyenne ne le depasse pas. Le clone ne libere rien, la
    version diurne finit par murir le manteau et liberer de l'eau."""
    tmin, tmax = torch.tensor([-8.0]), torch.tensor([6.0])
    sans = _cumul_fonte(60, False, tmin, tmax)
    avec = _cumul_fonte(60, True, tmin, tmax)
    assert sans == 0.0
    assert avec > 0.0


def test_un_hiver_franchement_froid_ne_fond_pas_davantage():
    """Garde-fou : sous un froid ou meme le maximum reste sous le seuil, l'option ne doit
    rien liberer non plus, sinon elle fabriquerait de l'eau."""
    tmin, tmax = torch.tensor([-25.0]), torch.tensor([-12.0])
    assert _cumul_fonte(60, True, tmin, tmax) == 0.0
