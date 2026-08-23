"""Modulation saisonniere du facteur de fonte (R32, 2026-08-22).

Premiere extension de physique ASSUMEE au-dela de la fidelite Hydrotel (accord d'Essi :
« on pourrait tres bien ajouter des processus qui permettraient de capter les
irregularites »). Le diagnostic qui la motive, sur 3203 intervalles apparies CanSWE :
accumulation a 58 % de la mesure, fonte de coeur d'hiver presque juste (27.1 % contre
22.9 mesure et 24.6 de reference), date de disparition juste a +2/0/0 jours. Un facteur
degre-jour CONSTANT, cale sur la crue, fond trop quand la radiation est minimale --
novembre-decembre, la ou le manteau simule vaut 0.55-0.66 du mesure.

Le contrat le plus important ici est la FIDELITE PAR DEFAUT : sans amplitude posee, le
clone doit rester identique au bit pres, parce que toute la validation per-UHRH contre
le binaire C++ repose dessus.
"""
import torch
import pytest

from hydrotel_clone.snow import DegreJourModifie, init_state


def _params(n=4, **extra):
    t = lambda v: torch.full((n,), float(v), dtype=torch.float64)
    p = dict(lat=t(46.0), ce1=t(0.85), ce0=t(0.55),
             pct_conifers=t(0.3), pct_feuillus=t(0.4), pct_autres=t(0.3),
             coeff_fonte_conifers=t(0.009), coeff_fonte_feuillus=t(0.012),
             coeff_fonte_decouver=t(0.016),
             seuil_fonte_conifers=t(0.0), seuil_fonte_feuillus=t(0.0),
             seuil_fonte_decouver=t(0.0),
             taux_fonte_geo=t(0.5), densite_max=t(466.0), constante_tassement=t(0.1))
    p.update(extra)
    return p


def _hiver_puis_fonte(p, n=4, jours=range(330, 366)):
    """Accumule 60 mm puis rechauffe : retourne (apport total, stock final)."""
    s = init_state(n, dtype=torch.float64)
    tmin, tmax = torch.full((n,), -12.0, dtype=torch.float64), torch.full((n,), -4.0, dtype=torch.float64)
    zero = torch.zeros(n, dtype=torch.float64)
    apports = 0.0
    for j in list(jours):
        neige = torch.full((n,), 4.0, dtype=torch.float64) if j % 3 == 0 else zero
        # REDOUX DOUX (tmax +2), pas des degel complets : la fonte doit etre limitee
        # par le COEFFICIENT et non par le stock, sinon tout fond quel que soit le
        # facteur et le test ne mesure rien (premiere version : degels a +6, stock
        # nul dans tous les cas, test aveugle).
        chaud = (j % 5 == 0)
        # le jour julien arrive en TENSEUR, comme la colonne le fournit (doy_t)
        jt = torch.tensor(float(j), dtype=torch.float64)
        a, s = DegreJourModifie(24)(tmin + (11 if chaud else 0), tmax + (6 if chaud else 0),
                                    zero, neige, jt, s, p)
        apports = apports + a
    return apports, s["couvert_nival_mm"]


def test_sans_amplitude_le_clone_est_identique_au_bit_pres():
    """LE contrat : l'ajout du mecanisme ne doit RIEN changer tant qu'il n'est pas
    demande. La validation per-UHRH contre le binaire C++ repose la-dessus."""
    a0, st0 = _hiver_puis_fonte(_params())
    a1, st1 = _hiver_puis_fonte(_params(melt_seasonal_amp=None))
    assert torch.equal(a0, a1) and torch.equal(st0, st1)


def test_amplitude_zero_egale_absence():
    """amp=0.0 doit reproduire le clone fidele (sinus multiplie par zero)."""
    a0, st0 = _hiver_puis_fonte(_params())
    a1, st1 = _hiver_puis_fonte(_params(melt_seasonal_amp=0.0))
    assert torch.allclose(a0, a1) and torch.allclose(st0, st1)


def test_en_decembre_l_amplitude_RETIENT_la_neige():
    """Le but mesurable : moins de fonte en decembre => plus de stock au sol.

    C'est la cible CanSWE (11=0.66, 12=0.55 a remonter vers 1.0), pas un effet de bord :
    si ce test casse, la modulation ne sert plus a rien meme si elle est jolie.
    """
    _, st_plat = _hiver_puis_fonte(_params())
    _, st_mod = _hiver_puis_fonte(_params(melt_seasonal_amp=0.5))
    assert (st_mod > st_plat).all(), "en decembre le facteur vaut ~0.5x : le stock doit monter"


def test_en_juin_l_amplitude_ACCELERE_la_fonte():
    """Symetrie du sinus : maximum au solstice d'ete. Verifie sur un manteau pose en
    juin, ou le facteur vaut ~1.5x a amp=0.5."""
    p_plat, p_mod = _params(), _params(melt_seasonal_amp=0.5)
    n = 4
    fontes = {}
    for nom, p in (("plat", p_plat), ("mod", p_mod)):
        s = init_state(n, dtype=torch.float64)
        zero = torch.zeros(n, dtype=torch.float64)
        # poser 80 mm de manteau autour du jour 160, puis rechauffer
        for j in range(155, 160):
            _, s = DegreJourModifie(24)(zero - 8, zero - 2, zero,
                                        torch.full((n,), 16.0, dtype=torch.float64),
                                        torch.tensor(float(j), dtype=torch.float64), s, p)
        a, _ = DegreJourModifie(24)(zero + 4, zero + 14, zero, zero,
                                    torch.tensor(170.0, dtype=torch.float64), s, p)
        fontes[nom] = a
    assert (fontes["mod"] > fontes["plat"]).all()


def test_moyenne_annuelle_du_facteur_vaut_un():
    """La modulation redistribue la fonte dans l'annee sans changer le taux MOYEN :
    le calage regional ancre (taux med 7.2/8.5/10.1 sur OUTV) reste le taux moyen."""
    j = torch.arange(1, 366, dtype=torch.float64)
    s = 1.0 + 0.5 * torch.sin(2.0 * torch.pi * (j - 81.0) / 365.0)
    assert abs(float(s.mean()) - 1.0) < 5e-3
    # extremes aux solstices
    assert abs(float(s[171]) - 1.5) < 0.01, "21 juin : maximum"
    assert abs(float(s[354]) - 0.5) < 0.01, "21 decembre : minimum"


# ── Partage pluie-neige au bulbe humide (generalisation de R35) ──────────────

def _split(col_attrs, P=10.0, tmin=-1.0, tmax=2.0, ea=None, n=4):
    from meandre.vertical.hydrotel_column import HydrotelColumn
    col = HydrotelColumn.__new__(HydrotelColumn)   # pas besoin du reste de la colonne
    col.t_neige_seuil = col_attrs.get("seuil", 0.0)
    if "mode" in col_attrs:
        col.split_mode = col_attrs["mode"]
    t = lambda v: torch.full((n,), float(v))
    ea_t = None if ea is None else t(ea)
    return col._split_precip(t(P), t(tmin), t(tmax), ea=ea_t)


def test_sans_mode_le_partage_air_est_identique():
    """Fidelite par defaut : sans split_mode, ea est ignore et rien ne change."""
    p0, n0 = _split({"seuil": 0.0})
    p1, n1 = _split({"seuil": 0.0}, ea=0.4)
    assert torch.equal(p0, p1) and torch.equal(n0, n1)


def test_air_sec_fait_plus_de_neige():
    """Le coeur de la physique : a meme temperature, l'air sec refroidit le flocon
    par evaporation et le fait survivre plus chaud. HR 50 % doit donner plus de
    neige que HR ~100 %."""
    T = 1.5
    es = 0.6108 * torch.exp(torch.tensor(17.27 * T / (T + 237.3)))
    _, n_sec = _split({"seuil": -0.8, "mode": "wet_bulb"},
                      tmin=T - 1, tmax=T + 1, ea=float(es) * 0.5)
    _, n_hum = _split({"seuil": -0.8, "mode": "wet_bulb"},
                      tmin=T - 1, tmax=T + 1, ea=float(es) * 0.98)
    assert float(n_sec[0]) > float(n_hum[0])


def test_air_sature_retombe_sur_le_seuil_air():
    """A saturation Twb = T : le mode bulbe humide doit se comporter comme un seuil
    AIR de meme valeur (a la precision de Stull pres). C'est la limite qui rend les
    deux modes comparables."""
    T = 0.5
    es = 0.6108 * torch.exp(torch.tensor(17.27 * T / (T + 237.3)))
    p_wb, n_wb = _split({"seuil": 0.0, "mode": "wet_bulb"},
                        tmin=T - 2, tmax=T + 2, ea=float(es))
    p_air, n_air = _split({"seuil": 0.0}, tmin=T - 2, tmax=T + 2)
    assert abs(float(n_wb[0]) - float(n_air[0])) < 0.6, "Stull vaut ~0.3 degre pres"
