"""Clone FIDÈLE de la fonte de neige DEGRÉ-JOUR MODIFIÉ d'Hydrotel, porté
ligne-à-ligne du C++ (source/degre_jour_modifie.cpp) vers PyTorch, vectorisé sur
les nœuds, différentiable. Sous-projet PROPRE, indépendant du reste de méandre.

Modèle d'Hydrotel SLSO (simulation.csv : FONTE DE NEIGE;DEGRE JOUR MODIFIE).
PAS un simple degré-jour : bilan d'énergie avec contenu de chaleur (cold
content), perte convective (diffusion thermique, fonction d'erreur), chaleur de
la pluie / géothermique / radiation, albédo évolutif, compaction, rétention
d'eau liquide. La fonte degré-jour est modulée par l'INDICE DE RADIATION
(géométrie solaire pente/exposition) et atténuée par (1 − albédo).

Calculé séparément sur 3 classes d'occupation (conifères / feuillus / découvert)
puis agrégé par pourcentage de couverture (CalculeFonte appelée 3× par UHRH,
degre_jour_modifie.cpp:634-673 ; agrégation l.705-730).

Source C++ : CalculeFonte (l.1241), CalculIndiceRadiation (l.1030),
CalculDensiteNeige (util.cpp:771), ConductiviteNeige (l.2044), Erf (l.2061).

Unités internes = celles d'Hydrotel : stock/hauteur/eau en m, chaleur en J/m²
(densité surfacique d'énergie), température en °C, pas de temps en heures.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor

# ── Constantes (constantes.hpp) ──
DENSITE_EAU = 1000.0        # kg/m3
CHALEUR_FONTE = 335000.0    # J/kg (chaleur latente de fusion)
CHALEUR_NEIGE = 2093.4      # J/kg/K (chaleur spécifique neige)
CHALEUR_EAU = 4184.0        # J/kg/K (chaleur spécifique eau)
RAD1 = 57.295779513         # 1 radian en degré
dRAD1 = 57.295779513078550
CONSTANTE_SOLAIRE = 1360.8  # W/m2 (2008)
EXENTRICITE = 0.01671022    # excentricité orbite terrestre
DEG1 = 58.1313429643110     # 1 jour en degré


def calcul_densite_neige(temperature: Tensor) -> Tensor:
    """Densité de la neige fraîche [kg/m3] (util.cpp:771). T en °C."""
    rho = 151.0 + 10.63 * temperature + 0.2767 * temperature * temperature
    rho = torch.where(temperature < -17.0, torch.full_like(temperature, 50.0), rho)
    rho = torch.where(temperature > 0.0, torch.full_like(temperature, 150.0), rho)
    return rho


def conductivite_neige(densite: Tensor) -> Tensor:
    """Conductivité thermique de la neige (degre_jour_modifie.cpp:2044). densite [kg/m3]."""
    d0, d1, d2, d3, d4 = 0.36969, 1.58688e-03, 3.02462e-06, 5.19756e-09, 1.56984e-11
    p0 = 1.0
    p1 = densite - 329.6
    p2 = (densite - 260.378) * p1 - 21166.4 * p0
    p3 = (densite - 320.69) * p2 - 24555.8 * p1
    p4 = (densite - 263.363) * p3 - 11739.3 * p2
    return d0 * p0 + d1 * p1 + d2 * p2 + d3 * p3 + d4 * p4


def erf_hydrotel(x: Tensor) -> Tensor:
    """Fonction d'erreur, approximation rationnelle Abramowitz-Stegun 7.1.25
    (degre_jour_modifie.cpp:2061). Requiert x >= 0 (vrai ici : hneige/(2√(αt))>0)."""
    t = 1.0 / (1.0 + 0.47047 * x)
    return 1.0 - (0.3480242 * t - 0.0958798 * t * t + 0.7478556 * t * t * t) * torch.exp(-x * x)


def init_ce(lat_dd: Tensor, pente: Tensor, orientation: Tensor) -> tuple[Tensor, Tensor]:
    """_ce1 (pente effective, deg) et _ce0 (azimut effectif, deg) pour l'indice de
    radiation (Initialise, degre_jour_modifie.cpp:149-155). lat en degrés, pente en
    rad (tan stockée comme pente ? non : zone.PrendrePente() = tangente de pente),
    orientation = code 0-7 (45° par pas). Statiques par nœud."""
    theta = lat_dd / RAD1
    k = torch.atan(pente)
    h = ((495.0 - orientation * 45.0) % 360.0) / RAD1
    ce1 = torch.asin(torch.sin(k) * torch.cos(h) * torch.cos(theta)
                     + torch.cos(k) * torch.sin(theta)) * RAD1
    ce0 = torch.atan(torch.sin(h) * torch.sin(k)
                     / (torch.cos(k) * torch.cos(theta)
                        - torch.cos(h) * torch.sin(k) * torch.sin(theta))) * RAD1
    return ce1, ce0


def indice_radiation(lat_dd: Tensor, ce1: Tensor, ce0: Tensor, jour: Tensor,
                     pas_de_temps: int = 24) -> Tensor:
    """Indice de radiation (degre_jour_modifie.cpp:1030), pas JOURNALIER (24h, la
    branche sous-journalière est sautée). jour = jour julien. Retourne |i_j2/i_j1|."""
    i0 = CONSTANTE_SOLAIRE
    w = 15.0 / dRAD1
    theta = lat_dd / dRAD1
    theta1 = ce1 / dRAD1
    alpha = ce0 / dRAD1
    e2 = (1.0 - EXENTRICITE * torch.cos((jour - 4.0) / DEG1)) ** 2
    i_e2 = i0 / e2
    decli = 0.410152374218 * torch.sin((jour - 80.25) / DEG1)

    # demi-durée du jour, surface horizontale
    tampon_h = -torch.tan(theta) * torch.tan(decli)
    duree_hor = torch.acos(torch.clamp(tampon_h, -1.0, 1.0)) / w
    duree_hor = torch.where(tampon_h > 1.0, torch.zeros_like(duree_hor), duree_hor)
    duree_hor = torch.where(tampon_h < -1.0, torch.full_like(duree_hor, 12.0), duree_hor)
    # durée du jour, surface en pente
    tampon_p = -torch.tan(theta1) * torch.tan(decli)
    duree_pte = torch.acos(torch.clamp(tampon_p, -1.0, 1.0)) / w
    duree_pte = torch.where(tampon_p > 1.0, torch.zeros_like(duree_pte), duree_pte)
    duree_pte = torch.where(tampon_p < -1.0, torch.full_like(duree_pte, 12.0), duree_pte)

    t1_pte = -duree_pte - alpha / w
    t2_pte = duree_pte - alpha / w
    t1_pte = torch.maximum(t1_pte, -duree_hor)
    t2_pte = torch.minimum(t2_pte, duree_hor)

    t1_hor_sim = -duree_hor
    t2_hor_sim = duree_hor

    # ensoleillement surface horizontale
    i_j1 = 3600.0 * i_e2 * ((t2_hor_sim - t1_hor_sim) * torch.sin(theta) * torch.sin(decli)
                            + 1.0 / w * torch.cos(theta) * torch.cos(decli)
                            * (torch.sin(w * t2_hor_sim) - torch.sin(w * t1_hor_sim)))
    i_j1 = torch.where(t1_hor_sim > t2_hor_sim, torch.zeros_like(i_j1), i_j1)
    # ensoleillement surface en pente
    i_j2 = 3600.0 * i_e2 * ((t2_pte - t1_pte) * torch.sin(theta1) * torch.sin(decli)
                            + 1.0 / w * torch.cos(theta1) * torch.cos(decli)
                            * (torch.sin(w * t2_pte + alpha) - torch.sin(w * t1_pte + alpha)))
    i_j2 = torch.where(t1_pte > t2_pte, torch.zeros_like(i_j2), i_j2)

    return torch.where(i_j1 != 0.0, torch.abs(i_j2 / i_j1), torch.ones_like(i_j1))


def calcule_fonte(tmin, tmax, pluie_m, neige_m, indice_rad,
                  stock, hauteur, chaleur, eau_retenue, albedo,
                  coeff_fonte, seuil_fonte, taux_fonte_geo, densite_max,
                  constante_tassement, pas_de_temps=24, methode_albedo=1):
    """CalculeFonte (degre_jour_modifie.cpp:1241) pour UNE classe d'occupation, un
    pas de temps, vectorisé. Tout en m / °C / heures. coeff_fonte déjà en m/°C/jour
    (taux mm/jour /1000). Retourne (fonte_m, stock, hauteur, chaleur, eau_retenue,
    albedo) — fonte_m = lame d'eau libérée (avant pondération par la classe)."""
    pdts = pas_de_temps * 3600
    temperature_moyenne = (tmin + tmax) / 2.0

    has_snow = (stock > 0.0) | (neige_m > 0.0)

    # hauteur epsilon si stock>0 et hauteur==0 (l.1252)
    hauteur = torch.where((stock > 0.0) & (hauteur == 0.0),
                          torch.full_like(hauteur, 1e-11), hauteur)

    drel = calcul_densite_neige(temperature_moyenne) / DENSITE_EAU

    # ajout neige fraîche (l.1270-1272). NB : Hydrotel stocke la neige en HAUTEUR
    # (grille_meteo.cpp:800 ChangeNeige(SWE/densité)), puis stock += hauteur·drel =
    # SWE. Ici neige_m est donné en SWE (= neige.csv « EEN »), donc hauteur =
    # neige_m/drel et stock += neige_m directement (le drel s'annule, fidèle au C++).
    neige_depth = neige_m / torch.clamp(drel, min=1e-9)
    stock_n = stock + neige_m
    hauteur_n = hauteur + neige_depth
    chaleur_n = chaleur + neige_m * DENSITE_EAU * CHALEUR_NEIGE * temperature_moyenne

    dennei = stock_n / torch.clamp(hauteur_n, min=1e-12)

    # perte de chaleur par convection si T < seuil (l.1278-1289)
    cold = temperature_moyenne < seuil_fonte
    tneige_cc = chaleur_n / torch.clamp(stock_n * CHALEUR_NEIGE * DENSITE_EAU, min=1e-12)
    hneige = torch.where(hauteur_n < 0.4, 0.5 * hauteur_n, 0.2 + 0.25 * (hauteur_n - 0.4))
    alpha_c = conductivite_neige(dennei * DENSITE_EAU) / torch.clamp(
        dennei * DENSITE_EAU * CHALEUR_NEIGE, min=1e-12)
    erf_arg = hneige / (2.0 * torch.sqrt(torch.clamp(alpha_c * pdts, min=1e-12)))
    erf_v = erf_hydrotel(torch.clamp(erf_arg, min=0.0))
    tneige_new = temperature_moyenne + (tneige_cc - temperature_moyenne) * erf_v
    chaleur_conv = tneige_new * stock_n * DENSITE_EAU * CHALEUR_NEIGE
    chaleur_n = torch.where(cold, chaleur_conv, chaleur_n)

    # ajustement eau retenue au pas précédent (l.1292)
    chaleur_n = chaleur_n + eau_retenue * DENSITE_EAU * CHALEUR_FONTE
    # ajout pluie (l.1295-1296)
    stock_n = stock_n + pluie_m
    chaleur_n = chaleur_n + pluie_m * DENSITE_EAU * (CHALEUR_FONTE + CHALEUR_EAU * temperature_moyenne)
    # chaleur géothermique (l.1299) : taux_fonte_geo en mm/jour
    chaleur_n = chaleur_n + (taux_fonte_geo * pas_de_temps / 24.0) / 1000.0 * DENSITE_EAU * CHALEUR_FONTE

    # albédo méthode 1 (l.1309-1343). eq_neige/st_neige en mm SWE (neige_m = SWE).
    eq_neige = neige_m * 1000.0
    st_neige = (stock_n - neige_m) * 1000.0
    liquide = ((pluie_m > 0.0) | (tneige_new >= 0.0)).to(stock.dtype)
    one_m_exp_eq = 1.0 - torch.exp(-0.5 * eq_neige)
    alb_t1 = one_m_exp_eq * 0.8 + (1.0 - one_m_exp_eq) * (
        0.5 + (albedo - 0.5) * torch.exp(-0.2 * pas_de_temps / 24.0 * (1.0 + liquide)))
    beta2 = torch.where(albedo < 0.5, torch.full_like(albedo, 0.2), 0.2 + (albedo - 0.5))
    alb_avec = (1.0 - torch.exp(-beta2 * st_neige)) * alb_t1 + (
        1.0 - (1.0 - torch.exp(-beta2 * st_neige))) * 0.15
    alb_sans = one_m_exp_eq * 0.8 + (1.0 - one_m_exp_eq) * 0.15
    albedo_new = torch.where(st_neige > 0.0, alb_avec, alb_sans)
    if methode_albedo == 1:
        albedo = albedo_new

    # fonte par radiation degré-jour (l.1347-1351)
    fonte = torch.where(temperature_moyenne > seuil_fonte,
                        coeff_fonte * (temperature_moyenne - seuil_fonte) * indice_rad * (1.0 - albedo),
                        torch.zeros_like(stock_n))
    fonte = fonte * (pas_de_temps / 24.0)
    chaleur_n = chaleur_n + fonte * DENSITE_EAU * CHALEUR_FONTE

    # compaction (l.1354-1368)
    compaction = hauteur_n * (constante_tassement * (pas_de_temps / 24.0)) * (
        1.0 - dennei / densite_max * 1000.0)
    compaction = torch.clamp(compaction, min=0.0)
    hauteur_n = hauteur_n - compaction
    densto = stock_n / torch.clamp(hauteur_n, min=1e-12)
    over_dens = densto * 1000.0 > densite_max
    densto = torch.where(over_dens, densite_max / 1000.0, densto)
    hauteur_n = torch.where(over_dens, stock_n / torch.clamp(densto, min=1e-12), hauteur_n)

    # surplus calorifique → fonte (l.1371-1386)
    chaud = chaleur_n > 0.0
    fonte_surplus = chaleur_n / CHALEUR_FONTE / DENSITE_EAU
    fonte_surplus = torch.minimum(fonte_surplus, stock_n)
    fonte = torch.where(chaud, fonte_surplus, torch.zeros_like(fonte))
    stock_n = torch.where(chaud, stock_n - fonte, stock_n)
    fonte_moins_pluie = fonte - pluie_m
    hauteur_n = torch.where(chaud & (fonte_moins_pluie > 0.0),
                            hauteur_n - fonte_moins_pluie / torch.clamp(densto, min=1e-12), hauteur_n)
    hauteur_n = torch.where(chaud & (hauteur_n <= 0.0),
                            stock_n / torch.clamp(densto, min=1e-12), hauteur_n)
    chaleur_n = torch.where(chaud, chaleur_n - fonte * DENSITE_EAU * CHALEUR_FONTE, chaleur_n)

    # réinit si stock négligeable (l.1389-1395)
    vide = stock_n < 0.0001
    stock_n = torch.where(vide, torch.zeros_like(stock_n), stock_n)
    hauteur_n = torch.where(vide, torch.zeros_like(hauteur_n), hauteur_n)
    chaleur_n = torch.where(vide, torch.zeros_like(chaleur_n), chaleur_n)
    eau_ret_new = torch.where(vide, torch.zeros_like(eau_retenue), eau_retenue)

    # eau retenue dans le stock (l.1397-1410)
    rmax = (0.1 * dennei) * stock_n
    retient = rmax > fonte
    stock_n = torch.where(retient, stock_n + fonte, stock_n + rmax)
    eau_ret_new = torch.where(vide, eau_ret_new,
                              torch.where(retient, fonte, rmax))
    fonte = torch.where(retient, torch.zeros_like(fonte), fonte - rmax)

    # branche sans neige : apport = pluie (l.1415-1418), états inchangés
    fonte_out = torch.where(has_snow, fonte, pluie_m)
    stock_out = torch.where(has_snow, stock_n, stock)
    hauteur_out = torch.where(has_snow, hauteur_n, hauteur)
    chaleur_out = torch.where(has_snow, chaleur_n, chaleur)
    eau_ret_out = torch.where(has_snow, eau_ret_new, eau_retenue)
    albedo_out = torch.where(has_snow, albedo, albedo)
    return fonte_out, stock_out, hauteur_out, chaleur_out, eau_ret_out, albedo_out


class DegreJourModifie(torch.nn.Module):
    """Fonte degré-jour modifié, 3 classes (conifères/feuillus/découvert), un pas
    de temps journalier, vectorisé sur les nœuds. État = 4 tenseurs par classe
    (stock, hauteur, chaleur, eau_retenue) + albédo par classe."""

    CLASSES = ("conifers", "feuillus", "decouver")

    def __init__(self, pas_de_temps: int = 24):
        super().__init__()
        self.pas_de_temps = pas_de_temps

    def forward(self, tmin, tmax, pluie_mm, neige_mm, jour, state, p):
        """Un pas de temps. tmin/tmax °C ; pluie/neige mm ; jour = jour julien ;
        state = dict {classe: (stock,hauteur,chaleur,eau_ret), 'albedo_'+classe};
        p = dict params par nœud (lat, ce1, ce0, pct_conifers/feuillus/autres,
        coeff_fonte_*, seuil_fonte_*, taux_fonte_geo, densite_max, constante_tassement).
        Retourne apport_mm (mm) et nouveau state."""
        ir = indice_radiation(p["lat"], p["ce1"], p["ce0"], jour, self.pas_de_temps)
        pluie_m = pluie_mm / 1000.0
        neige_m = neige_mm / 1000.0
        new_state = {}
        apport = torch.zeros_like(pluie_m)
        stock_moyen = torch.zeros_like(pluie_m)
        pct = {"conifers": p["pct_conifers"], "feuillus": p["pct_feuillus"], "decouver": p["pct_autres"]}
        for c in self.CLASSES:
            st, ha, ch, er = state[c]
            alb = state["albedo_" + c]
            fonte, st2, ha2, ch2, er2, alb2 = calcule_fonte(
                tmin, tmax, pluie_m, neige_m, ir, st, ha, ch, er, alb,
                p["coeff_fonte_" + c], p["seuil_fonte_" + c], p["taux_fonte_geo"],
                p["densite_max"], p["constante_tassement"], self.pas_de_temps)
            new_state[c] = (st2, ha2, ch2, er2)
            new_state["albedo_" + c] = alb2
            apport = apport + pct[c] * fonte
            stock_moyen = stock_moyen + pct[c] * st2
        apport_mm = torch.clamp(apport * 1000.0, min=0.0)
        new_state["couvert_nival_mm"] = stock_moyen * 1000.0
        return apport_mm, new_state


def init_state(n_nodes, device="cpu", dtype=torch.float64):
    """État neige initial vide (pas de neige) pour n_nodes."""
    z = lambda: torch.zeros(n_nodes, device=device, dtype=dtype)
    s = {}
    for c in DegreJourModifie.CLASSES:
        s[c] = (z(), z(), z(), z())
        s["albedo_" + c] = z()
    return s
