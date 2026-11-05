"""Clone FIDÈLE du bilan vertical BV3C2 d'Hydrotel, porté ligne-à-ligne du C++
(github INRS hydrotel, source/bv3c2.cpp) vers PyTorch, vectorisé sur les nœuds,
différentiable. Sous-projet PROPRE, indépendant du reste de méandre.

Source C++ : CalculeRuisselement (l.2175), TriCoucheOct97 (l.1841),
CalculeUHRH (l.751), ConductiviteHydrolique (l.1807), CalculePsi (l.1819).

Mécanismes clés (que le portage précédent ratait) :
  1. RUISSELLEMENT de surface = lruis·fsa + leau·fse + lprec·fsi
     - fsa = fraction perméable (sol) : ruissellement limité par pinf
     - fse = fraction EAU (lacs) : (pluie − ET) ruisselle direct
     - fsi = fraction IMPERMÉABLE : 100 % de la pluie ruisselle
     C'est ça qui génère des pics au pas JOURNALIER sans intensité horaire.
  2. Excès de SATURATION : quand le profil se remplit (couches profondes
     saturent), le surplus REFOULE vers le haut et déborde en couche 1 →
     ruissellement (l.2113 ruis += (theta1−thetas)·z1/dtc).
  3. pinf = min(prec, ks), avec portes GEL (sol gelé+neige<10 → pinf=0) et
     SATURATION (theta1=thetas → pinf=0). Hortonien quasi nul au pas journalier
     (prec=apport/24 < ks), donc les pics viennent de 1 et 2, pas du hortonien.

Unités internes = celles d'Hydrotel : m, m/h, dt en HEURES (24 = journalier).
"""
from __future__ import annotations
import os
import torch

from meandre.utils import settings
from torch import Tensor

# Table proprietehydrolique.sol (Hydrotel) — thetas, thetacc, thetapf, ks(m/h), psis(m), lambda
SOIL_TEXTURES = {
    "sand": dict(thetas=0.417, thetacc=0.091, thetapf=0.033, ks=0.2100, psis=0.1598, lam=0.694),
    "loamy_sand": dict(thetas=0.401, thetacc=0.125, thetapf=0.055, ks=0.0611, psis=0.2058, lam=0.553),
    "sandy_loam": dict(thetas=0.412, thetacc=0.207, thetapf=0.095, ks=0.0259, psis=0.302, lam=0.378),
    "loam": dict(thetas=0.434, thetacc=0.270, thetapf=0.117, ks=0.0132, psis=0.4012, lam=0.252),
    "silt_loam": dict(thetas=0.486, thetacc=0.330, thetapf=0.133, ks=0.0068, psis=0.5087, lam=0.234),
    "clay": dict(thetas=0.385, thetacc=0.396, thetapf=0.272, ks=0.0006, psis=0.856, lam=0.165),
    "peat": dict(thetas=0.930, thetacc=0.275, thetapf=0.050, ks=1.008, psis=0.0103, lam=0.370),
}
EPAISSEUR = (0.21941, 0.15725, 2.65)   # z11, z22, z33 (m), bv3c.csv SLSO
KREC_DEFAULT = 1.2869e-7               # m/h
import os as _os
_DTC_DETACHE = settings.get("debug.detach_dtc")
# Diagnostic (2026-09-26) : MEANDRE_CASCADE_DETACHE=1 detache les quantites deplacees par la
# cascade de saturation et la correction de negativite, pour mesurer leur part dans
# l'explosion du gradient a travers l'humidite du sol. Lu a chaque appel.


_SEMI_IMPLICITE = settings.get("column.semi_implicit_soil")


def _cd(x):
    return x.detach() if settings.get("debug.detach_cascade") else x


CIN_DEFAULT = 0.03                     # VARIATION MAXIMALE (Courant)
DT_H = 24.0                            # pas de temps (heures)


def campbell_K(theta, thetas, ks, b):
    """ConductiviteHydrolique (l.1807) : K = Ks·omega^(2b+3), omega=max(theta/thetas,0.05)."""
    omega = torch.clamp(theta / (thetas + 1e-9), min=0.05)
    return ks * omega.pow(2.0 * b + 3.0)


def campbell_psi(theta, thetas, psis, b, omegpi, mm, nn):
    """CalculePsi (l.1819) : puissance sous omegpi, spline quadratique au-dessus."""
    omega = torch.clamp(theta / (thetas + 1e-9), min=0.05)
    psi_pow = psis * omega.pow(-b)
    psi_spline = -mm * (omega - nn) * (omega - 1.0)
    return torch.where(omega < omegpi, psi_pow, psi_spline)


def vg_K(theta, thetas, ks, n, theta_r):
    """Conductivité de Mualem-van Genuchten : K = Ks·Se^½·(1 − (1 − Se^(1/m))^m)², m = 1 − 1/n."""
    m = 1.0 - 1.0 / n
    se = torch.clamp((theta - theta_r) / torch.clamp(thetas - theta_r, min=1e-6), 1e-3, 1.0)
    return ks * se.sqrt() * (1.0 - (1.0 - se.pow(1.0 / m)).clamp(min=0.0).pow(m)).pow(2.0)


def vg_psi(theta, thetas, n, theta_r, alpha):
    """Succion de van Genuchten (m, positive) : psi = (Se^(−1/m) − 1)^(1/n) / alpha."""
    m = 1.0 - 1.0 / n
    se = torch.clamp((theta - theta_r) / torch.clamp(thetas - theta_r, min=1e-6), 1e-3, 1.0)
    return (se.pow(-1.0 / m) - 1.0).clamp(min=0.0).pow(1.0 / n) / alpha


PSI_FC_M = 33.0 / 9.80665      # capacité au champ : succion de 33 kPa, en mètres d'eau
PSI_WP_M = 1500.0 / 9.80665    # point de flétrissement : 1 500 kPa


def vg_through(thetas, theta_fc, theta_wp, n, iterations=60):
    """Teneur résiduelle et alpha de van Genuchten pour que la courbe de rétention passe par la
    capacité au champ et le point de flétrissement du tronçon, n donné.

    Le rapport des succions aux deux points ne dépend que de theta_r : il croît de sa valeur à
    theta_r = 0 vers l'infini quand theta_r approche le point de flétrissement. On le cherche par
    dichotomie ; si le rapport à theta_r = 0 dépasse déjà la cible, theta_r = 0. Alpha s'en déduit
    à la capacité au champ. Aucun paramètre ajusté : les deux points sont ceux du sol.
    """
    m = 1.0 - 1.0 / n
    target = PSI_WP_M / PSI_FC_M

    def x(theta, theta_r):
        se = torch.clamp((theta - theta_r) / torch.clamp(thetas - theta_r, min=1e-6), 1e-6, 1.0)
        return (se.pow(-1.0 / m) - 1.0).clamp(min=1e-12).pow(1.0 / n)

    with torch.no_grad():
        lo = torch.zeros_like(theta_wp)
        hi = theta_wp * 0.999
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            above = x(theta_wp, mid) / x(theta_fc, mid) > target
            hi = torch.where(above, mid, hi)
            lo = torch.where(above, lo, mid)
        theta_r = torch.where(x(theta_wp, torch.zeros_like(lo)) / x(theta_fc, torch.zeros_like(lo)) > target, torch.zeros_like(lo), 0.5 * (lo + hi))
    alpha = x(theta_fc, theta_r) / PSI_FC_M
    return theta_r, alpha


def init_spline_coeffs(b):
    """omegpi, mm, nn pour le raccord C1 de psi (bv3c2.cpp Initialise, ~l.400-416).
    omegpi=(1+2b)/(2+2b). À omegpi, psi_pow et sa dérivée doivent égaler la spline
    psi=-mm(omega-nn)(omega-1) avec psi(1)=0."""
    omegpi = (1.0 + 2.0 * b) / (2.0 + 2.0 * b)
    # NOTE Hydrotel calcule mm, nn à l'init depuis psis,b ; on reconstruit le
    # raccord C1 : psi_pow(omegpi)=psis·omegpi^-b ; dpsi=-psis·b·omegpi^(-b-1).
    # Pour la spline psi=-mm(omega-nn)(omega-1) : psi(1)=0 ok ;
    # psi(omegpi)=-mm(omegpi-nn)(omegpi-1) ; psi'(omega)=-mm(2omega-nn-1).
    # On résout mm, nn pour valeur+dérivée continues en omegpi.
    return omegpi


class BV3C2Clone(torch.nn.Module):
    """Bilan vertical BV3C2 fidèle, un pas de temps (journalier), vectorisé."""

    def __init__(self, n_substep=48, static=False, frozen_gate_continuous=False):
        super().__init__()
        self.n_substep = n_substep   # sous-pas internes (le C++ adapte ; on fixe)
        # EXP-2 : porte de gel CONTINUE sur l'infiltration. Le clone fidèle coupe
        # pinf=0 dès frost>0 (binaire). Ici pinf = min(prec,ks1)·(1−froz_frac),
        # cohérent avec le drainage déjà continu (throttle=1−froz_frac). Plus
        # physique (gel partiel → infiltration partielle), et relève potentiellement
        # les pics de freshet (gel → plus de ruissellement de fonte). Opt-in.
        self.frozen_gate_continuous = bool(frozen_gate_continuous)
        # Porte SURFACIQUE (2026-09-28), à la manière de Koren et al. (1999) : la part gelée
        # de la couche de surface est la part IMPERMÉABLE de l'aire, et la pluie qui y tombe
        # ruisselle. La porte continue, qui réduit la conductivité, ne mord presque jamais
        # parce que la conductivité saturée dépasse de loin l'intensité d'une pluie d'hiver.
        self.frozen_gate_areal = False
        # Reglages relus a la construction : les constantes de module ci-dessus sont evaluees
        # a l'import, avant que le pilote ne lise son fichier de configuration.
        self.semi_implicite = settings.get("column.semi_implicit_soil")
        # static=True : pas de break data-dépendant (les itérations no-op après
        # convergence sont idempotentes, dtc=0). Permet torch.compile (pas de
        # synchro GPU `bool().all()` par itération, boucle statique fusionnable).
        # Résultats IDENTIQUES au mode break à n_substep égal. À réserver à un
        # n_substep modéré (ex. 48) — PAS à la validation n_substep=1500.
        self.static = static

    def forward(self, theta1, theta2, theta3, apport_mm, etp_mm,
                frozen_depth_cm, swe_mm, p, etr1_mm=None, etr2_mm=None, etr3_mm=None):
        """Un pas de temps BV3C2. theta en m3/m3, apport/etp en mm/jour.
        p : dict de tenseurs par nœud (thetas/ks/psis/b/omegpi/mm/nn par couche,
        z1/z2/z3, slope, krec, cin, fsa/fse/fsi, coef_recharge).
        Retourne prod_surf, prod_hypo, prod_base (mm), recharge (mm), thetas new."""
        eps = 1e-9
        z1, z2, z3 = p["z1"], p["z2"], p["z3"]
        ths1, ths2, ths3 = p["thetas1"], p["thetas2"], p["thetas3"]
        ks1, ks2, ks3 = p["ks1"], p["ks2"], p["ks3"]
        # Ruissellement pris sur l'apport AVANT le sol (aire contributive, processus « runoff »
        # declares). Processus de la fraction SOL (fsa) : les fractions eau (fse) et impermeable
        # (fsi) recoivent l'apport TOTAL.
        runoff_before_soil = torch.zeros_like(apport_mm)
        apport_total_mm = apport_mm
        # ── AIRE CONTRIBUTIVE VARIABLE (2026-10-16, R386, opt-in MEANDRE_VSA_B) ──
        # Une part de l'apport, croissante avec l'humidite de la couche de surface, ruisselle
        # directement : les bandes riveraines et bas de versant satures, forme VIC/TOPMODEL,
        # part = (theta1/thetas1)^b. Le clone n'avait de ruissellement de saturation que si
        # toute la couche 1 saturait, ce qui n'arrive jamais a l'echelle du troncon ; les
        # orages d'ete rendaient 0,64 du volume d'evenement observe (0,32 en Monteregie).
        # Retire de l'apport du sol, ajoute au ruissellement de fsa.
        _vsa = settings.get("column.saturated_area_exponent")
        # RUISSELLEMENT DECLARE (2026-10-30, R447) : un processus « runoff » du profil, par
        # exemple SPILL_GW, prend la place de l'aire contributive ci-dessous ; il n'ajoute aucun etat.
        _profile_runoff = p.get("soil_profile")
        _declared_runoff = [q for q in getattr(_profile_runoff, "processes", ()) if q.kind == "runoff"]
        _r_decl_out = torch.zeros_like(apport_mm)
        if _declared_runoff:
            _r_decl = torch.zeros_like(apport_mm)
            for q in _declared_runoff:
                _ctx = {"input": apport_mm / 1000.0}
                # Le stock de nappe de la veille (m), posé par la colonne quand la nappe libre
                # est activée : la manette des formes d'aire saturée comme SPILL_GW.
                if p.get("s_gw") is not None:
                    _ctx["s_gw"] = p["s_gw"]
                _r_decl = _r_decl + q.flux(_ctx)
            _r_decl = torch.minimum(_r_decl * 1000.0, apport_mm)
            _r_decl_out = _r_decl
            apport_mm = apport_mm - _r_decl
            runoff_before_soil = runoff_before_soil + _r_decl
            _vsa = None
        if _vsa is not None:
            _se1 = torch.clamp(theta1 / torch.clamp(ths1, min=1e-6), 0.0, 1.0)
            _r_vsa = apport_mm * _se1 ** float(_vsa)
            apport_mm = apport_mm - _r_vsa
            runoff_before_soil = runoff_before_soil + _r_vsa
        b1, b2, b3 = p["b1"], p["b2"], p["b3"]
        psis1, psis2, psis3 = p["psis1"], p["psis2"], p["psis3"]
        krec, slope, cin = p["krec"], p["slope"], p["cin"]
        fsa, fse, fsi = p["fsa"], p["fse"], p["fsi"]
        coef_rech = p["coef_recharge"]

        # ── CalculeRuisselement (l.2175) : prec constant ; pinf/ruis RECALCULÉS
        # CHAQUE sous-pas dans la boucle (porte theta1==thetas sur le t1 COURANT). ──
        prec = (apport_mm / 1000.0) / DT_H              # mm/j -> m/h
        # Le clone ne ferme le sol gele que sous moins de 10 mm de neige, alors que le module
        # de gel tient deja compte de l'isolation du manteau : la condition compte deux fois
        # la neige, et laisse s'infiltrer les redoux d'hiver sur sol gele (2026-09-29).
        frozen = (frozen_depth_cm > 0.0) & (swe_mm < 10.0)
        # Amplitude de la porte continue (2026-10-23) : mesurée contre la réponse observée aux pluies
        # d'automne, la porte d'amplitude 1 ruisselle deux à trois fois trop sur sol gelé sans neige.
        _gate_amp = settings.get("column.frozen_gate_amplitude")

        # ── Sous-pas internes adaptatifs (TriCoucheOct97), accumulation lames ──
        # FIDÈLE Hydrotel : PAS de clamp de flux. On laisse theta dépasser thetas
        # puis on cascade le surplus en RUISSELLEMENT (l.2113), et on gère la
        # négativité en refoulant depuis les couches voisines (l.2118-2158). La
        # stabilité vient du sous-pas ADAPTATIF Courant (|flux·dtc/(theta·z)|<cin,
        # l.1954-2031), pas d'un clamp qui tuait la saturation.
        sin_slope = torch.sin(torch.atan(slope))
        et_mh = (etp_mm / 1000.0) / DT_H
        e1 = (etr1_mm / 1000.0 / DT_H) if etr1_mm is not None else et_mh
        e2 = (etr2_mm / 1000.0 / DT_H) if etr2_mm is not None else torch.zeros_like(et_mh)
        e3 = (etr3_mm / 1000.0 / DT_H) if etr3_mm is not None else torch.zeros_like(et_mh)
        cin = p["cin"]
        t1, t2, t3 = theta1, theta2, theta3
        lruis = torch.zeros_like(t1); lhyp = torch.zeros_like(t1); lbase = torch.zeros_like(t1)
        froz_frac = torch.clamp(frozen_depth_cm / 100.0 / z1, 0.0, 1.0)
        # Le gel freine aussi le drainage de la couche 1 ; même amplitude que la porte d'infiltration.
        throttle = torch.where(frozen, torch.clamp(1.0 - _gate_amp * froz_frac, 0.0, 1.0), torch.ones_like(t1))
        tr = torch.full_like(t1, DT_H)                  # temps restant (h) par nœud
        fdtcmin = DT_H / (24.0 * 60.0 * 60.0 * 1000.0)  # _fDTCMin C++ (l.744)

        # RETENTION ET CONDUCTIVITE (2026-11-03) : Campbell par defaut (clone fidele), ou
        # Mualem-van Genuchten quand la colonne pose vg_n/vg_alpha/theta_r (column.retention).
        if p.get("vg_alpha1") is not None:
            _K1 = lambda t: vg_K(t, ths1, ks1, p["vg_n1"], p["theta_r1"])
            _K2 = lambda t: vg_K(t, ths2, ks2, p["vg_n2"], p["theta_r2"])
            _K3 = lambda t: vg_K(t, ths3, ks3, p["vg_n3"], p["theta_r3"])
            _P1 = lambda t: vg_psi(t, ths1, p["vg_n1"], p["theta_r1"], p["vg_alpha1"])
            _P2 = lambda t: vg_psi(t, ths2, p["vg_n2"], p["theta_r2"], p["vg_alpha2"])
            _P3 = lambda t: vg_psi(t, ths3, p["vg_n3"], p["theta_r3"], p["vg_alpha3"])
        else:
            _K1 = lambda t: campbell_K(t, ths1, ks1, b1)
            _K2 = lambda t: campbell_K(t, ths2, ks2, b2)
            _K3 = lambda t: campbell_K(t, ths3, ks3, b3)
            _P1 = lambda t: campbell_psi(t, ths1, psis1, b1, p["omegpi1"], p["mm1"], p["nn1"])
            _P2 = lambda t: campbell_psi(t, ths2, psis2, b2, p["omegpi2"], p["mm2"], p["nn2"])
            _P3 = lambda t: campbell_psi(t, ths3, psis3, b3, p["omegpi3"], p["mm3"], p["nn3"])

        for _ in range(self.n_substep):                 # cap d'itérations (sécurité)
            k1 = _K1(t1); k2 = _K2(t2); k3 = _K3(t3)
            ps1 = _P1(t1); ps2 = _P2(t2); ps3 = _P3(t3)
            k12 = torch.maximum(k1, k2); k23 = torch.maximum(k2, k3)
            qq12 = k12 * (2.0 * (ps2 - ps1) / (z1 + z2) + 1.0)
            qq23 = k23 * (2.0 * (ps3 - ps2) / (z2 + z3) + 1.0)
            q2 = k2 * sin_slope * z2
            # Couche 3 du clone : aucune sortie laterale, percolation lineaire en teneur en eau.
            q3_lat = torch.zeros_like(q2)
            q3 = krec * z3 * t3
            # ── PROFIL DE SOL DECLARE (opt-in, 2026-09-20) ───────────────────
            # Quand une section `[soil]` declare des processus par couche, elle REMPLACE
            # les sorties calculees ci-dessus. Une couche sans processus declare n'a pas
            # de flux : la declaration est complete, comme chez Raven, et non un ajout aux
            # branches historiques. Sans declaration, rien de ce qui precede ne change et
            # le clone reste fidele au binaire C++.
            _profile = p.get("soil_profile")
            if _profile is not None:
                _ctx2 = {"theta": t2, "theta_fc": p.get("thetacc2"), "thickness": z2,
                         "conductivity": k2, "k_sat": ks2, "sin_slope": sin_slope, "porosity": ths2,
                         "theta_wp": p.get("thetapf2"), "agri_frac": p.get("agri_frac"), "visc": p.get("visc2")}
                _ctx3 = {"theta": t3, "theta_fc": p.get("thetacc3"), "thickness": z3,
                         "conductivity": k3, "k_sat": ks3, "sin_slope": sin_slope, "porosity": ths3,
                         "theta_wp": p.get("thetapf3"), "agri_frac": p.get("agri_frac"), "visc": p.get("visc3")}
                _zero = torch.zeros_like(q2)
                _q2 = _profile.total(2, "lateral", _ctx2)
                q2 = _zero if _q2 is None else _q2
                _q3l = _profile.total(3, "lateral", _ctx3)
                q3_lat = _zero if _q3l is None else _q3l
                _q3 = _profile.total(3, "percolation", _ctx3)
                q3 = _zero if _q3 is None else _q3
            qq12 = qq12 * throttle; qq23 = qq23 * throttle; q2 = q2 * throttle
            q3_lat = q3_lat * throttle
            # Porte de gel sur le drainage profond. Hydrotel le divise par deux des que le
            # sol est gele, ce qui couvre la fonte : mesure du 2026-09-18 sur l'Outaouais,
            # la teneur en eau de L3 culmine en avril mais la recharge y atteint son
            # MINIMUM annuel, inversion qu'aucun autre terme de q3 ne peut produire. Le
            # facteur est expose pour pouvoir juger la porte ; 0.5 == fidele.
            q3 = torch.where(frozen, q3 * p.get("l3_gel_facteur", 0.5), q3)
            # Gradient disponible sous le profil : 1 quand la nappe est loin, 0 quand elle
            # affleure la base du sol. Absent du dictionnaire = clone fidele.
            if "l3_gradient" in p:
                q3 = q3 * p["l3_gradient"]
            # CalculeRuisselement (l.2191-2201) sur t1 COURANT : si t1 saturé,
            # pinf=0 → toute la pluie part en hortonien ; sinon pinf=min(prec,ks).
            omega1_sat = t1 >= (ths1 - 1e-4)
            if self.frozen_gate_areal:
                imperm = froz_frac * frozen.to(froz_frac.dtype)
                pinf = torch.where(omega1_sat, torch.zeros_like(prec), (1.0 - imperm) * torch.minimum(prec, ks1))
            elif self.frozen_gate_continuous:
                # EXP-2 : capacité d'infiltration réduite CONTINÛMENT par le gel
                # (froz_frac de Rankinen, qui tient déjà compte de l'isolation neige).
                # La reduction ne s'applique que hors manteau protecteur (moins de 10 mm de
                # neige), comme la regle d'origine : sous la neige le sol reste permeable, et
                # l'ignorer faisait perdre l'hiver en Outaouais (correlation 0,57 -> 0,33, 2026-09-28).
                infil_cap = torch.minimum(prec, ks1 * (1.0 - _gate_amp * froz_frac * frozen.to(froz_frac.dtype)))
                pinf = torch.where(omega1_sat, torch.zeros_like(prec), infil_cap)
            else:
                pinf = torch.where(frozen | omega1_sat, torch.zeros_like(prec), torch.minimum(prec, ks1))
            ruis_rate = torch.clamp(prec - pinf, min=0.0)   # m/h (hortonien)
            # ── dtc FIDÈLE C++ (l.1925-2034) : flux relatifs, test Courant, dtcTemp
            # quantifié pas/(iVal+1|+2) en min avec l'échelle {48,288,1152}. ──
            q12z = qq12 / z1; q23z = qq23 / z2; q2s = q2 / z2
            one = torch.ones_like(tr)
            dtc0 = torch.where(pinf > 0.0, torch.minimum(tr, one), tr)   # cap 1h si infiltration
            def viol(d): return (torch.abs(q12z * d) >= cin * t1) | (torch.abs((q23z + q2s) * d) >= cin * t2)
            v0 = viol(dtc0)
            # bloc 1954 : dtcTemp = min des dVal non nuls, plancher fdtcmin, quantifié.
            # Dénominateurs SÉCURISÉS (|q|→1 quand nul) : le where sélectionne 0, mais
            # autograd dérive AUSSI la branche non choisie → diviser par 0 y mettrait
            # un gradient NaN. Forward inchangé.
            zr = torch.zeros_like(t1)
            aq12 = torch.abs(q12z)
            dVal1 = torch.where((t1 != 0) & (q12z != 0), cin * t1 / torch.where(aq12 > 0, aq12, one), zr)
            dq2 = q23z + q2s; adq2 = torch.abs(dq2)
            dVal2 = torch.where((t2 != 0) & (dq2 != 0), cin * t2 / torch.where(adq2 > 0, adq2, one), zr)
            both = (dVal1 != 0) & (dVal2 != 0)
            dtcTemp = torch.where(both, torch.minimum(dVal1, dVal2), torch.where(dVal1 != 0, dVal1, dVal2))
            nonzero = dtcTemp != 0
            dtcTemp_c = torch.where(dtcTemp < fdtcmin, torch.full_like(dtcTemp, fdtcmin), dtcTemp)
            iVal = torch.floor(DT_H / torch.clamp(dtcTemp_c, min=fdtcmin))
            even = (iVal % 2.0 == 0.0)
            dtcTemp_q = torch.minimum(dtc0, DT_H / torch.where(even, iVal + 2.0, iVal + 1.0))
            bDtcMod = v0 & nonzero
            # bloc 2022 : échelle discrète {pas/48, pas/288, pas/1152} si Courant violé
            l1 = viol(dtc0)
            dtc_l = torch.where(l1, torch.full_like(dtc0, DT_H / 48.0), dtc0)
            l2 = l1 & viol(torch.full_like(dtc0, DT_H / 48.0))
            dtc_l = torch.where(l2, torch.full_like(dtc0, DT_H / 288.0), dtc_l)
            l3 = l2 & viol(torch.full_like(dtc0, DT_H / 288.0))
            dtc_l = torch.where(l3, torch.full_like(dtc0, DT_H / 1152.0), dtc_l)
            # l.2033 : if bDtcMod: dtc = min(dtc_ladder, dtcTemp_quantifié)
            dtc = torch.where(bDtcMod, torch.minimum(dtc_l, dtcTemp_q), dtc_l)
            dtc = torch.minimum(dtc, tr)                 # ne dépasse pas le temps restant
            active = (tr > 1e-7).to(dtc.dtype)
            dtc = dtc * active                           # nœuds finis : dtc=0
            # Le sous-pas est un choix NUMERIQUE (condition de Courant), pas une grandeur
            # physique : un solveur adaptatif ne derive pas son controleur de pas. Sans ce
            # detachement, le gradient passe par dtc = cin * theta / flux et explose avec
            # l'horizon (1e24 sur K_sat_1 en un an a 16 sous-pas, 2026-09-26). Opt-in par
            # MEANDRE_DTC_DETACHE=1 ; la valeur simulee est identique.
            if _DTC_DETACHE:
                dtc = dtc.detach()
            # ET BRUTE (C++ l.2036 : v_etr1 sans clamp) ; la négativité éventuelle
            # est refoulée depuis la couche du dessous (l.2118), PAS masquée.
            f1 = (pinf - qq12 - e1) / z1
            f2 = (qq12 - qq23 - e2 - q2) / z2
            f3 = (qq23 - q3 - q3_lat - e3) / z3
            if _SEMI_IMPLICITE or getattr(self, "semi_implicite", False):
                # EULER LINEAIREMENT IMPLICITE sur les echanges de Darcy entre couches
                # (2026-09-28). Le critere de Courant borne la taille des flux, pas leur
                # raideur : pres de l'equilibre et de la saturation, le flux est petit mais sa
                # derivee grande, et le pas explicite amplifie une perturbation jusqu'a cent
                # fois par jour (registre, R197). La simulation reste bornee par les
                # ecretages, pas sa derivee. On resout (I - dt A) d = dt f, ou A est la
                # jacobienne des echanges qq12 et qq23 par rapport aux teneurs, estimee par
                # differences finies et DETACHEE (methode de Rosenbrock a jacobienne
                # approchee, qui reste consistante). La ou dt A est petit, d = dt f : le
                # schema explicite du clone est retrouve.
                with torch.no_grad():
                    h = 1e-5
                    dk1 = (_K1(t1 + h) - k1) / h
                    dk2 = (_K2(t2 + h) - k2) / h
                    dk3 = (_K3(t3 + h) - k3) / h
                    dp1 = (_P1(t1 + h) - ps1) / h
                    dp2 = (_P2(t2 + h) - ps2) / h
                    dp3 = (_P3(t3 + h) - ps3) / h
                    g12 = 2.0 * (ps2 - ps1) / (z1 + z2) + 1.0
                    g23 = 2.0 * (ps3 - ps2) / (z2 + z3) + 1.0
                    c12 = 2.0 / (z1 + z2)
                    c23 = 2.0 / (z2 + z3)
                    m1 = (k1 >= k2).to(t1.dtype)
                    m2 = (k2 >= k3).to(t1.dtype)
                    a12_1 = m1 * dk1 * g12 - k12 * c12 * dp1
                    a12_2 = (1.0 - m1) * dk2 * g12 + k12 * c12 * dp2
                    a23_2 = m2 * dk2 * g23 - k23 * c23 * dp2
                    a23_3 = (1.0 - m2) * dk3 * g23 + k23 * c23 * dp3
                    A = torch.zeros(t1.shape[0], 3, 3, dtype=t1.dtype, device=t1.device)
                    A[:, 0, 0] = -a12_1 / z1
                    A[:, 0, 1] = -a12_2 / z1
                    A[:, 1, 0] = a12_1 / z2
                    A[:, 1, 1] = (a12_2 - a23_2) / z2
                    A[:, 1, 2] = -a23_3 / z2
                    A[:, 2, 1] = a23_2 / z3
                    A[:, 2, 2] = a23_3 / z3
                    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
                    M = torch.eye(3, dtype=t1.dtype, device=t1.device).expand_as(A) - dtc.detach()[:, None, None] * A
                    Minv = torch.linalg.inv(M)
                ff = torch.stack([f1, f2, f3], dim=1) * dtc[:, None]
                d = torch.bmm(Minv, ff[:, :, None])[:, :, 0]
                t1 = t1 + d[:, 0]
                t2 = t2 + d[:, 1]
                t3 = t3 + d[:, 2]
            else:
                t1 = t1 + dtc * f1
                t2 = t2 + dtc * f2
                t3 = t3 + dtc * f3
            # cascade SATURATION fidèle C++ (l.2046-2116) : on REMPLIT d'abord la
            # capacité disponible (refoulement bas→haut PUIS redistribution
            # haut→bas) ; seul l'excès quand le profil est plein déborde en
            # ruissellement. surplus=0 hors débordement → blocs applicables tels quels.
            zr = torch.zeros_like(t1)
            # bloc A : couche 3 sature → vers 2
            s = torch.clamp(t3 - ths3, min=0.0); s = _cd(s); t2 = t2 + s * z3 / z2; t3 = t3 - s
            # bloc B : couche 2 sature ET 3 a de la place → vers 3 (re-refoule si 3 sature)
            doB = (t2 > ths2) & (t3 < ths3)
            s = torch.where(doB, t2 - ths2, zr) * z2; s = _cd(s); t3 = t3 + s / z3; t2 = torch.where(doB, ths2, t2)
            s = torch.clamp(t3 - ths3, min=0.0); s = _cd(s); t2 = t2 + s * z3 / z2; t3 = t3 - s
            # bloc C : couche 2 sature → vers 1
            s = torch.clamp(t2 - ths2, min=0.0); s = _cd(s); t1 = t1 + s * z2 / z1; t2 = t2 - s
            # bloc D : couche 1 sature ET place en dessous → vers 2 (puis cascade)
            doD = (t1 > ths1) & ((t2 < ths2) | (t3 < ths3))
            s = torch.where(doD, t1 - ths1, zr) * z1; s = _cd(s); t2 = t2 + s / z2; t1 = torch.where(doD, ths1, t1)
            doDi = (t2 > ths2) & (t3 < ths3)
            s = torch.where(doDi, t2 - ths2, zr) * z2; s = _cd(s); t3 = t3 + s / z3; t2 = torch.where(doDi, ths2, t2)
            s = torch.clamp(t3 - ths3, min=0.0); s = _cd(s); t2 = t2 + s * z3 / z2; t3 = t3 - s
            s = torch.clamp(t2 - ths2, min=0.0); s = _cd(s); t1 = t1 + s * z2 / z1; t2 = t2 - s
            # bloc E : couche 1 encore sature → RUISSELLEMENT (l.2113)
            ov1 = _cd(torch.clamp(t1 - ths1, min=0.0)); t1 = t1 - ov1
            # NÉGATIVITÉ (l.2118-2158) : refoule depuis la couche du dessous
            neg1 = _cd(torch.clamp(-t1, min=0.0)); t1 = t1 + neg1; t2 = t2 - neg1 * z1 / z2
            neg2 = _cd(torch.clamp(-t2, min=0.0)); t2 = t2 + neg2; t3 = t3 - neg2 * z2 / z3
            t3 = torch.clamp(t3, min=0.0)
            lruis = lruis + ruis_rate * dtc + ov1 * z1
            lhyp = lhyp + (q2 + q3_lat) * dtc
            lbase = lbase + q3 * dtc
            tr = torch.clamp(tr - dtc, min=0.0)
            if not self.static:                      # break = synchro GPU + graph-break
                if bool((tr <= 1e-7).all()):
                    break

        # CONSERVATION DE LA MASSE (bug trouvé le 2026-08-09). Le C++ boucle JUSQU'À
        # tr=0 (while, onde TriCoucheOct97) ; le clone plafonne à n_substep itérations
        # pour le GPU. Quand le gel/la crue force l'échelle de Courant à DT_H/1152, un
        # jour exige ~1152 sous-pas : le plafond (48) tombait après ~1 h de journée et
        # le temps restant DISPARAISSAIT avec sa pluie — fuite mesurée de 210 mm/an
        # (21 % de P) sur OUTV, concentrée en mars-avril et oct-déc, nulle l'été.
        # L'eau du temps non traité RUISSELLE, NETTE de l'évapotranspiration de ce même
        # temps (l'ETR du jour est comptée en entier par le module d'ET en amont : sans
        # cette déduction le bilan sur-corrige exactement de l'ETR non prélevée, mesuré
        # +199 mm/an). Ces situations sont celles du sol gelé ou saturé, où le C++
        # produit lui aussi du ruissellement.
        # Fermeture EXACTE en deux écritures, comme si le C++ avait continué sa boucle :
        # (1) la pluie du temps restant ruisselle (sol gelé/saturé) ; (2) l'évapo-
        # transpiration du temps restant est RETIRÉE DU STOCK — le module d'ET la
        # déclare en entier au bilan, mais la boucle plafonnée ne la prélevait qu'au
        # prorata du temps traité : l'eau « évaporée » restait dans le sol et
        # ressortait plus tard en débordement (sur-correction mesurée +199 puis +130
        # mm/an avec les fermetures naïves). Négativité refoulée comme dans la boucle.
        lruis = lruis + prec * tr
        t1 = t1 - e1 * tr / z1
        t2 = t2 - e2 * tr / z2
        t3 = t3 - e3 * tr / z3
        neg1 = torch.clamp(-t1, min=0.0); t1 = t1 + neg1; t2 = t2 - neg1 * z1 / z2
        neg2 = torch.clamp(-t2, min=0.0); t2 = t2 + neg2; t3 = t3 - neg2 * z2 / z3
        t3 = torch.clamp(t3, min=0.0)

        # ── CalculeUHRH (l.820) : production avec split occupation du sol ──
        # leau = (pluie − ET) sur fraction EAU ; lprec = pluie sur IMPERMÉABLE
        # fractions eau/imperméable : apport TOTAL (le ruissellement avant le sol ne les concerne pas)
        lprec = (apport_total_mm / 1000.0)               # m (pluie totale du pas)
        leau = torch.clamp((apport_total_mm - etp_mm) / 1000.0, min=0.0)
        # le ruissellement pris avant le sol s'ajoute au ruissellement de la fraction sol
        # perméable (fsa), avant pondération.
        lruis = lruis + runoff_before_soil / 1000.0           # m
        prod_surf = lruis * fsa + leau * fse + lprec * fsi      # m
        prod_hypo = lhyp * fsa
        prod_base = lbase * fsa
        recharge = coef_rech * (prod_hypo + prod_base)
        prod_hypo = prod_hypo - prod_hypo * coef_rech
        prod_base = prod_base - prod_base * coef_rech

        # Fraction de la journee que la boucle n'a PAS traitee, faute d'iterations. Sa
        # pluie ruisselle et son evapotranspiration est retiree du stock, mais ni le
        # drainage profond ni l'hypodermique de ce temps ne sont accumules : la recharge
        # y est donc perdue au profit du ruissellement. Expose pour que cette perte se
        # mesure par mois au lieu de se deduire.
        diag = dict(pinf=pinf, ruis_hortonien=ruis_rate * DT_H * 1000.0,
                    sat_t1=(t1 / (ths1 + eps)),
                    temps_non_traite=tr / DT_H,
                    # Evaporation de la fraction EAU (2026-10-02) : l'eau libre perd
                    # min(apport, ETP), puisque seul l'excedent ruisselle. Perte reelle,
                    # jamais publiee : sans elle l'audit de bilan lit +1,5 % en Outaouais.
                    evap_eau_mm=(lprec - leau) * fse * 1000.0,
                    ruiss_eau_mm=leau * fse * 1000.0,
                    # Ruissellement des processus declares (« runoff »), mm sur le troncon : les lacs
                    # hors reseau le laissent passer (2026-11-04, R492).
                    ruiss_declare_mm=_r_decl_out * fsa)
        return (torch.clamp(prod_surf, min=0.0) * 1000.0,      # mm
                torch.clamp(prod_hypo, min=0.0) * 1000.0,
                torch.clamp(prod_base, min=0.0) * 1000.0,
                recharge * 1000.0, (t1, t2, t3), diag)


def make_params(texture1="silt_loam", texture2="loam", texture3="loam", slope=0.04,
                fsa=0.90, fse=0.05, fsi=0.05, krec=KREC_DEFAULT, cin=CIN_DEFAULT,
                coef_recharge=0.0, device="cpu"):
    """Paramètres par nœud (scalaires broadcastables). b=1/lambda, omegpi+spline."""
    def T(x): return torch.tensor(float(x), device=device)
    tx = [SOIL_TEXTURES[texture1], SOIL_TEXTURES[texture2], SOIL_TEXTURES[texture3]]
    z = EPAISSEUR; p = {}
    for i, t in enumerate(tx, start=1):
        b = 1.0 / t["lam"]
        omegpi = (1.0 + 2.0 * b) / (2.0 + 2.0 * b)
        # spline -mm(omega-nn)(omega-1) raccordée C1 en omegpi à psis·omega^-b
        psi_i = t["psis"] * omegpi ** (-b)            # psi à omegpi (branche puissance)
        dpsi_i = -t["psis"] * b * omegpi ** (-b - 1.0)  # dpsi/domega à omegpi
        A = omegpi
        # spline psi = -mm(omega-nn)(omega-1), raccord C1 en omegpi (valeur+pente).
        # Résolution analytique (voir dérivation) : r = psi_i/dpsi_i
        r = psi_i / dpsi_i
        nn_v = (A * A - A - 2.0 * r * A + r) / (A - 1.0 - r)
        mm_v = -dpsi_i / (2.0 * A - nn_v - 1.0)
        p[f"z{i}"] = T(z[i-1]); p[f"thetas{i}"] = T(t["thetas"]); p[f"ks{i}"] = T(t["ks"])
        p[f"b{i}"] = T(b); p[f"psis{i}"] = T(t["psis"]); p[f"omegpi{i}"] = T(omegpi)
        p[f"mm{i}"] = T(mm_v); p[f"nn{i}"] = T(nn_v)
    p["slope"] = T(slope); p["krec"] = T(krec); p["cin"] = T(cin)
    p["fsa"] = T(fsa); p["fse"] = T(fse); p["fsi"] = T(fsi); p["coef_recharge"] = T(coef_recharge)
    return p
