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
import torch
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

    def __init__(self, n_substep=48):
        super().__init__()
        self.n_substep = n_substep   # sous-pas internes (le C++ adapte ; on fixe)

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
        b1, b2, b3 = p["b1"], p["b2"], p["b3"]
        psis1, psis2, psis3 = p["psis1"], p["psis2"], p["psis3"]
        krec, slope, cin = p["krec"], p["slope"], p["cin"]
        fsa, fse, fsi = p["fsa"], p["fse"], p["fsi"]
        coef_rech = p["coef_recharge"]

        # ── CalculeRuisselement (l.2175) : prec constant ; pinf/ruis RECALCULÉS
        # CHAQUE sous-pas dans la boucle (porte theta1==thetas sur le t1 COURANT). ──
        prec = (apport_mm / 1000.0) / DT_H              # mm/j -> m/h
        frozen = (frozen_depth_cm > 0.0) & (swe_mm < 10.0)

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
        throttle = torch.where(frozen, torch.clamp(1.0 - froz_frac, 0.0, 1.0), torch.ones_like(t1))
        tr = torch.full_like(t1, DT_H)                  # temps restant (h) par nœud
        fdtcmin = DT_H / (24.0 * 60.0 * 60.0 * 1000.0)  # _fDTCMin C++ (l.744)

        for _ in range(self.n_substep):                 # cap d'itérations (sécurité)
            k1 = campbell_K(t1, ths1, ks1, b1); k2 = campbell_K(t2, ths2, ks2, b2); k3 = campbell_K(t3, ths3, ks3, b3)
            ps1 = campbell_psi(t1, ths1, psis1, b1, p["omegpi1"], p["mm1"], p["nn1"])
            ps2 = campbell_psi(t2, ths2, psis2, b2, p["omegpi2"], p["mm2"], p["nn2"])
            ps3 = campbell_psi(t3, ths3, psis3, b3, p["omegpi3"], p["mm3"], p["nn3"])
            k12 = torch.maximum(k1, k2); k23 = torch.maximum(k2, k3)
            qq12 = k12 * (2.0 * (ps2 - ps1) / (z1 + z2) + 1.0)
            qq23 = k23 * (2.0 * (ps3 - ps2) / (z2 + z3) + 1.0)
            q2 = k2 * sin_slope * z2; q3 = krec * z3 * t3
            qq12 = qq12 * throttle; qq23 = qq23 * throttle; q2 = q2 * throttle
            q3 = torch.where(frozen, q3 * 0.5, q3)
            # CalculeRuisselement (l.2191-2201) sur t1 COURANT : si t1 saturé,
            # pinf=0 → toute la pluie part en hortonien ; sinon pinf=min(prec,ks).
            omega1_sat = t1 >= (ths1 - 1e-4)
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
            # ET BRUTE (C++ l.2036 : v_etr1 sans clamp) ; la négativité éventuelle
            # est refoulée depuis la couche du dessous (l.2118), PAS masquée.
            t1 = t1 + dtc * (pinf - qq12 - e1) / z1
            t2 = t2 + dtc * (qq12 - qq23 - e2 - q2) / z2
            t3 = t3 + dtc * (qq23 - q3 - e3) / z3
            # cascade SATURATION fidèle C++ (l.2046-2116) : on REMPLIT d'abord la
            # capacité disponible (refoulement bas→haut PUIS redistribution
            # haut→bas) ; seul l'excès quand le profil est plein déborde en
            # ruissellement. surplus=0 hors débordement → blocs applicables tels quels.
            zr = torch.zeros_like(t1)
            # bloc A : couche 3 sature → vers 2
            s = torch.clamp(t3 - ths3, min=0.0); t2 = t2 + s * z3 / z2; t3 = t3 - s
            # bloc B : couche 2 sature ET 3 a de la place → vers 3 (re-refoule si 3 sature)
            doB = (t2 > ths2) & (t3 < ths3)
            s = torch.where(doB, t2 - ths2, zr) * z2; t3 = t3 + s / z3; t2 = torch.where(doB, ths2, t2)
            s = torch.clamp(t3 - ths3, min=0.0); t2 = t2 + s * z3 / z2; t3 = t3 - s
            # bloc C : couche 2 sature → vers 1
            s = torch.clamp(t2 - ths2, min=0.0); t1 = t1 + s * z2 / z1; t2 = t2 - s
            # bloc D : couche 1 sature ET place en dessous → vers 2 (puis cascade)
            doD = (t1 > ths1) & ((t2 < ths2) | (t3 < ths3))
            s = torch.where(doD, t1 - ths1, zr) * z1; t2 = t2 + s / z2; t1 = torch.where(doD, ths1, t1)
            doDi = (t2 > ths2) & (t3 < ths3)
            s = torch.where(doDi, t2 - ths2, zr) * z2; t3 = t3 + s / z3; t2 = torch.where(doDi, ths2, t2)
            s = torch.clamp(t3 - ths3, min=0.0); t2 = t2 + s * z3 / z2; t3 = t3 - s
            s = torch.clamp(t2 - ths2, min=0.0); t1 = t1 + s * z2 / z1; t2 = t2 - s
            # bloc E : couche 1 encore sature → RUISSELLEMENT (l.2113)
            ov1 = torch.clamp(t1 - ths1, min=0.0); t1 = t1 - ov1
            # NÉGATIVITÉ (l.2118-2158) : refoule depuis la couche du dessous
            neg1 = torch.clamp(-t1, min=0.0); t1 = t1 + neg1; t2 = t2 - neg1 * z1 / z2
            neg2 = torch.clamp(-t2, min=0.0); t2 = t2 + neg2; t3 = t3 - neg2 * z2 / z3
            t3 = torch.clamp(t3, min=0.0)
            lruis = lruis + ruis_rate * dtc + ov1 * z1
            lhyp = lhyp + q2 * dtc
            lbase = lbase + q3 * dtc
            tr = torch.clamp(tr - dtc, min=0.0)
            if bool((tr <= 1e-7).all()):
                break

        # ── CalculeUHRH (l.820) : production avec split occupation du sol ──
        # leau = (pluie − ET) sur fraction EAU ; lprec = pluie sur IMPERMÉABLE
        lprec = (apport_mm / 1000.0)                     # m (pluie totale du pas)
        leau = torch.clamp((apport_mm - etp_mm) / 1000.0, min=0.0)
        prod_surf = lruis * fsa + leau * fse + lprec * fsi      # m
        prod_hypo = lhyp * fsa
        prod_base = lbase * fsa
        recharge = coef_rech * (prod_hypo + prod_base)
        prod_hypo = prod_hypo - prod_hypo * coef_rech
        prod_base = prod_base - prod_base * coef_rech

        diag = dict(pinf=pinf, ruis_hortonien=ruis_rate * DT_H * 1000.0,
                    sat_t1=(t1 / (ths1 + eps)))
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
