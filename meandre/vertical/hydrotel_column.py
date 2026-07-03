"""Colonne verticale FIDÈLE Hydrotel — orchestrateur natif méandre (Phase A).

Chaîne, dans l'ordre EXACT d'Hydrotel (cf BV3C2::Calcule), les modules clonés et
validés un à un contre le C++ (hydrotel_clone/, Phase B) :

  forçage → split pluie/neige → FONTE NEIGE (degré-jour modifié)
         → GEL RANKINEN (profil de température → profondeur de gel)
         → ETP (McGuinness | Hydro-Québec | Penman-Monteith, au choix)
         → ETR par couche (CalculeEtr : sol nu Beer + transpiration racine×θ)
         → BILAN SOL BV3C2 (cascade saturation + split occupation fsa/fse/fsi,
           porte gel alimentée par RANKINEN)
         → MILIEU HUMIDE ISOLÉ (réservoir SWAT, optionnel par nœud)
         → production_surf/hypo/base (mm) = apport latéral au routage.

Tout est vectorisé sur les nœuds et différentiable. Les paramètres statiques par
nœud (texture, occupation, profondeurs, params neige/gel/wetland) sont posés une
fois via set_static() ; le sous-ensemble apprenable sera fourni par le NeRF en
Phase A (TODO ci-dessous). Ceci est le SQUELETTE : structure + forward complet
qui tourne ; le câblage NeRF/territorial et l'intégration dans model.py suivent.

DÉCISIONS Phase A (paramètres du constructeur) :
  - et_mode : formulation ETP (les 3 se valent ~ sous loss MODIS/GRACE).
  - use_frost : gel RANKINEN actif (amélioration méandre ; Hydrotel SLSO l'a OFF)
    → alimente la porte gel du sol (remplace le frost.py problématique).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from meandre.utils.state import HydroState
from hydrotel_clone.snow import DegreJourModifie, init_ce, init_state as snow_init_state
from hydrotel_clone.frost import Rankinen, n_intervalles
from hydrotel_clone.et import hydro_quebec_etp, calcule_etr
from hydrotel_clone.mcguinness import mcguinness_etp
from hydrotel_clone.bv3c2 import BV3C2Clone, make_params, SOIL_TEXTURES
from hydrotel_clone.milieu_humide import init_wetland_geom, calcul_milieu_humide_isole
from meandre.vertical.aquifer import AquiferModule


def _interp1d(x, xp, fp):
    """Interpolation linéaire 1D torch reproduisant np.interp (xp croissant,
    clampé aux bords). x : tenseur 0-dim. Sans .item()/float() → pas de synchro
    GPU, compilable. Remplace le np.interp par pas de temps de la phénologie."""
    i = torch.searchsorted(xp, x.reshape(1), right=True).clamp(1, xp.numel() - 1)[0]
    x0, x1, y0, y1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    y = torch.where(x <= xp[0], fp[0], y)
    return torch.where(x >= xp[-1], fp[-1], y)


@dataclass
class HydrotelColumnState:
    """État interne de la colonne (plus riche que HydroState : la neige a 3
    classes avec cold content + eau retenue + albédo, le gel a un profil de
    température, le wetland un volume). theta1/2/3 et le couvert nival agrégé
    sont exposés pour la compatibilité avec HydroState/routage."""
    theta1: Tensor
    theta2: Tensor
    theta3: Tensor
    snow: dict                 # {classe: (stock,hauteur,chaleur,eau), 'albedo_'+classe, 'couvert_nival_mm'}
    frost_profile: Tensor      # (n_nodes, n_depth) température du sol
    wet_vol: Tensor            # (n_nodes,) volume milieu humide [m3]
    # États des cascades de Nash de l'hydrogramme de VERSANT (use_hillslope_uh) :
    # surface (uh_s1,uh_s2) + interflow (uh_s3,uh_s4). None si UH désactivé.
    uh_s1: Tensor | None = None
    uh_s2: Tensor | None = None
    uh_s3: Tensor | None = None
    uh_s4: Tensor | None = None

    def detach(self) -> "HydrotelColumnState":
        sn = {}
        for k, v in self.snow.items():
            sn[k] = tuple(t.detach() for t in v) if isinstance(v, tuple) else v.detach()
        _d = lambda t: t.detach() if t is not None else None
        return HydrotelColumnState(
            self.theta1.detach(), self.theta2.detach(), self.theta3.detach(),
            sn, self.frost_profile.detach(), self.wet_vol.detach(),
            _d(self.uh_s1), _d(self.uh_s2), _d(self.uh_s3), _d(self.uh_s4))


class HydrotelColumn(nn.Module):
    """Colonne verticale fidèle Hydrotel, un pas de temps journalier, vectorisée."""

    def __init__(self, et_mode: str = "mcguinness", use_frost: bool = True,
                 soil_n_substep: int = 48, frost_intervalle: float = 0.05,
                 frost_temp_ini: float = 4.0, frost_seuil: float = -0.5,
                 frost_fs: float = 2.35, frost_kt: float = 0.8,
                 frost_cs: float = 1.0e6, frost_cice: float = 4.0e6,
                 t_neige_seuil: float = 0.0, compile_soil: bool = False,
                 compile_column: bool = False, use_hillslope_uh: bool = False,
                 melt_mode: str = "degree_day", use_aquifer: bool = False,
                 use_hortonian: bool = False) -> None:
        super().__init__()
        self.et_mode = str(et_mode)
        self.use_frost = bool(use_frost)
        self.use_hillslope_uh = bool(use_hillslope_uh)
        # Mode de fonte : "degree_day" (clone fidèle, indice radiation géométrique)
        # ou "eti" (Enhanced Temperature Index, radiation RÉELLE sw_in = canal FB).
        self.melt_mode = str(melt_mode)
        self.sw_channel = 6 if self.melt_mode == "eti" else None   # FB = canal 6 du forçage (cache -eb)
        # Hortonien sous-journalier (excès d'infiltration intensité-dépendant) : DT_eff au canal 6 (cache -intens).
        self.use_hortonian = bool(use_hortonian)
        self.storm_channel = 6 if self.use_hortonian else None
        # GARDE-FOU : les deux modes lisent le canal 6 mais attendent des caches
        # DIFFÉRENTS (FB en W/m² vs DT_eff en h). Les combiner lirait silencieusement
        # la mauvaise grandeur physique (revue 2026-07-01). Interdit tant que les
        # index ne sont pas séparés/validés contre les noms de variables du cache.
        if self.melt_mode == "eti" and self.use_hortonian:
            raise ValueError(
                "melt_mode='eti' et use_hortonian=True sont incompatibles : les deux "
                "lisent le canal de forçage 6 (FB vs DT_eff). Construire un cache "
                "combiné et séparer sw_channel/storm_channel avant de les cumuler.")
        # Aquifère restituant OPTIONNEL (meandre > Hydrotel) : route le drainage L3
        # (sinon baseflow instantané, recharge perdue cf. C++) dans un réservoir
        # linéaire qui SOUTIENT L'ÉTIAGE. Prélèvements souterrains agissent dans le
        # réservoir. k_gw par nœud (NeRF), contraint GRACE. OFF = clone Hydrotel fidèle.
        self.use_aquifer = bool(use_aquifer)
        self.aquifer = AquiferModule() if self.use_aquifer else None
        self.t_neige_seuil = t_neige_seuil   # seuil pluie/neige (split de phase, TODO: règle Hydrotel exacte)
        self.snow = DegreJourModifie(pas_de_temps=24)
        self.frost = Rankinen(frost_intervalle, frost_temp_ini, frost_seuil, frost_fs,
                              frost_kt, frost_cs, frost_cice, pas_de_temps=24)
        # Deux modes de compilation (le wall-clock est CPU-dispatch-bound sur la
        # boucle par jour, GPU ~0% en eager) :
        #  - compile_column : compile TOUT le compute du pas (snow+gel+ET+sol) en
        #    peu de kernels/jour → réduit le dispatch Python. Le sol est static
        #    (sans break) mais PAS compilé séparément (le compile externe l'inline).
        #  - compile_soil : compile seulement le sol (sous-ensemble de l'effet).
        # Boucle de sous-pas static = résultats IDENTIQUES au mode break (vérifié).
        self.compile_column = bool(compile_column)
        soil_static = bool(compile_soil or compile_column)
        self.soil = BV3C2Clone(n_substep=soil_n_substep, static=soil_static)
        if compile_soil and not compile_column:
            self.soil = torch.compile(self.soil, dynamic=False)
        self._fwd_compiled = None
        # Ancrage OPTIONNEL sur la calibration Hydrotel (reproduce). None = init
        # NeRF/littérature (objectif ultime : découplé). Posé via set_calibrated_soil.
        self._calib_soil = None
        self._static = None      # posé par set_static()
        self.z1 = 0.15           # épaisseur couche 1 (config ; Z2/Z3 du NeRF)

        # ── Paramètres globaux apprenables (absents du NeRF) ──
        import math as _m
        self._b_bounds = (1.4, 6.0); self._psis_bounds = (0.01, 1.0); self._krec_bounds = (1e-7, 1e-4)
        def _inv(v, lo, hi):
            f = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
            return torch.tensor(_m.log(f / (1 - f)))
        tx = SOIL_TEXTURES["sandy_loam"]   # init Campbell (sera régionalisé plus tard)
        for i in (1, 2, 3):
            setattr(self, f"b{i}_raw", nn.Parameter(_inv(1.0 / tx["lam"], *self._b_bounds)))
            setattr(self, f"psis{i}_raw", nn.Parameter(_inv(tx["psis"], *self._psis_bounds)))
        self.krec_raw = nn.Parameter(_inv(1e-6, *self._krec_bounds))
        self.log_etr_alpha = nn.Parameter(torch.tensor(_m.log(4.5)))      # assèchement ETR
        # fonte degré-jour par classe (softplus pour positivité), init Hydrotel 12/14/16
        self.sp_fonte_conif = nn.Parameter(torch.tensor(_m.log(_m.exp(12.0) - 1)))
        self.sp_fonte_feu = nn.Parameter(torch.tensor(_m.log(_m.exp(14.0) - 1)))
        self.sp_fonte_dec = nn.Parameter(torch.tensor(_m.log(_m.exp(16.0) - 1)))
        # ETI (mode "eti") : facteurs de fonte température (tf, m/°C/j) et radiation
        # (srf, m/j par W/m²), softplus pour positivité. Init Pellicciotti 2005 (en
        # journalier) : tf≈1.2 mm/°C/j, srf≈0.2 mm/j par W/m². Apprenables.
        self.sp_tf = nn.Parameter(torch.tensor(_m.log(_m.exp(0.0012) - 1)))
        self.sp_srf = nn.Parameter(torch.tensor(_m.log(_m.exp(0.00020) - 1)))
        # Hydrogramme unitaire de VERSANT (cascade de Nash 2 réservoirs, fidèle
        # Hydrotel — porté de column.py). Lisse le ruissellement AVANT le canal,
        # par étalement des temps de parcours (préserve les pics, contrairement à
        # l'atténuation Muskingum). DEUX échelles : surface POINTUE (k court),
        # interflow LARGE (k long). À coupler avec pure_advection (canal cinématique).
        if self.use_hillslope_uh:
            self.log_uh_k_surf = nn.Parameter(torch.tensor(_m.log(0.3)))    # ~0.3 j
            self.log_uh_k_inter = nn.Parameter(torch.tensor(_m.log(2.5)))   # ~2.5 j

    def _sig(self, raw, bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)

    @staticmethod
    def _spline(b, psis):
        A = (1.0 + 2.0 * b) / (2.0 + 2.0 * b)
        psi_i = psis * A.pow(-b); dpsi_i = -psis * b * A.pow(-b - 1.0); r = psi_i / dpsi_i
        nn_ = (A * A - A - 2.0 * r * A + r) / (A - 1.0 - r)
        mm_ = -dpsi_i / (2.0 * A - nn_ - 1.0)
        return A, mm_, nn_

    # ── Paramètres statiques par nœud ───────────────────────────────────
    def set_static(self, p_snow: dict, p_soil: dict, p_etr: dict,
                   wetland: dict | None = None, n_depth: int = 31) -> None:
        """Pose les paramètres statiques par nœud (une fois par simulate).
        p_snow : params DegreJourModifie (lat, ce1, ce0, pct_*, coeff_fonte_*, ...).
        p_soil : params BV3C2 (make_params : Campbell + fsa/fse/fsi + krec/cin/slope + z).
        p_etr  : params ETR (thetacc, thetapf, alpha, des, coef_assech, z11/z22/z33,
                 classes : liste de (pct, jours_bp, leaf_bp, root_bp)).
        wetland: params milieu humide par nœud (A,B,wetnvol,wetmxvol,wet_k,c_ev,
                 c_prod,hru_ha,wet_fr) ou None si aucun.
        n_depth: nombre de nœuds du profil de gel."""
        self._static = dict(snow=p_snow, soil=p_soil, etr=p_etr, wetland=wetland, n_depth=n_depth)

    def set_calibrated_soil(self, p_soil: dict):
        """Ancre le sol sur la calibration Hydrotel (params par nœud). Quand posé,
        params_from_nerf l'utilise À LA PLACE du sol NeRF (reproduce). Optionnel.
        Mémorise aussi les z médians pour rendre ETR/gel cohérents avec ces z."""
        self._calib_soil = p_soil
        self._calib_z = (float(p_soil["z1"].median()), float(p_soil["z2"].median()),
                         float(p_soil["z3"].median()))

    # ── Seam NeRF/territorial → params (Phase A) ────────────────────────
    def params_from_nerf(self, sp, territorial, node_coords):
        """Assemble les params statiques par nœud depuis le NeRF (SpatialParams)
        et le territorial. Sol : thetas/ks/fc/wp/Z du NeRF, Campbell b/psis/krec
        globaux apprenables. Split fsa/fse/fsi FIDÈLE depuis l'occupation brute.
        Classes neige/ETR depuis l'occupation disponible (forêt agrégée si le
        split conif/feuillus brut manque — voir _raw_keep / rebuild).
        Pose le résultat via set_static() et retourne (p_snow, p_soil, p_etr)."""
        like = sp.porosity_1
        gp = territorial.get_physical
        z = lambda k, d: (gp(k) if gp(k) is not None else torch.full_like(like, d))

        # occupation brute
        f_water = z("f_water_raw", 0.0); f_lake = z("lake_fraction_raw", 0.0)
        f_urban = z("f_urban_raw", 0.0); f_forest = z("f_forest_raw", 0.0)
        f_wet = z("f_wetland_raw", 0.0)
        fse = torch.clamp(f_water + f_lake, 0.0, 1.0)
        fsi = torch.clamp(f_urban, 0.0, 1.0)
        fsa = torch.clamp(1.0 - fse - fsi, 0.0, 1.0)
        # pente = géométrie universelle. Priorité : slope_fraction (présent dans le
        # cache PHYSITEL, déjà en fraction), sinon mean_slope_pct_raw/100, sinon défaut.
        slope_frac = gp("slope_fraction")
        if slope_frac is not None:
            slope = torch.clamp(slope_frac.to(like.dtype), 1e-3, 0.5)
        else:
            slope = torch.clamp(z("mean_slope_pct_raw", 4.0) / 100.0, 1e-3, 0.5)

        # orientation depuis l'aspect (sin/cos dans .data) → code 0-7
        try:
            si = territorial.data[:, territorial.columns.index("sin_aspect")]
            co = territorial.data[:, territorial.columns.index("cos_aspect")]
            asp = (torch.rad2deg(torch.atan2(si, co)) % 360.0)
            orient = torch.round(asp / 45.0) % 8
        except (ValueError, AttributeError):
            orient = torch.full_like(like, 7.0)
        lat = node_coords[:, 1].to(like.dtype)
        ce1, ce0 = init_ce(lat, slope, orient)

        # classes neige : forêt → feuillus (split conif brut indispo sur OD), reste découvert
        f_conif_raw = gp("f_forest_conifer_raw")
        if f_conif_raw is not None:        # disponible après rebuild → split fidèle
            pct_conif = f_conif_raw
            pct_feu = gp("f_forest_deciduous_raw") + (gp("f_forest_mixed_raw") or 0.0)
        else:                               # fallback OD : toute la forêt en feuillus
            pct_conif = torch.zeros_like(like); pct_feu = f_forest
        sp_ = torch.nn.functional.softplus
        p_snow = dict(lat=lat, ce1=ce1, ce0=ce0,
                      pct_conifers=pct_conif, pct_feuillus=pct_feu,
                      pct_autres=torch.clamp(1.0 - pct_conif - pct_feu, 0.0, 1.0),
                      coeff_fonte_conifers=sp_(self.sp_fonte_conif) / 1000.0 * torch.ones_like(like),
                      coeff_fonte_feuillus=sp_(self.sp_fonte_feu) / 1000.0 * torch.ones_like(like),
                      coeff_fonte_decouver=sp_(self.sp_fonte_dec) / 1000.0 * torch.ones_like(like),
                      seuil_fonte_conifers=torch.zeros_like(like), seuil_fonte_feuillus=torch.zeros_like(like),
                      seuil_fonte_decouver=torch.zeros_like(like),
                      taux_fonte_geo=torch.full_like(like, 0.5), densite_max=torch.full_like(like, 466.0),
                      constante_tassement=torch.full_like(like, 0.1),
                      melt_mode=self.melt_mode,
                      tf=sp_(self.sp_tf) * torch.ones_like(like),
                      srf=sp_(self.sp_srf) * torch.ones_like(like))

        # sol BV3C2 : NeRF (thetas/ks) + Campbell global
        b1 = self._sig(self.b1_raw, self._b_bounds); b2 = self._sig(self.b2_raw, self._b_bounds); b3 = self._sig(self.b3_raw, self._b_bounds)
        ps1 = self._sig(self.psis1_raw, self._psis_bounds); ps2 = self._sig(self.psis2_raw, self._psis_bounds); ps3 = self._sig(self.psis3_raw, self._psis_bounds)
        o1, m1, n1 = self._spline(b1, ps1); o2, m2, n2 = self._spline(b2, ps2); o3, m3, n3 = self._spline(b3, ps3)
        ob = lambda v: v * torch.ones_like(like)
        p_soil = dict(z1=torch.full_like(like, self.z1), z2=sp.Z2, z3=sp.Z3,
                      thetas1=sp.porosity_1, thetas2=sp.porosity_2, thetas3=sp.porosity_3,
                      ks1=sp.K_sat_1 / 24.0, ks2=sp.K_sat_2 / 24.0, ks3=sp.K_sat_3 / 24.0,
                      b1=ob(b1), b2=ob(b2), b3=ob(b3), psis1=ob(ps1), psis2=ob(ps2), psis3=ob(ps3),
                      omegpi1=ob(o1), omegpi2=ob(o2), omegpi3=ob(o3), mm1=ob(m1), mm2=ob(m2), mm3=ob(m3),
                      nn1=ob(n1), nn2=ob(n2), nn3=ob(n3), krec=ob(self._sig(self.krec_raw, self._krec_bounds)),
                      slope=slope, cin=torch.full_like(like, 0.03),
                      fsa=fsa, fse=fse, fsi=fsi, coef_recharge=torch.zeros_like(like))

        # Ancrage Hydrotel (reproduce) : remplace le sol NeRF par la calibration
        # par nœud si fournie. Optionnel — retiré pour découpler.
        if self._calib_soil is not None:
            p_soil = {k: v.to(like.device).to(like.dtype) for k, v in self._calib_soil.items()}

        # ETR : thetacc/thetapf du NeRF (couche 1), alpha global ; classes dispo
        alpha = torch.exp(self.log_etr_alpha)
        et_classes = []
        if pct_feu.sum() > 0:
            et_classes.append((pct_feu, _JBP, _LEAF["feuillus"], _ROOT["feuillus"]))
        if pct_conif.sum() > 0:
            et_classes.append((pct_conif, _JBP, _LEAF["conifers"], _ROOT["conifers"]))
        if f_wet.sum() > 0:
            et_classes.append((f_wet, _JBP, _LEAF["humides"], _ROOT["humides"]))
        # DÉGRADATION GRACIEUSE : sans descriptif d'occupation (ex réseau PHYSITEL,
        # qui ne porte pas les fractions par classe), l'ET ne doit PAS tomber à 0.
        # Classe végétation par défaut sur la fraction perméable (LAI/racines
        # génériques boréal QC). La variation spatiale de l'ET vient alors du NeRF
        # (thetacc/thetapf/ks → stress + eau dispo) + alpha apprenable. Occupation
        # = optionnelle, pas un prérequis (objectif découplage).
        if not et_classes:
            et_classes.append((fsa, _JBP, _LEAF["default"], _ROOT["default"]))
        # z des couches : calibrés Hydrotel si ancré (cohérence ETR/gel/sol), sinon NeRF
        z11, z22, z33 = self.z1, float(sp.Z2.mean()), float(sp.Z3.mean())
        if self._calib_soil is not None:
            z11, z22, z33 = self._calib_z
        # K_c (coefficient cultural) par nœud, prédit par le NeRF (borné [0.3,1.5],
        # prior vers 0.85). Multiplie l'ETP → corrige la sur-évaporation McGuinness
        # et laisse le NeRF caler le volume (β) par nœud. Levier de découplage.
        kc = sp.K_c if hasattr(sp, "K_c") else torch.ones_like(like)
        p_etr = dict(thetacc=sp.theta_fc_1, thetapf=sp.theta_wp_1, alpha=alpha * torch.ones_like(like),
                     des=torch.full_like(like, 0.6), coef_assech=torch.full_like(like, 1.0),
                     z11=z11, z22=z22, z33=z33, classes=et_classes, K_c=kc)

        # milieu humide isolé : actif SI le territorial porte la géométrie par nœud
        # (wet_a_raw). Sinon None (colonne sol seul, ex SLSO). Masqué + sûr gradient.
        wetland = self._wetland_from_territorial(territorial, like)

        n_depth = n_intervalles(z11 + z22 + z33, self.frost.dz)
        self.set_static(p_snow, p_soil, p_etr, wetland=wetland, n_depth=n_depth)
        if self.use_aquifer:
            self._static["k_gw"] = sp.k_gw   # récession aquifère par nœud (1/j)
        return p_snow, p_soil, p_etr

    def _wetland_from_territorial(self, territorial, like):
        """Construit le dict milieu humide isolé par nœud depuis territorial.physical
        (colonnes *_raw agrégées UHRH→troncon par le loader). Retourne None si pas de
        géométrie (wet_a_raw absent). Les nœuds sans MH sont masqués (wmask) et reçoivent
        une géométrie factice positive pour éviter log10(0)=NaN dans le gradient."""
        from hydrotel_clone.milieu_humide import wetland_geom_vec
        gp = territorial.get_physical
        wet_a = gp("wet_a_raw")
        if wet_a is None:
            return None
        z = lambda k, d: (gp(k) if gp(k) is not None else torch.full_like(like, d))
        area = torch.clamp(z("area_km2_local", 1.0), min=1e-6)   # aire locale du troncon [km2]
        wetdmax = z("wetdmax_raw", 0.3); frac = z("frac_raw", 0.8); wetdnor = z("wetdnor_raw", 0.2)
        wet_dra_fr = z("wet_dra_fr_raw", 0.0)
        wet_k = z("ksat_bs_raw", 0.5); c_ev = z("c_ev_raw", 0.6); c_prod = z("c_prod_raw", 10.0)
        wmask = wet_a > 0.0
        # géométrie factice POSITIVE sur les nœuds sans MH (sinon log10(0)=−inf→NaN
        # dans wetland_geom_vec ; ces nœuds sont masqués en aval de toute façon).
        wet_a_safe = torch.where(wmask, wet_a, torch.ones_like(wet_a))
        wetdmax_s = torch.where(wmask, wetdmax, torch.full_like(wetdmax, 0.3))
        frac_s = torch.where(wmask, frac, torch.full_like(frac, 0.8))
        wetdnor_s = torch.where(wmask, wetdnor, torch.full_like(wetdnor, 0.2))
        A, B, wetnvol, wetmxvol = wetland_geom_vec(wet_a_safe, wetdmax_s, frac_s, wetdnor_s)
        return dict(wet_fr_area=torch.clamp(wet_a / area, 0.0, 1.0), hru_ha=area * 100.0,
                    wet_dra_fr=torch.where(wmask, wet_dra_fr, torch.zeros_like(wet_dra_fr)),
                    A=A, B=B, wetnvol=wetnvol, wetmxvol=wetmxvol,
                    wet_k=wet_k, c_ev=c_ev, c_prod=c_prod, wmask=wmask)

    # ── État initial ────────────────────────────────────────────────────
    def init_state(self, n_nodes, theta_init, swe_init_mm=None, device="cpu",
                   dtype=torch.float64) -> HydrotelColumnState:
        sn = snow_init_state(n_nodes, device=device, dtype=dtype)
        z = lambda v: torch.full((n_nodes,), float(v), device=device, dtype=dtype)
        nd = self._static["n_depth"] if self._static else 31
        frost_profile = torch.full((n_nodes, nd), self.frost.temp_ini_base, device=device, dtype=dtype)
        _uh = (torch.zeros(n_nodes, device=device, dtype=dtype) if self.use_hillslope_uh else None)
        return HydrotelColumnState(
            theta1=z(theta_init[0]), theta2=z(theta_init[1]), theta3=z(theta_init[2]),
            snow=sn, frost_profile=frost_profile, wet_vol=torch.zeros(n_nodes, device=device, dtype=dtype),
            uh_s1=(_uh.clone() if _uh is not None else None),
            uh_s2=(_uh.clone() if _uh is not None else None),
            uh_s3=(_uh.clone() if _uh is not None else None),
            uh_s4=(_uh.clone() if _uh is not None else None))

    # ── Adaptateur interface VerticalColumn (pour model.py simulate) ────
    def setup_simulate(self, spatial_params, territorial, node_coords, init_state):
        """Appelé UNE fois avant la boucle simulate : assemble les params depuis le
        NeRF et initialise l'état interne riche (neige/gel/wetland) depuis l'état
        méandre. theta de départ = init_state.theta1/2/3."""
        self.params_from_nerf(spatial_params, territorial, node_coords)
        n = init_state.theta1.shape[0]
        dev, dt = init_state.theta1.device, init_state.theta1.dtype
        aux = self.init_state(n, theta_init=(0.0, 0.0, 0.0), device=dev, dtype=dt)
        aux.theta1, aux.theta2, aux.theta3 = init_state.theta1, init_state.theta2, init_state.theta3
        # volume initial du milieu humide depuis le territorial si dispo (sinon 0)
        wv0 = territorial.get_physical("wet_vol_init_raw")
        if wv0 is not None:
            aux.wet_vol = wv0.to(dev).to(dt)
        self._aux = aux

    def detach_aux(self):
        if getattr(self, "_aux", None) is not None:
            self._aux = self._aux.detach()

    def column_step(self, enriched, state, doy=None, return_diagnostics=False,
                    gw_withdrawal_mm=None, **_):
        """Un pas, interface ColumnOutput. enriched[:, :6] = P,Tmin,Tmax,Rn,u2,ea.
        theta est re-synchronisé depuis `state` (pour intégrer une éventuelle
        correction résiduelle), le reste de l'état riche est interne (self._aux).
        gw_withdrawal_mm : prélèvement/rejet souterrain (mm/j, +ajout/−retrait)."""
        from meandre.utils.state import ColumnOutput
        a = self._aux
        a.theta1, a.theta2, a.theta3 = state.theta1, state.theta2, state.theta3
        P, tmin, tmax = enriched[:, 0], enriched[:, 1], enriched[:, 2]
        Rn, u2, ea = enriched[:, 3], enriched[:, 4], enriched[:, 5]
        # Fonte ETI : courte longueur d'onde incidente brute = canal FB (index 6).
        sw_in = enriched[:, self.sw_channel] if self.sw_channel is not None else None
        # Hortonien sous-journalier : durée effective d'orage DT_eff = canal (index storm_channel).
        storm_hours = enriched[:, self.storm_channel] if self.storm_channel is not None else None
        prod, a, diag = self.forward(P, tmin, tmax, Rn, u2, ea,
                                     doy if doy is not None else 1, a, sw_in=sw_in, storm_hours=storm_hours)
        self._aux = a
        # Prélèvements/rejets SOUTERRAINS. La colonne Hydrotel fidèle n'a pas de
        # réservoir d'aquifère restituant (cf. C++ : la recharge fuit, le baseflow
        # pb est généré instantanément depuis le drainage L3). Un prélèvement
        # souterrain intercepte donc l'eau qui serait devenue baseflow CE JOUR :
        # on l'applique à pb (et donc à prod = ps_surf+ph+pb), borné >= 0. Signe :
        # +ajout (recharge artificielle/rejet), −retrait (pompage), cohérent avec
        # WithdrawalData/AquiferModule. forward() (validé décimale) reste intouché.
        pb = diag["prod_base"]
        if self.use_aquifer:
            # AQUIFÈRE RESTITUANT : le drainage L3 (pb) RECHARGE un réservoir linéaire
            # au lieu de sortir instantanément. Soutien d'étiage + prélèvement souterrain
            # agissent sur la VRAIE réserve (cf. AquiferModule). k_gw par nœud, GRACE le
            # contraint. prod = ps_surf+ph+pb -> on remplace pb par le baseflow retardé.
            kgw = self._static.get("k_gw")
            Q_bf, S_gw_new = self.aquifer(pb, state.S_gw, kgw, gw_withdrawal=gw_withdrawal_mm)
            prod = prod - pb + Q_bf
            diag["prod_base"] = Q_bf
            diag["lateral_mm"] = prod
        else:
            S_gw_new = state.S_gw
            if gw_withdrawal_mm is not None:
                pb_new = torch.clamp(pb + gw_withdrawal_mm, min=0.0)
                prod = prod + (pb_new - pb)
                diag["prod_base"] = pb_new
                diag["lateral_mm"] = prod
        new_state = HydroState(
            theta1=a.theta1, theta2=a.theta2, theta3=a.theta3,
            swe=diag["couvert_nival_mm"], t_soil=a.frost_profile[:, 0],
            canopy_storage=state.canopy_storage, wetland_storage=state.wetland_storage,
            S_gw=S_gw_new, T_water=state.T_water,
            cold_content=state.cold_content, gdd_cum=state.gdd_cum)
        return ColumnOutput(
            lateral_inflow=prod, state=new_state, snowmelt=diag["apport"],
            recharge=torch.zeros_like(prod), Q_baseflow=diag["prod_base"],
            diag=(diag if return_diagnostics else None))

    # ── Split de phase pluie/neige FIDÈLE (THIESSEN::PassagePluieNeige, thiessen1.cpp:259-279) ──
    def _split_precip(self, P, tmin, tmax):
        """Partition graduée pluie/neige au pas journalier. taux = fraction PLUIE :
        0 si tmax<seuil (tout neige), 1 si tmin>=seuil (tout pluie), sinon
        (tmax-seuil)/(tmax-tmin). Retourne (pluie, neige) en SWE (snow.py convertit
        en hauteur en interne). Seuil = t_neige_seuil (DELISLE 0°C)."""
        s = self.t_neige_seuil
        taux = torch.clamp((tmax - s) / (tmax - tmin + 1e-6), 0.0, 1.0)   # fraction pluie
        taux = torch.where(tmax < s, torch.zeros_like(taux), taux)
        taux = torch.where(tmin >= s, torch.ones_like(taux), taux)
        return taux * P, (1.0 - taux) * P   # pluie (SWE), neige (SWE)

    # ── ETP au choix ────────────────────────────────────────────────────
    def _etp(self, tmin_j, tmax_j, Rn, u2, ea, lat, doy):
        if self.et_mode == "mcguinness":
            return mcguinness_etp(tmin_j, tmax_j, lat, doy)
        if self.et_mode == "hydro_quebec":
            return hydro_quebec_etp(tmin_j, tmax_j)
        if self.et_mode == "penman":
            from meandre.vertical.evapotranspiration import ETModule
            return ETModule("penman").penman_monteith(tmin_j, tmax_j, Rn, u2, ea)
        raise ValueError(f"et_mode inconnu: {self.et_mode}")

    def _pheno_tensors(self, classes, ref):
        """Cache les breakpoints phénologie (jbp/leaf/root) en tenseurs au device/
        dtype de ref, construits une seule fois. Évite numpy + alloc par pas."""
        key = (ref.device, ref.dtype)
        if getattr(self, "_pheno_cache_key", None) != key:
            T = lambda v: torch.as_tensor(v, dtype=ref.dtype, device=ref.device)
            self._pheno_cache = [(pct, T(jbp), T(leaf_bp), T(root_bp))
                                 for (pct, jbp, leaf_bp, root_bp) in classes]
            self._pheno_cache_key = key
        return self._pheno_cache

    def forward(self, P, tmin, tmax, Rn, u2, ea, doy, state: HydrotelColumnState,
                tmin_j=None, tmax_j=None, sw_in=None, storm_hours=None) -> tuple[Tensor, HydrotelColumnState, dict]:
        """Dispatcher : appelle le forward compilé (compile_column) ou eager.
        Le compilé fond le compute par jour en peu de kernels (le wall-clock est
        dominé par le dispatch Python de la boucle par jour, GPU sinon ~0%).
        sw_in : courte longueur d'onde incidente (W/m²) pour la fonte ETI.
        storm_hours : durée effective d'orage (h, daily) pour l'hortonien sous-journalier."""
        if getattr(self, "compile_column", False):
            if getattr(self, "_fwd_compiled", None) is None:
                try:
                    self._fwd_compiled = torch.compile(self._forward_impl, dynamic=False)
                except Exception:
                    self._fwd_compiled = self._forward_impl   # fallback eager
            try:
                return self._fwd_compiled(P, tmin, tmax, Rn, u2, ea, doy, state, tmin_j, tmax_j, sw_in, storm_hours)
            except Exception:
                # compile échoue à l'exécution → bascule eager définitivement
                self._fwd_compiled = self._forward_impl
                return self._forward_impl(P, tmin, tmax, Rn, u2, ea, doy, state, tmin_j, tmax_j, sw_in, storm_hours)
        return self._forward_impl(P, tmin, tmax, Rn, u2, ea, doy, state, tmin_j, tmax_j, sw_in, storm_hours)

    def _forward_impl(self, P, tmin, tmax, Rn, u2, ea, doy, state: HydrotelColumnState,
                      tmin_j=None, tmax_j=None, sw_in=None, storm_hours=None) -> tuple[Tensor, HydrotelColumnState, dict]:
        """Un pas de temps. P/tmin/tmax/Rn/u2/ea : forçage (n_nodes,). doy : jour
        julien (scalaire ou n_nodes). Retourne (prod_totale_mm, new_state, diag)."""
        assert self._static is not None, "appeler set_static() avant forward()"
        ps, pso, pe = self._static["snow"], self._static["soil"], self._static["etr"]
        tmin_j = tmin if tmin_j is None else tmin_j
        tmax_j = tmax if tmax_j is None else tmax_j
        doy_t = (doy.to(P.dtype) if torch.is_tensor(doy)
                 else torch.tensor(float(doy), dtype=P.dtype, device=P.device))

        # 1. split pluie/neige → fonte neige → apport
        pluie, neige = self._split_precip(P, tmin, tmax)
        apport, snow_new = self.snow(tmin, tmax, pluie, neige, doy_t, state.snow, ps, sw_in=sw_in)
        # hauteur agrégée du couvert nival [m] pour le gel
        haut = sum(ps[f"pct_{c}" if c != "decouver" else "pct_autres"] * snow_new[c][1]
                   for c in DegreJourModifie.CLASSES)

        # 2. gel RANKINEN → profondeur de gel [cm]
        if self.use_frost:
            frost_profile, prof_gel_cm = self.frost(
                tmin, tmax, haut, state.frost_profile,
                pe["z11"], pe["z22"], pe["z33"])
        else:
            frost_profile = state.frost_profile
            prof_gel_cm = torch.zeros_like(P)

        # 3. ETP × K_c (coefficient cultural NeRF par nœud) — corrige le biais
        # McGuinness et donne au NeRF un levier direct sur le volume (β).
        # K_c=1.0 par défaut si non fourni (chemins set_static hand-built).
        etp = self._etp(tmin_j, tmax_j, Rn, u2, ea, ps["lat"], doy_t) * pe.get("K_c", 1.0)

        # 4. ETR par couche (sur theta DÉBUT de pas). Phénologie interpolée en
        # TORCH (breakpoints cachés en tenseurs) — plus de np.interp ni de synchro
        # float(doy) par pas de temps. Résultats identiques (interp linéaire clampé).
        pheno = self._pheno_tensors(pe["classes"], P)
        d = doy_t.reshape(-1)[0]                  # jour julien scalaire (tenseur, sans synchro)
        etp_classes, roots, leaves = [], [], []
        for (pct, jbp_t, leaf_t, root_t) in pheno:
            etp_classes.append(etp * pct / 1000.0)
            roots.append(_interp1d(d, jbp_t, root_t).expand_as(P))
            leaves.append(_interp1d(d, jbp_t, leaf_t).expand_as(P))
        e1, e2, e3 = calcule_etr(state.theta1, state.theta2, state.theta3,
                                 etp_classes, roots, leaves, pe["thetacc"], pe["thetapf"],
                                 pe["alpha"], pe["z11"], pe["z22"], pe["z33"], pe["des"], pe["coef_assech"])

        # 5. bilan sol BV3C2 (porte gel via prof_gel_cm + couvert nival)
        couvert_mm = snow_new["couvert_nival_mm"]
        # Durée d'orage EFFECTIVE pondérée par la masse pluie vs fonte : seule la
        # PLUIE porte l'intensité convective DT_eff ; la fonte s'écoule sur 24 h.
        # Sans ça, un jour de fonte + averse courte fait passer toute la lame de
        # fonte dans le cap hortonien -> horton fictif massif au freshet (revue
        # 2026-07-01). apport = pluie (si pas de neige) ou fonte+pluie relâchées.
        if storm_hours is not None:
            melt_mm = torch.clamp(apport - pluie, min=0.0)
            w_rain = pluie / torch.clamp(pluie + melt_mm, min=1e-6)
            storm_hours = w_rain * storm_hours + (1.0 - w_rain) * 24.0
        ps_surf, ph, pb, rech, (t1, t2, t3), sdiag = self.soil(
            state.theta1, state.theta2, state.theta3, apport, etp, prof_gel_cm, couvert_mm, pso,
            etr1_mm=e1 * 1000.0, etr2_mm=e2 * 1000.0, etr3_mm=e3 * 1000.0, storm_hours=storm_hours)
        prod = ps_surf + ph + pb   # mm

        # 5b. Hydrogramme de VERSANT (cascade de Nash, fidèle Hydrotel, porté de
        # column.py). Lisse les composantes RAPIDES par étalement des temps de
        # parcours de versant AVANT le canal : surface POINTUE (k court, pic
        # préservé) + interflow LARGE (k long), baseflow direct. À coupler avec
        # pure_advection (canal cinématique). Sans lui, le Muskingum diffusif lisse
        # au mauvais endroit (pansement, cf 2026-06-27). forward sans le flag =
        # inchangé (fidélité 370.9 préservée).
        uh1n = uh2n = uh3n = uh4n = None
        if self.use_hillslope_uh:
            def _nash(inflow, s1, s2, log_k):
                if s1 is None: s1 = torch.zeros_like(inflow)
                if s2 is None: s2 = torch.zeros_like(inflow)
                k = torch.nn.functional.softplus(log_k) + 0.05      # jours
                a = 1.0 - torch.exp(-1.0 / k)                       # relâché/jour
                s1n = s1 + inflow; o1 = s1n * a; s1_new = s1n - o1
                s2n = s2 + o1;     o2 = s2n * a; s2_new = s2n - o2
                return o2, s1_new, s2_new
            surf_out, uh1n, uh2n = _nash(ps_surf, state.uh_s1, state.uh_s2, self.log_uh_k_surf)
            inter_out, uh3n, uh4n = _nash(ph, state.uh_s3, state.uh_s4, self.log_uh_k_inter)
            prod = surf_out + inter_out + pb

        # 6. milieu humide isolé (optionnel). production_surf/hypo/base.csv d'Hydrotel
        # est POST-MH (bv3c2.cpp l.838-895) : la prod totale passe dans le réservoir
        # SWAT puis prod = prodOld·(1−wetdrafr) + wetprod. Vectorisé sur tous les nœuds,
        # masqué (wmask) pour que les nœuds sans MH soient un no-op EXACT.
        wet_vol = state.wet_vol
        if self._static["wetland"] is not None:
            w = self._static["wetland"]
            apport_w = apport * w["wet_fr_area"]   # apport × fraction superficie wetland
            # clamp wet_vol>0 : évite 0^A=NaN dans le gradient (volume physique, pas
            # de masquage ; les vrais nœuds MH ont wet_vol>0, ce plancher est négligeable)
            vol_in = torch.clamp(state.wet_vol, min=1e-9)
            wet_vol_n, wsep, wflwi, wflwo, wprod = calcul_milieu_humide_isole(
                vol_in, apport_w, etp, prod, w["hru_ha"], w["wet_dra_fr"],
                w["A"], w["B"], w["wetnvol"], w["wetmxvol"], w["wet_k"], w["c_ev"], w["c_prod"])
            prod_w = prod * (1.0 - w["wet_dra_fr"]) + wprod
            prod = torch.where(w["wmask"], prod_w, prod)
            wet_vol = torch.where(w["wmask"], wet_vol_n, state.wet_vol)

        new_state = HydrotelColumnState(t1, t2, t3, snow_new, frost_profile, wet_vol,
                                        uh1n, uh2n, uh3n, uh4n)
        etr_tot = (e1 + e2 + e3) * 1000.0
        diag = dict(apport=apport, etp=etp, etr1=e1 * 1000.0, etr2=e2 * 1000.0, etr3=e3 * 1000.0,
                    prof_gel_cm=prof_gel_cm, couvert_nival_mm=couvert_mm,
                    prod_surf=ps_surf, prod_hypo=ph, prod_base=pb,
                    # clés attendues par model.py / SimDiagnostics
                    etr=etr_tot, snowmelt=apport, lateral_mm=prod)
        return prod, new_state, diag


# ── Cycles annuels par défaut (physio/ind_fol.def, pro_rac.def DELISLE) ──
_JBP = [1, 100, 135, 166, 180, 210, 244, 270, 274, 280, 365]
_LEAF = {"feuillus": [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3], "ouverts": [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 1],
         "humides": [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2], "conifers": [5] * 11,
         "mixtes": [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3], "agri": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0],
         # végétation générique boréal QC (défaut quand l'occupation manque) :
         # LAI saisonnier modéré, racine ~1.2 m (cf pro_rac.def SLSO forêt = 1.26)
         "default": [2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 2]}
_ROOT = {"feuillus": [1.5] * 11, "ouverts": [0.5] * 11, "humides": [0.75] * 11,
         "conifers": [1.0] * 11, "mixtes": [1.25] * 11,
         "agri": [0, 0, 0.3, 0.55, 0.7, 0.8, 0.8, 0.8, 0.3, 0, 0],
         "default": [1.2] * 11}


def build_static_params(n_nodes, lat, slope, orientation, texture, z, occupation,
                        device="cpu", dtype=torch.float64):
    """Construit les 3 dicts de params statiques par nœud à partir de descripteurs
    simples. SEAM Phase A : le NeRF/territorial fournira le sous-ensemble apprenable
    (texture/profondeurs/occupation régionalisés) à la place de ces scalaires.

    occupation : dict {classe: pourcentage} pour conifers/feuillus/mixtes/agri/
        urbain/routes/ouverts/eau/sols_nus/humides. eau→fse, urbain+routes→fsi,
        reste→fsa. Les classes ET = perméables non nulles (hors eau/imperm).
    texture : nom dans SOIL_TEXTURES. z = (z1,z2,z3) épaisseurs [m]."""
    T = lambda v: torch.full((n_nodes,), float(v), device=device, dtype=dtype)
    ce1, ce0 = init_ce(T(lat), T(slope), T(orientation))
    occ = {k: occupation.get(k, 0.0) for k in
           ("conifers", "feuillus", "mixtes", "agri", "urbain", "routes", "ouverts", "eau", "sols_nus", "humides")}
    pct_conif = occ["conifers"]
    pct_feu = occ["feuillus"] + occ["mixtes"]   # CLASSE INTEGRE FEUILLUS = feuillus + mixtes (DELISLE)
    p_snow = dict(lat=T(lat), ce1=ce1, ce0=ce0,
                  pct_conifers=T(pct_conif), pct_feuillus=T(pct_feu),
                  pct_autres=T(max(1.0 - pct_conif - pct_feu, 0.0)),
                  coeff_fonte_conifers=T(.012), coeff_fonte_feuillus=T(.014), coeff_fonte_decouver=T(.016),
                  seuil_fonte_conifers=T(0.0), seuil_fonte_feuillus=T(0.0), seuil_fonte_decouver=T(0.0),
                  taux_fonte_geo=T(0.5), densite_max=T(466.0), constante_tassement=T(0.1))

    fse = occ["eau"]
    fsi = occ["urbain"] + occ["routes"]
    fsa = max(1.0 - fse - fsi, 0.0)
    p_soil = make_params(texture, texture, texture, slope=slope, fsa=fsa, fse=fse, fsi=fsi,
                         krec=1e-6, cin=0.3, coef_recharge=0.0, device=device)
    for i in (1, 2, 3):
        p_soil[f"z{i}"] = T(z[i - 1])

    tx = SOIL_TEXTURES[texture]
    et_classes = []
    for c in ("conifers", "feuillus", "mixtes", "agri", "ouverts", "sols_nus", "humides"):
        pct = occ.get(c if c != "feuillus" else "feuillus", 0.0)
        if c in _LEAF and pct > 0:
            et_classes.append((pct, _JBP, _LEAF[c], _ROOT[c]))
    p_etr = dict(thetacc=T(tx["thetacc"]), thetapf=T(tx["thetapf"]), alpha=T(_TEXTURE_ALPHA[texture]),
                 des=T(0.6), coef_assech=T(1.0), z11=z[0], z22=z[1], z33=z[2], classes=et_classes)
    return p_snow, p_soil, p_etr


# alpha (assèchement ETR) par texture, proprietehydrolique.sol
_TEXTURE_ALPHA = {"sand": 10.0, "loamy_sand": 6.0, "sandy_loam": 4.5, "loam": 3.5,
                  "silt_loam": 3.0, "clay": 0.5, "peat": 6.0}
