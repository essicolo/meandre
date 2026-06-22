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

    def detach(self) -> "HydrotelColumnState":
        sn = {}
        for k, v in self.snow.items():
            sn[k] = tuple(t.detach() for t in v) if isinstance(v, tuple) else v.detach()
        return HydrotelColumnState(
            self.theta1.detach(), self.theta2.detach(), self.theta3.detach(),
            sn, self.frost_profile.detach(), self.wet_vol.detach())


class HydrotelColumn(nn.Module):
    """Colonne verticale fidèle Hydrotel, un pas de temps journalier, vectorisée."""

    def __init__(self, et_mode: str = "mcguinness", use_frost: bool = True,
                 soil_n_substep: int = 48, frost_intervalle: float = 0.05,
                 frost_temp_ini: float = 4.0, frost_seuil: float = -0.5,
                 frost_fs: float = 2.35, frost_kt: float = 0.8,
                 frost_cs: float = 1.0e6, frost_cice: float = 4.0e6,
                 t_neige_seuil: float = 0.0, compile_soil: bool = False) -> None:
        super().__init__()
        self.et_mode = str(et_mode)
        self.use_frost = bool(use_frost)
        self.t_neige_seuil = t_neige_seuil   # seuil pluie/neige (split de phase, TODO: règle Hydrotel exacte)
        self.snow = DegreJourModifie(pas_de_temps=24)
        self.frost = Rankinen(frost_intervalle, frost_temp_ini, frost_seuil, frost_fs,
                              frost_kt, frost_cs, frost_cice, pas_de_temps=24)
        # compile_soil : boucle de sous-pas STATIQUE (sans break) + torch.compile.
        # ~7× plus rapide sur GPU (fusion, supprime la synchro bool().all() par
        # itération), résultats IDENTIQUES au mode break à n_substep égal (vérifié).
        # Coût : ~40s de compilation au 1er pas. Requiert un backend (Triton/WSL).
        self.soil = BV3C2Clone(n_substep=soil_n_substep, static=bool(compile_soil))
        if compile_soil:
            self.soil = torch.compile(self.soil, dynamic=False)
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
                      constante_tassement=torch.full_like(like, 0.1))

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

        # ETR : thetacc/thetapf du NeRF (couche 1), alpha global ; classes dispo
        alpha = torch.exp(self.log_etr_alpha)
        et_classes = []
        if pct_feu.sum() > 0:
            et_classes.append((pct_feu, _JBP, _LEAF["feuillus"], _ROOT["feuillus"]))
        if pct_conif.sum() > 0:
            et_classes.append((pct_conif, _JBP, _LEAF["conifers"], _ROOT["conifers"]))
        if f_wet.sum() > 0:
            et_classes.append((f_wet, _JBP, _LEAF["humides"], _ROOT["humides"]))
        p_etr = dict(thetacc=sp.theta_fc_1, thetapf=sp.theta_wp_1, alpha=alpha * torch.ones_like(like),
                     des=torch.full_like(like, 0.6), coef_assech=torch.full_like(like, 1.0),
                     z11=self.z1, z22=sp.Z2.mean().item(), z33=sp.Z3.mean().item(), classes=et_classes)

        # milieu humide isolé : actif SI le territorial porte la géométrie par nœud
        # (wet_a_raw). Sinon None (colonne sol seul, ex SLSO). Masqué + sûr gradient.
        wetland = self._wetland_from_territorial(territorial, like)

        n_depth = n_intervalles(self.z1 + float(sp.Z2.mean()) + float(sp.Z3.mean()), self.frost.dz)
        self.set_static(p_snow, p_soil, p_etr, wetland=wetland, n_depth=n_depth)
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
        return HydrotelColumnState(
            theta1=z(theta_init[0]), theta2=z(theta_init[1]), theta3=z(theta_init[2]),
            snow=sn, frost_profile=frost_profile, wet_vol=torch.zeros(n_nodes, device=device, dtype=dtype))

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

    def column_step(self, enriched, state, doy=None, return_diagnostics=False, **_):
        """Un pas, interface ColumnOutput. enriched[:, :6] = P,Tmin,Tmax,Rn,u2,ea.
        theta est re-synchronisé depuis `state` (pour intégrer une éventuelle
        correction résiduelle), le reste de l'état riche est interne (self._aux)."""
        from meandre.vertical.column import ColumnOutput
        a = self._aux
        a.theta1, a.theta2, a.theta3 = state.theta1, state.theta2, state.theta3
        P, tmin, tmax = enriched[:, 0], enriched[:, 1], enriched[:, 2]
        Rn, u2, ea = enriched[:, 3], enriched[:, 4], enriched[:, 5]
        prod, a, diag = self.forward(P, tmin, tmax, Rn, u2, ea,
                                     doy if doy is not None else 1, a)
        self._aux = a
        new_state = HydroState(
            theta1=a.theta1, theta2=a.theta2, theta3=a.theta3,
            swe=diag["couvert_nival_mm"], t_soil=a.frost_profile[:, 0],
            canopy_storage=state.canopy_storage, wetland_storage=state.wetland_storage,
            S_gw=state.S_gw, T_water=state.T_water,
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

    def forward(self, P, tmin, tmax, Rn, u2, ea, doy, state: HydrotelColumnState,
                tmin_j=None, tmax_j=None) -> tuple[Tensor, HydrotelColumnState, dict]:
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
        apport, snow_new = self.snow(tmin, tmax, pluie, neige, doy_t, state.snow, ps)
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

        # 3. ETP
        etp = self._etp(tmin_j, tmax_j, Rn, u2, ea, ps["lat"], doy_t)

        # 4. ETR par couche (sur theta DÉBUT de pas)
        etp_classes, roots, leaves = [], [], []
        import numpy as _np
        jr = float(doy) if not torch.is_tensor(doy) else float(doy_t.flatten()[0])
        for (pct, jbp, leaf_bp, root_bp) in pe["classes"]:
            etp_classes.append(etp * pct / 1000.0)
            roots.append(torch.full_like(P, float(_np.interp(jr, jbp, root_bp))))
            leaves.append(torch.full_like(P, float(_np.interp(jr, jbp, leaf_bp))))
        e1, e2, e3 = calcule_etr(state.theta1, state.theta2, state.theta3,
                                 etp_classes, roots, leaves, pe["thetacc"], pe["thetapf"],
                                 pe["alpha"], pe["z11"], pe["z22"], pe["z33"], pe["des"], pe["coef_assech"])

        # 5. bilan sol BV3C2 (porte gel via prof_gel_cm + couvert nival)
        couvert_mm = snow_new["couvert_nival_mm"]
        ps_surf, ph, pb, rech, (t1, t2, t3), sdiag = self.soil(
            state.theta1, state.theta2, state.theta3, apport, etp, prof_gel_cm, couvert_mm, pso,
            etr1_mm=e1 * 1000.0, etr2_mm=e2 * 1000.0, etr3_mm=e3 * 1000.0)
        prod = ps_surf + ph + pb   # mm

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

        new_state = HydrotelColumnState(t1, t2, t3, snow_new, frost_profile, wet_vol)
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
         "mixtes": [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 3], "agri": [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0]}
_ROOT = {"feuillus": [1.5] * 11, "ouverts": [0.5] * 11, "humides": [0.75] * 11,
         "conifers": [1.0] * 11, "mixtes": [1.25] * 11,
         "agri": [0, 0, 0.3, 0.55, 0.7, 0.8, 0.8, 0.8, 0.3, 0, 0]}


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
