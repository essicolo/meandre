"""PhenologyModulator — modulation temporelle de K_c par GDD cumulé.

Remplace la modulation hardcodée `phenology = σ((T_air-5)/2) × exp(-SWE/10)`
du vertical_column par une fonction APPRISE conditionnée sur GDD cumulé
(growing degree days, indicateur agronomique standard).

Architecture : double logistique (Korva-Mol 1968, FAO-56 Allen 1998)
  - rampe d'émergence : sigmoide montante autour de GDD_emerg
  - plateau au stade reproductif
  - sénescence : sigmoide descendante après GDD_mid + offset

Forme apprise :
    shape(GDD) = σ((GDD - GDD_emerg) / 50) × σ(-(GDD - GDD_mid - 600) / 100)
    K_c_eff(t, n) = K_c_min + (K_c_max_factor × K_c_base(n) - K_c_min) × shape(GDD)

4 paramètres apprenables, **tous nommés et physiquement interprétables** :
  - GDD_emerg  (°C·j) : seuil d'émergence végétation
  - GDD_mid    (°C·j) : milieu du plateau pic LAI
  - K_c_min    (sans dim) : floor sol nu, dormance
  - K_c_max_factor (sans dim) : amplificateur max sur K_c_base (typique 1.0-1.2)

Init : valeurs littérature pour forêt boréale (zone tempérée Québec) :
  GDD_emerg ≈ 150 (débourrement)
  GDD_mid   ≈ 800 (mi-saison)
  K_c_min   ≈ 0.3 (dormance hivernale)
  K_c_max_factor ≈ 1.0 (pas d'amplification au-delà du K_c littérature)

Validable contre :
  - MODIS NDVI / LAI (Glenn 2011, Calera 2017)
  - phénologie eddy covariance (FluxNet)
  - dates débourrement Environnement Canada

Ref : Allen 1998 FAO-56, Glenn et al. 2011, Hatfield & Dold 2018.
"""
from __future__ import annotations
import math
import os

import torch
import torch.nn as nn
from torch import Tensor
from meandre.utils import settings


class PhenologyModulator(nn.Module):
    """K_c modulé par GDD via double logistique apprenable.

    Parameters
    ----------
    gdd_emerg_init : float
        GDD cumulé seuil d'émergence (°C·j). Défaut 150 (forêt boréale Québec).
    gdd_mid_init : float
        GDD cumulé milieu plateau (°C·j). Défaut 800.
    k_c_min_init : float
        K_c floor en dormance. Défaut 0.3.
    k_c_max_factor_init : float
        Amplificateur max sur K_c_base. Défaut 1.0 (pas d'amplification).
    sharpness_emerg : float
        Pente sigmoide émergence (°C·j). Default 50. Plus grand = transition douce.
    sharpness_senesc : float
        Pente sigmoide sénescence (°C·j). Default 100.
    senesc_offset : float
        Décalage en °C·j entre GDD_mid et début sénescence. Default 600.
    """

    def __init__(
        self,
        # Init révisée 2026-06-06 pour SLSO (forêt boréale Québec)
        # GDD_emerg=80 : débourrement vers début mai (Aulne, Bouleau)
        # GDD_mid=600  : pic LAI mi-juillet
        # Cohérent avec Cleland et al. 2007, Chen & Ahmad 2015
        gdd_emerg_init: float = 80.0,
        gdd_mid_init: float = 600.0,
        k_c_min_init: float | None = None,
        k_c_max_factor_init: float | None = None,
        sharpness_emerg: float = 50.0,
        sharpness_senesc: float = 100.0,
        senesc_offset: float = 600.0,
        params=None,
    ) -> None:
        """`params` vient de la configuration déclarée ([phenology] du TOML) ; absent, le
        réglage phenology.params s'applique (meandre/utils/settings.py)."""
        super().__init__()
        # Valeurs de départ absentes : réglages déclarés de [phenology_constants].
        if k_c_min_init is None:
            k_c_min_init = settings.get("phenology_constants.kc_min_init")
        if k_c_max_factor_init is None:
            k_c_max_factor_init = settings.get("phenology_constants.kc_max_factor_init")
        # Paramètres appris (4)
        self.gdd_emerg = nn.Parameter(torch.tensor(float(gdd_emerg_init)))
        self.gdd_mid = nn.Parameter(torch.tensor(float(gdd_mid_init)))
        self.k_c_min = nn.Parameter(torch.tensor(float(k_c_min_init)))
        self.k_c_max_factor = nn.Parameter(torch.tensor(float(k_c_max_factor_init)))
        # SÉNESCENCE PAR LA PHOTOPÉRIODE (2026-09-28, suggestion d'Essi) : la chute d'automne n'est plus déclenchée par un cumul de degrés-jours mais par
        # la longueur du jour, qui décroît après le solstice ; c'est le déclencheur
        # principal de la sénescence des feuillus tempérés (Delpierre et al. 2009). Seuil
        # de longueur du jour photo_crit_h = 12 + 3·tanh(photo_crit), initialisé à 12 h.
        self.mode = "photo"
        self.photo_crit = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("photo_pente_h", torch.tensor(0.5))
        # PARAMÈTRES CALÉS SUR MODIS (2026-09-28). MEANDRE_PHENOLOGIE_PARAMS =
        # « emerg,crit,s1,s2 » : seuil de débourrement (degrés-jours), seuil de longueur du
        # jour de la sénescence (heures), et les deux largeurs de transition. Ces valeurs
        # sont ajustées sur l'indice foliaire observé (`.runs/quebec/caler_phenologie_modis.py`)
        # et GELÉES : la saison vient d'une observation indépendante du débit, et reste
        # pilotée par la seule météo, donc utilisable en prédiction.
        _cal = params if params is not None else settings.get("phenology.params")
        if _cal:
            _vals = _cal.split(",") if isinstance(_cal, str) else list(_cal)
            e, c, s1, s2 = [float(x) for x in _vals]
            with torch.no_grad():
                self.gdd_emerg.fill_(e)
                _pc, _pr = float(settings.get("phenology_constants.photoperiod_center_h")), float(settings.get("phenology_constants.photoperiod_half_range_h"))
                self.photo_crit.fill_(math.atanh(max(min((c - _pc) / _pr, 0.999), -0.999)))
            self.gdd_emerg.requires_grad_(False)
            self.photo_crit.requires_grad_(False)
            self.gdd_mid.requires_grad_(False)
            self._s1_cal, self._s2_cal = s1, s2
        # Hyperparamètres fixes (largeurs des transitions, non appris)
        self.register_buffer("sharpness_emerg", torch.tensor(float(sharpness_emerg)))
        self.register_buffer("sharpness_senesc", torch.tensor(float(sharpness_senesc)))
        self.register_buffer("senesc_offset", torch.tensor(float(senesc_offset)))

    def shape(self, gdd_cum: Tensor) -> Tensor:
        """Calcule la forme phénologique (0 = dormant, 1 = pic croissance).

        gdd_cum : (N,) ou (T, N) — GDD cumulé en °C·j
        Returns : même shape, ∈ [0, ~1]
        """
        ramp = torch.sigmoid((gdd_cum - self.gdd_emerg) / self.sharpness_emerg)
        senesc = torch.sigmoid(-(gdd_cum - self.gdd_mid - self.senesc_offset) / self.sharpness_senesc)
        return ramp * senesc

    @staticmethod
    def day_length(lat_deg: Tensor, doy: int) -> Tensor:
        """Longueur du jour en heures (déclinaison de Cooper, angle horaire au coucher)."""
        decl = math.radians(23.44) * math.sin(2.0 * math.pi * (284 + doy) / 365.0)
        x = -torch.tan(torch.deg2rad(lat_deg)) * math.tan(decl)
        return 24.0 / math.pi * torch.arccos(x.clamp(-1.0, 1.0))

    def shape_value(self, gdd_cum: Tensor, doy: int | None = None, lat_deg: Tensor | None = None) -> Tensor:
        """Forme saisonnière de la végétation, de 0 (dormance) à 1 (plateau) : levée par les
        degrés-jours, sénescence par la longueur du jour ; sans jour ni latitude, la double
        logistique en degrés-jours."""
        if doy is not None and lat_deg is not None:
            s1 = getattr(self, "_s1_cal", None) or self.sharpness_emerg
            s2 = getattr(self, "_s2_cal", None) or self.photo_pente_h
            ramp = torch.sigmoid((gdd_cum - self.gdd_emerg) / s1)
            if doy > int(settings.get("phenology_constants.senescence_start_doy")):
                seuil = float(settings.get("phenology_constants.photoperiod_center_h")) + float(settings.get("phenology_constants.photoperiod_half_range_h")) * torch.tanh(self.photo_crit)
                senesc = torch.sigmoid((self.day_length(lat_deg, doy) - seuil) / s2)
            else:
                senesc = torch.ones_like(ramp)
            shape = ramp * senesc
        else:
            shape = self.shape(gdd_cum)                                   # ∈ [0, 1]
        return shape

    def crop_kc(self, gdd_cum: Tensor, K_c_base: Tensor, params, doy: int | None = None, lat_deg: Tensor | None = None) -> Tensor:
        """K_c effectif des grandes cultures (2026-11-10) : même dormance et même plateau que la
        forme principale (k_c_min, k_c_max_factor appris), mais levée et sénescence propres,
        déclarées (« levée en degrés-jours, seuil de longueur du jour en heures, largeur de la
        levée, largeur de la sénescence ») et non apprises. Un maïs ou un soya lève fin mai et
        ne couvre le sol qu'en juillet (FAO-56) ; la forme des feuillus monte dès la mi-mai."""
        e, c, s1, s2 = (float(x) for x in params)
        ramp = torch.sigmoid((gdd_cum - e) / s1)
        if doy is not None and lat_deg is not None and doy > int(settings.get("phenology_constants.senescence_start_doy")):
            senesc = torch.sigmoid((self.day_length(lat_deg, doy) - c) / s2)
        else:
            senesc = torch.ones_like(ramp)
        return self.kc_from_shape(ramp * senesc, K_c_base)

    def forward(self, gdd_cum: Tensor, K_c_base: Tensor, doy: int | None = None, lat_deg: Tensor | None = None) -> Tensor:
        """Modulateur K_c effectif au temps t.

        gdd_cum  : (N,) ou (T, N) — GDD cumulé
        K_c_base : (N,) — K_c de référence par nœud (sortie NeRF)
        Returns  : K_c_eff même forme que gdd_cum, en respectant l'unité de K_c_base
        """
        return self.kc_from_shape(self.shape_value(gdd_cum, doy, lat_deg), K_c_base)

    def kc_from_shape(self, shape: Tensor, K_c_base: Tensor) -> Tensor:
        """K_c effectif d'une forme saisonnière de 0 (dormance) à 1 (plateau)."""
        # Bornes déclarées de [phenology_constants] : K_c_min entre 0,05 et 1, K_c_max_factor entre 0,5 et 2 ; la borne basse de K_c_min sert aussi de plancher au K_c effectif.
        _kmin_lo, _kmin_hi = settings.get("phenology_constants.kc_min_bounds")
        _kmax_lo, _kmax_hi = settings.get("phenology_constants.kc_max_factor_bounds")
        kc_min_safe = self.k_c_min.clamp(min=_kmin_lo, max=_kmin_hi)
        kc_max_safe = self.k_c_max_factor.clamp(min=_kmax_lo, max=_kmax_hi)
        K_c_eff = kc_min_safe + (kc_max_safe * K_c_base - kc_min_safe) * shape
        return K_c_eff.clamp(min=_kmin_lo)                                    # plancher de sécurité

    def extra_repr(self) -> str:
        return (f"GDD_emerg={self.gdd_emerg.item():.1f}, "
                f"GDD_mid={self.gdd_mid.item():.1f}, "
                f"K_c_min={self.k_c_min.item():.3f}, "
                f"K_c_max_factor={self.k_c_max_factor.item():.3f}")


def update_gdd_cum(
    gdd_cum_prev: Tensor, T_mean: Tensor, doy: int | Tensor, T_base: float | None = None,
) -> Tensor:
    """Update GDD cumulé pour un pas de temps.

    Si doy == 1 (1er janvier), reset à 0. Sinon ajoute relu(T_mean - T_base).

    Parameters
    ----------
    gdd_cum_prev : (N,) GDD cumulé du jour précédent
    T_mean : (N,) température moyenne aujourd'hui (°C)
    doy : int ou (1,) tensor, jour de l'année (1-366)
    T_base : float, seuil base (°C). Absent, le réglage phenology_constants.gdd_base_c (10, standard agronomique).

    Returns
    -------
    gdd_cum_new : (N,) GDD cumulé après mise à jour
    """
    if T_base is None:
        T_base = float(settings.get("phenology_constants.gdd_base_c"))
    dgd = torch.relu(T_mean - T_base)
    doy_val = doy if isinstance(doy, int) else int(doy.item())
    if doy_val == 1:
        return dgd                                                        # reset annuel
    return gdd_cum_prev + dgd
