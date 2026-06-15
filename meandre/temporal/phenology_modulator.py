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
import torch
import torch.nn as nn
from torch import Tensor


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
        k_c_min_init: float = 0.3,
        k_c_max_factor_init: float = 1.0,
        sharpness_emerg: float = 50.0,
        sharpness_senesc: float = 100.0,
        senesc_offset: float = 600.0,
    ) -> None:
        super().__init__()
        # Paramètres appris (4)
        self.gdd_emerg = nn.Parameter(torch.tensor(float(gdd_emerg_init)))
        self.gdd_mid = nn.Parameter(torch.tensor(float(gdd_mid_init)))
        self.k_c_min = nn.Parameter(torch.tensor(float(k_c_min_init)))
        self.k_c_max_factor = nn.Parameter(torch.tensor(float(k_c_max_factor_init)))
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

    def forward(self, gdd_cum: Tensor, K_c_base: Tensor) -> Tensor:
        """Modulateur K_c effectif au temps t.

        gdd_cum  : (N,) ou (T, N) — GDD cumulé
        K_c_base : (N,) — K_c de référence par nœud (sortie NeRF)
        Returns  : K_c_eff même forme que gdd_cum, en respectant l'unité de K_c_base
        """
        shape = self.shape(gdd_cum)                                       # ∈ [0, 1]
        # Born K_c_min ≥ 0.05 (floor strict), K_c_max_factor ≥ 0.5 (pas de réduction excessive)
        kc_min_safe = self.k_c_min.clamp(min=0.05, max=1.0)
        kc_max_safe = self.k_c_max_factor.clamp(min=0.5, max=2.0)
        K_c_eff = kc_min_safe + (kc_max_safe * K_c_base - kc_min_safe) * shape
        return K_c_eff.clamp(min=0.05)                                    # safety floor

    def extra_repr(self) -> str:
        return (f"GDD_emerg={self.gdd_emerg.item():.1f}, "
                f"GDD_mid={self.gdd_mid.item():.1f}, "
                f"K_c_min={self.k_c_min.item():.3f}, "
                f"K_c_max_factor={self.k_c_max_factor.item():.3f}")


def update_gdd_cum(
    gdd_cum_prev: Tensor, T_mean: Tensor, doy: int | Tensor, T_base: float = 10.0,
) -> Tensor:
    """Update GDD cumulé pour un pas de temps.

    Si doy == 1 (1er janvier), reset à 0. Sinon ajoute relu(T_mean - T_base).

    Parameters
    ----------
    gdd_cum_prev : (N,) GDD cumulé du jour précédent
    T_mean : (N,) température moyenne aujourd'hui (°C)
    doy : int ou (1,) tensor, jour de l'année (1-366)
    T_base : float, seuil base (°C). Default 10 (standard agronomique).

    Returns
    -------
    gdd_cum_new : (N,) GDD cumulé après mise à jour
    """
    dgd = torch.relu(T_mean - T_base)
    doy_val = doy if isinstance(doy, int) else int(doy.item())
    if doy_val == 1:
        return dgd                                                        # reset annuel
    return gdd_cum_prev + dgd
