"""Clone FIDÈLE du modèle de température du sol / gel RANKINEN d'Hydrotel, porté
ligne-à-ligne du C++ (source/rankinen.cpp), vectorisé sur les nœuds,
différentiable. Sous-projet PROPRE.

RANKINEN (Rankinen et al. 2004) : profil de température du sol discrétisé
(intervalle Δ, p.ex. 0.05 m, du sommet 0 à la base Z11+Z22+Z33). À chaque pas,
chaque nœud de profondeur relaxe vers la température de l'air à un taux atténué
en profondeur (∝ KT/(CA·(2z)²)), puis est amorti par l'isolation de la neige
(× exp(−FS·Ds), Ds = HAUTEUR du couvert nival en m). La profondeur de gel est la
profondeur où T franchit le seuil de gel (interpolée linéairement).

PAS de couplage entre nœuds de profondeur (relaxation indépendante vers Tair), ce
qui rend le profil trivialement vectorisable. Équations 1-2 : rankinen.cpp:241-246.

NB : Hydrotel SLSO/DELISLE ne l'activent pas par défaut (TEMPERATURE DU SOL vide).
Cloné pour une intégration CORRECTE du gel dans méandre (le frost.py historique
de méandre était problématique).
"""
from __future__ import annotations
import torch
from torch import Tensor


def n_intervalles(profondeur_totale: float, intervalle: float) -> int:
    """Nombre de nœuds du profil (rankinen.cpp:100-107)."""
    n = int(profondeur_totale / intervalle)
    frac = profondeur_totale / intervalle - n
    return n + (2 if frac > 0.0 else 1)


class Rankinen(torch.nn.Module):
    """Profil de température du sol RANKINEN, gel en sortie. Vectorisé sur les
    nœuds ; le profil de profondeur est un axe (n_nodes, n_depth)."""

    def __init__(self, intervalle=0.05, temp_ini_base=4.0, seuil_gel=-0.5, fs=2.35,
                 kt=0.8, cs=1.0e6, cice=4.0e6, pas_de_temps=24):
        super().__init__()
        self.dz = intervalle
        self.temp_ini_base = temp_ini_base
        self.seuil_gel = seuil_gel
        self.fs = fs
        self.kt = kt
        self.cs = cs
        self.cice = cice
        self.dt = pas_de_temps * 3600.0

    def init_profil(self, tmin, tmax, snow_depth_m, n_depth):
        """Profil initial linéaire (rankinen.cpp:168-193) : de la surface fB
        (Tair amortie par la neige) à la base temp_ini_base, sur la profondeur."""
        tair = (tmin + tmax) / 2.0
        fB = torch.where(snow_depth_m != 0.0, tair * torch.exp(-self.fs * snow_depth_m), tair)
        prof_tot = (n_depth - 1) * self.dz
        fM = (self.temp_ini_base - fB) / prof_tot                  # (n_nodes,)
        depths = torch.arange(n_depth, dtype=tair.dtype, device=tair.device) * self.dz  # (n_depth,)
        return fM[:, None] * depths[None, :] + fB[:, None]         # (n_nodes, n_depth)

    def forward(self, tmin, tmax, snow_depth_m, profil, z11, z22, z33,
                kt=None, cs=None, fs=None):
        """Un pas de temps. tmin/tmax [°C], snow_depth_m = hauteur couvert nival [m],
        profil (n_nodes, n_depth) = température aux profondeurs i·Δ. z11/z22/z33 :
        épaisseurs des couches.

        kt / cs / fs : PROPRIÉTÉS THERMIQUES PAR NŒUD (conductivité W/m/K, capacité
        volumique J/m3/K, amortissement nival 1/m). Absentes, on retombe sur les
        scalaires globaux du C++, qui est le clone fidèle. Elles existent parce que le
        gel doit dépendre du SOL et pas seulement de l'air : avec des scalaires
        uniformes, une argile saturée et un sable sec gèlent identiquement, alors que
        leurs propriétés diffèrent d'un facteur trois à quatre et que la chaleur latente
        de l'eau domine le bilan (remarque d'Essi, 2026-08-27).
        Retourne (profil_new, profondeur_gel_cm)."""
        tair = (tmin + tmax) / 2.0
        _fs = self.fs if fs is None else fs
        damp = torch.exp(-_fs * snow_depth_m)                      # (n_nodes,)
        n_depth = profil.shape[1]
        # ca = capacité APPARENTE : celle du sol plus celle de la glace, cette dernière
        # portant la chaleur latente de changement de phase. On module la part du sol,
        # la part glace restant une constante physique.
        ca = (self.cs if cs is None else cs) + self.cice
        _kt = self.kt if kt is None else kt

        # surface (index 0) : eq2 à profondeur 0 (rankinen.cpp:214-217)
        surf = torch.where(snow_depth_m != 0.0, tair * damp, tair)
        # nœuds de profondeur 1..n_depth-1 : VECTORISÉS sur l'axe profondeur (pas
        # de couplage entre nœuds, cf docstring) — remplace la boucle Python+stack.
        zs = torch.arange(1, n_depth, dtype=tair.dtype, device=tair.device) * self.dz  # (n_depth-1,)
        # rate devient (n_nodes, n_depth-1) des que kt ou ca sont des champs : la
        # diffusion thermique est alors PROPRE A CHAQUE TRONCON, ce qui est tout
        # l'objet de la correction.
        _k = _kt[:, None] if torch.is_tensor(_kt) and _kt.ndim else _kt
        _c = ca[:, None] if torch.is_tensor(ca) and ca.ndim else ca
        rate = self.dt * _k / (_c * (2.0 * zs[None, :]) ** 2)
        t_prev = profil[:, 1:]                                    # (n_nodes, n_depth-1)
        fT = t_prev + rate * (tair[:, None] - t_prev)             # eq1 relaxation
        deep = fT * damp[:, None]                                 # eq2 amortissement neige
        profil_new = torch.cat([surf[:, None], deep], dim=1)     # (n_nodes, n_depth)

        # profondeur de gel : dernier indice où T ≤ seuil, interpolé (rankinen.cpp:253-269)
        frozen = profil_new <= self.seuil_gel                     # (n_nodes, n_depth)
        depths = torch.arange(n_depth, dtype=tair.dtype, device=tair.device)
        idx_frozen = torch.where(frozen, depths[None, :], torch.full_like(profil_new, -1.0))
        last_frozen = idx_frozen.max(dim=1).values                # (n_nodes,) -1 si aucun
        prof_gel = torch.zeros_like(tair)
        has = last_frozen >= 0
        lf = last_frozen.clamp(min=0).long()
        fB = lf.to(tair.dtype) * self.dz
        is_last = lf == (n_depth - 1)
        # interpolation linéaire entre lf et lf+1 (sauf si dernier nœud)
        lf1 = (lf + 1).clamp(max=n_depth - 1)
        Tb = torch.gather(profil_new, 1, lf[:, None]).squeeze(1)
        Tb1 = torch.gather(profil_new, 1, lf1[:, None]).squeeze(1)
        interp = (self.seuil_gel - Tb) * self.dz / (Tb1 - Tb + 1e-12) + fB
        prof_gel = torch.where(has, torch.where(is_last, fB, interp), torch.zeros_like(tair))
        return profil_new, prof_gel * 100.0                       # cm
