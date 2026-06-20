"""Clone FIDÈLE de la chaîne évapotranspiration d'Hydrotel, porté ligne-à-ligne
du C++ vers PyTorch, vectorisé, différentiable. Sous-projet PROPRE.

Deux étages :
  1. ETP potentielle Hydro-Québec (hydro_quebec.cpp:72), formule température
     (Tmin/Tmax journalières). ETP par classe = ETP × pourcentage × coef_mult.
  2. ETR réelle par couche (bv3c2.cpp:2211 CalculeEtr) : évaporation sol nu
     (esnu, couche 1, atténuée par l'indice foliaire via Beer-Lambert) +
     transpiration (tp, répartie entre couches par racine × humidité). Boucle
     sur les classes d'occupation perméables (_index_autres = hors eau/imperm).

Constantes : BETA=1.1 (constantes.hpp:48). Params sol (thetacc/thetapf/alpha)
de proprietehydrolique.sol ; des/coef_assech/krec de bv3c.csv.
"""
from __future__ import annotations
import torch
from torch import Tensor

BETA = 1.1   # constantes.hpp:48


def hydro_quebec_etp(tmin_j: Tensor, tmax_j: Tensor, poids: float = 1.0) -> Tensor:
    """ETP potentielle Hydro-Québec [mm/jour] (hydro_quebec.cpp:72). Tmin/Tmax
    JOURNALIÈRES (°C). poids = Repartition (1.0 au pas journalier). C'est l'ETP
    de base par unité de surface ; l'ETP par classe = etp × pct × coef_mult."""
    return poids * 0.029718 * (tmax_j - tmin_j) * torch.exp(
        0.019 * ((9.0 / 5.0) * (tmax_j + tmin_j) + 64.0))


def _stress(ftheta: Tensor, alpha: Tensor) -> Tensor:
    """kas/kat (bv3c2.cpp:2266) : (1-e^{-α·f}) / (1 - 2e^{-α} + e^{-α·f})."""
    ea = torch.exp(-alpha)
    eaf = torch.exp(-alpha * ftheta)
    return (1.0 - eaf) / (1.0 - 2.0 * ea + eaf)


def calcule_etr(theta1, theta2, theta3, etp_classes, root_depth, leaf_index,
                thetacc, thetapf, alpha, z11, z22, z33, des, coef_assech):
    """CalculeEtr (bv3c2.cpp:2211), un pas de temps, sommé sur les classes
    perméables. Tout vectorisé sur les nœuds ; les listes par classe sont des
    listes Python (peu de classes).

    etp_classes : liste de tenseurs ETP par classe [m] (déjà × pct × coef).
    root_depth, leaf_index : listes (même longueur) [m] et [-] par classe/jour.
    thetacc/thetapf/alpha : props sol couche 1 (par nœud). z11/z22/z33 : épaisseurs [m].
    des : coef extinction Beer (indice foliaire). coef_assech : coef d'assèchement.
    Retourne (etr1, etr2, etr3) [m]."""
    z1 = z11
    z2 = z11 + z22
    z3 = z11 + z22 + z33
    etr1 = torch.zeros_like(theta1)
    etr2 = torch.zeros_like(theta1)
    etr3 = torch.zeros_like(theta1)
    denom_cc = (thetacc - thetapf)

    for etp, z, lai in zip(etp_classes, root_depth, leaf_index):
        pos = etp > 0.0
        evapo = etp * torch.exp(-des * lai)
        # évaporation sol nu depuis la couche 1
        ftheta1 = torch.clamp((theta1 - thetapf) / denom_cc, 0.0, 1.0)
        kas = _stress(ftheta1, alpha)
        esnu = (coef_assech * kas) * evapo
        etr1 = etr1 + torch.where(pos, esnu, torch.zeros_like(esnu))

        # répartition de la profondeur racinaire dans les couches
        dz1 = torch.where(z <= z1, z, torch.full_like(theta1, 0.0)) if not torch.is_tensor(z) else None
        # z peut être scalaire/tenseur ; on calcule dz1/dz2/dz3 par cas (vectorisé)
        zt = z if torch.is_tensor(z) else torch.full_like(theta1, float(z))
        z1t = torch.full_like(theta1, float(z1)) if not torch.is_tensor(z1) else z1
        z2t = torch.full_like(theta1, float(z2)) if not torch.is_tensor(z2) else z2
        z3t = torch.full_like(theta1, float(z3)) if not torch.is_tensor(z3) else z3
        zc = torch.minimum(zt, z3t)   # z plafonné à z3 (cas z>z3)
        dz1 = torch.clamp(torch.minimum(zc, z1t), min=0.0)
        dz2 = torch.clamp(torch.minimum(zc, z2t) - z1t, min=0.0)
        dz3 = torch.clamp(zc - z2t, min=0.0)

        theta_rz = (theta1 * dz1 + theta2 * dz2 + theta3 * dz3) / torch.clamp(zc, min=1e-12)
        ftheta_rz = torch.clamp((theta_rz - thetapf) / denom_cc, 0.0, 1.0)
        kat = _stress(ftheta_rz, alpha)
        ratio = torch.where(evapo > 0.0, esnu / torch.clamp(evapo, min=1e-12), torch.zeros_like(evapo))
        tp = (coef_assech * kat) * ((etp - evapo) * (BETA + (1.0 - BETA) * ratio))

        denom = theta_rz * zc
        ok = pos & (zc > 0.0) & (denom != 0.0)
        etr1 = etr1 + torch.where(ok, tp * (theta1 * dz1) / torch.clamp(denom, min=1e-12), torch.zeros_like(tp))
        etr2 = etr2 + torch.where(ok, tp * (theta2 * dz2) / torch.clamp(denom, min=1e-12), torch.zeros_like(tp))
        etr3 = etr3 + torch.where(ok, tp * (theta3 * dz3) / torch.clamp(denom, min=1e-12), torch.zeros_like(tp))

    return etr1, etr2, etr3


def interp_cycle(jours_bp, valeurs_bp, jour):
    """Interpolation linéaire d'un cycle annuel (def file) au jour julien."""
    import numpy as np
    return float(np.interp(jour, jours_bp, valeurs_bp))
