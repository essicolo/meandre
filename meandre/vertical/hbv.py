"""HBV-EC fidèle, différentiable (cœur de RavenTorch / δHBV).

Reproduit la structure HBV-light (Seibert 2005) que Raven émule en HBV-EC :
  Sol (beta)      : recharge = P_eff·(SM/FC)^BETA ; ET = PET·min(SM/(FC·LP),1)
  Réponse 2 rés.  : SUZ (supérieur) avec quickflow à seuil Q0=K0·max(SUZ−UZL,0),
                    interflow Q1=K1·SUZ, percolation PERC ; SLZ (inférieur)
                    baseflow Q2=K2·SLZ
  Routage MAXBAS  : convolution par un hydrogramme triangulaire (largeur MAXBAS)

Tout est lisse et différentiable. Le lissage vient de la STRUCTURE de réponse
(les réservoirs) et du MAXBAS, PAS d'une atténuation de canal — c'est pourquoi
HBV préserve les pics là où notre Muskingum les rabotait. Les paramètres sont
prédits par nœud par le NeRF.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

# Bornes physiques HBV-light (Seibert 2005, tableau de calage).
BOUNDS = {
    "BETA": (1.0, 6.0),       # forme de la recharge
    "FC": (50.0, 700.0),      # capacité au champ (mm)
    "LP": (0.3, 1.0),         # seuil ET (fraction de FC)
    "K0": (0.05, 0.99),       # récession quickflow (1/j)
    "K1": (0.01, 0.5),        # récession interflow (1/j)
    "K2": (0.001, 0.15),      # récession baseflow (1/j)
    "UZL": (0.0, 100.0),      # seuil quickflow (mm)
    "PERC": (0.0, 6.0),       # percolation max (mm/j)
    "MAXBAS": (1.0, 7.0),     # largeur de l'hydrogramme triangulaire (j)
}
HBV_PARAMS = list(BOUNDS.keys())
N_HBV = len(HBV_PARAMS)
MAXBAS_M = 8   # longueur du buffer MAXBAS (jours), ≥ max(MAXBAS)


def constrain(raw: Tensor, name: str) -> Tensor:
    lo, hi = BOUNDS[name]
    return lo + (hi - lo) * torch.sigmoid(raw)


def maxbas_weights(maxbas: Tensor, M: int = MAXBAS_M) -> Tensor:
    """Poids d'un hydrogramme unitaire TRIANGULAIRE de largeur `maxbas` (jours).

    Triangle symétrique sur [0, maxbas], sommet au milieu, aire 1. Discrétisé
    sur M jours (poids des jours au-delà de maxbas = 0). Différentiable en maxbas.
    maxbas: (n,) ; retourne (n, M) normalisé à somme 1.
    """
    t = torch.arange(M, device=maxbas.device, dtype=maxbas.dtype).unsqueeze(0)  # (1,M)
    mb = maxbas.unsqueeze(-1).clamp(min=1.0)                                     # (n,1)
    half = mb / 2.0
    # hauteur triangulaire au centre du jour t (montée puis descente).
    up = t + 0.5                                       # position centre du jour
    h = torch.where(up <= half, up, mb - up)
    h = torch.clamp(h, min=0.0)
    w = h / (h.sum(dim=-1, keepdim=True) + 1e-8)
    return w


class HBVModule(nn.Module):
    """HBV-EC : sol beta + réponse 2 réservoirs + MAXBAS. Différentiable.

    forward(P_eff, PET, state, params) -> (Q_routed_mm, new_state)
    où state = (SM, SUZ, SLZ, maxbas_buffer(n,M)) et params = dict de tenseurs
    HBV bornés par nœud (BETA, FC, LP, K0, K1, K2, UZL, PERC, MAXBAS).
    """

    def forward(
        self,
        P_eff: Tensor, PET: Tensor,
        SM: Tensor, SUZ: Tensor, SLZ: Tensor, mb_buf: Tensor,
        BETA: Tensor, FC: Tensor, LP: Tensor, K0: Tensor, K1: Tensor,
        K2: Tensor, UZL: Tensor, PERC: Tensor, MAXBAS: Tensor,
    ):
        eps = 1e-6
        # ── Sol (routine beta) ────────────────────────────────────────────
        Se = torch.clamp(SM / (FC + eps), 0.0, 1.0)
        recharge = P_eff * Se.pow(BETA)                 # vers la réponse
        SM1 = SM + (P_eff - recharge)
        # ET réelle (limitée par l'humidité)
        et_frac = torch.clamp(SM1 / (LP * FC + eps), 0.0, 1.0)
        ET = PET * et_frac
        ET = torch.minimum(ET, torch.clamp(SM1, min=0.0))
        SM_new = torch.clamp(SM1 - ET, min=0.0)

        # ── Réponse : réservoir supérieur SUZ ─────────────────────────────
        SUZ1 = SUZ + recharge
        Q0 = K0 * torch.clamp(SUZ1 - UZL, min=0.0)      # quickflow à seuil
        Q1 = K1 * SUZ1                                  # interflow
        perc = torch.minimum(PERC, torch.clamp(SUZ1, min=0.0))
        # conservation : borner les sorties à l'eau disponible (idiome sf)
        out_uz = Q0 + Q1 + perc
        scale = torch.clamp(SUZ1 / (out_uz + eps), max=1.0)
        Q0 = Q0 * scale; Q1 = Q1 * scale; perc = perc * scale
        SUZ_new = torch.clamp(SUZ1 - Q0 - Q1 - perc, min=0.0)

        # ── Réservoir inférieur SLZ (baseflow) ────────────────────────────
        SLZ1 = SLZ + perc
        Q2 = K2 * SLZ1
        Q2 = torch.minimum(Q2, torch.clamp(SLZ1, min=0.0))
        SLZ_new = torch.clamp(SLZ1 - Q2, min=0.0)

        Q_gen = Q0 + Q1 + Q2                            # mm/jour avant MAXBAS

        # ── Routage MAXBAS (hydrogramme triangulaire) ─────────────────────
        w = maxbas_weights(MAXBAS, mb_buf.shape[-1])    # (n, M)
        mb_new = mb_buf.clone()
        mb_new = mb_new + Q_gen.unsqueeze(-1) * w        # étale Q_gen sur M jours
        Q_routed = mb_new[:, 0]
        # décale le buffer (jour suivant)
        mb_shifted = torch.zeros_like(mb_new)
        mb_shifted[:, :-1] = mb_new[:, 1:]
        return Q_routed, SM_new, SUZ_new, SLZ_new, mb_shifted, ET
