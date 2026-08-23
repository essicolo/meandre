"""Lumped aquifer module — groundwater storage and baseflow generation.

Intercepts recharge from soil layer 3 and delays it through a linear
reservoir, producing a smoothed baseflow signal.

    dS_gw/dt = recharge - k_gw * S_gw

Analytical solution (exact for constant recharge over one day):

    S_gw(t+1) = S_gw(t) * exp(-k_gw) + (recharge / k_gw) * (1 - exp(-k_gw))
    Q_baseflow = k_gw * S_gw(t+1)

State: S_gw (mm) — groundwater storage per node.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AquiferModule(nn.Module):
    """Differentiable lumped linear-reservoir aquifer.

    Receives recharge (mm/day) from soil layer 3 and returns delayed
    baseflow (mm/day).  The recession constant k_gw (1/day) is supplied
    per-node from the SpatialFieldNetwork.
    """

    def __init__(self, k_gw_min: float = 1e-6) -> None:
        super().__init__()
        self.k_gw_min = k_gw_min

    def forward(
        self,
        recharge: Tensor,
        S_gw: Tensor,
        k_gw: Tensor,
        gw_withdrawal: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """One-day aquifer update.

        Args:
            recharge:      (n_nodes,) recharge from soil L3 (mm/day), >= 0.
            S_gw:          (n_nodes,) current groundwater storage (mm).
            k_gw:          (n_nodes,) recession coefficient (1/day).
            gw_withdrawal: (n_nodes,) net withdrawal from the aquifer (mm/day).
                Positive = water added (artificial recharge), negative =
                water removed (pumping).  Included as a constant forcing
                alongside recharge in the linear-reservoir ODE.

        Returns:
            Q_baseflow: (n_nodes,) baseflow discharge (mm/day).
            S_gw_new:   (n_nodes,) updated groundwater storage (mm).
        """
        # Clamp k_gw away from zero for numerical safety.
        # For very small k_gw, use Taylor expansion to avoid (1-exp(-k))/k loss.
        k = torch.clamp(k_gw, min=self.k_gw_min)

        # Treat pumping / artificial recharge as a constant flux over dt.
        # Sign: withdrawal tensor is positive=add, negative=remove.
        net_input = recharge
        if gw_withdrawal is not None:
            net_input = net_input + gw_withdrawal

        # Analytical linear reservoir solution (dt = 1 day implicit)
        decay = torch.exp(-k)
        # (1 - exp(-k)) / k  — Taylor-safe: for k < 1e-4, ≈ 1 - k/2
        one_minus_decay_over_k = torch.where(
            k > 1e-4,
            (1.0 - decay) / k,
            1.0 - k * 0.5 + k * k / 6.0,
        )

        S_gw_new = S_gw * decay + net_input * one_minus_decay_over_k
        # Clamp to 0: if pumping exceeds storage + recharge, the aquifer
        # simply empties (physically the well would go dry).
        S_gw_new = torch.clamp(S_gw_new, min=0.0)

        Q_baseflow = k * S_gw_new

        return Q_baseflow, S_gw_new


class PowerLawAquifer(nn.Module):
    """Reservoir souterrain NON LINEAIRE : Q = k * S^b, b > 1 (Wittenberg 1999).

    Pourquoi il existe (R29, 2026-08-22). Les recessions hivernales pures des jauges
    d'OUTV portent DEUX constantes de temps -- 37 jours en mediane, 111 jours pour la
    composante lente (centile 10) -- et un reservoir lineaire n'en a qu'UNE : quel que
    soit son k_gw, il vide trop vite l'etiage ou trop lentement la crue. La loi
    puissance donne les deux avec un seul jeu de parametres : a stock plein la vidange
    est rapide (S^b >> S), a stock bas elle s'etire d'elle-meme. C'est BASE_POWER_LAW
    chez Raven ; Kirchner (2009) montre que la plupart des bassins reels sont dans ce
    regime plutot que dans le lineaire.

    Discretisation : pas de solution analytique pour b != 1, on integre par sous-pas
    d'Euler implicites via une iteration de point fixe courte (5 iterations suffisent :
    la fonction est contractante pour dt*k*b*S^(b-1) < 1, et on sous-pas si besoin).
    Differentiable de bout en bout.

    UNITES ET ECHELLE. Q = k * S^b avec S en mm : pour rester dimensionnellement
    lisible, k est exprime via un DEBIT DE REFERENCE a un stock de reference :
        Q = q_ref * (S / s_ref)^b
    q_ref (mm/j) et b sont les parametres ; s_ref = 100 mm fixe. Ainsi q_ref garde le
    sens physique d'un debit de base a stock moyen, et b module la courbure seule --
    les deux ne se marchent pas dessus pendant l'apprentissage.
    """

    S_REF = 100.0     # mm, stock de reference (fixe : la courbure est portee par b)

    def __init__(self, n_substeps: int = 4) -> None:
        super().__init__()
        self.n_substeps = int(n_substeps)

    def forward(
        self,
        recharge: Tensor,
        S_gw: Tensor,
        q_ref: Tensor,
        b: Tensor,
        gw_withdrawal: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Mise a jour sur un jour. recharge/S_gw en mm ; q_ref en mm/j ; b sans unite.

        Retourne (Q_baseflow mm/j, S_gw_new mm). Conservation exacte du bilan
        discretise : S_new = S + apports - Q_total, Q_total etant la somme des
        sous-pas ; un test le verrouille.
        """
        dt = 1.0 / self.n_substeps
        net = recharge if gw_withdrawal is None else recharge + gw_withdrawal
        S = S_gw
        Q_total = torch.zeros_like(S_gw)
        for _ in range(self.n_substeps):
            S_in = torch.clamp(S + net * dt, min=0.0)
            # Euler implicite par point fixe : S1 = S_in - dt*Q(S1). Contractant tant
            # que dt * dQ/dS < 1 ; 5 iterations donnent < 0.1 % d'erreur aux regimes
            # concernes (q_ref <= 5 mm/j, b <= 3).
            S1 = S_in
            for _ in range(5):
                Q = q_ref * torch.clamp(S1 / self.S_REF, min=0.0) ** b
                S1 = torch.clamp(S_in - dt * Q, min=0.0)
            Q_final = (S_in - S1) / dt          # ce qui est REELLEMENT sorti
            Q_total = Q_total + Q_final * dt
            S = S1
        return Q_total, S
