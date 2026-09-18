"""Nappe libre : profondeur de la surface libre comme variable d'état.

Pourquoi cette pièce existe. L'aquifère restituant de la colonne est un réservoir linéaire
dont le stock se mesure en millimètres, sans surface libre. Trois conséquences mesurées le
2026-09-18 sur 118 puits du réseau de suivi. Son amplitude et son retard sont réglés par le
même coefficient, si bien qu'aucun réglage ne donne ensemble les 0,93 m de battement mesurés
et un maximum un mois après la fonte. Son stock en millimètres ne se compare aux mètres
mesurés dans un puits qu'à un facteur d'échelle libre, qui absorbe toute erreur de structure.
Et son flux ne peut pas s'inverser, donc une nappe peu profonde ne peut pas alimenter
l'évapotranspiration d'été, mécanisme qui creuse l'étiage réel.

La formulation ici tient ces trois points avec des paramètres qui ont chacun un sens
mesurable.

    S = Sy * (z_ref - z)        z profondeur de la surface libre sous le sol (m)
    dz/dt = -(R - Q_b - E) / Sy
    Q_b = K_b * (h / h_ref) ** n      h = max(z_riv - z, 0), charge au-dessus du lit
    E   = E_max * max(0, 1 - z / z_ext)

La porosité de drainage Sy convertit les millimètres en mètres : c'est elle qui rend la
simulation et le puits commensurables, et elle se prédit par la géologie du socle et les
dépôts de surface au lieu d'être un facteur d'échelle libre par puits.

L'exposant n vaut 2 par défaut, ce qui n'est pas un réglage : pour une nappe libre drainant
vers un cours d'eau, l'approximation de Dupuit-Boussinesq donne un débit proportionnel au
carré de la charge, et Wittenberg (1999) retrouve cet ordre sur des récessions observées.
n = 1 rend le réservoir linéaire, ce qui sert de cas de référence analytique.

L'extraction E suit la rampe linéaire du paquet d'évapotranspiration de MODFLOW : maximale
quand la surface libre affleure, nulle au-delà de la profondeur d'extinction z_ext.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class NappeLibre(nn.Module):
    """Nappe libre différentiable, profondeur de la surface libre en variable d'état.

    Toutes les grandeurs sont en mètres et en jours. Les flux d'entrée et de sortie sont
    des hauteurs d'eau par jour rapportées à la surface du sol, non des volumes.
    """

    def __init__(self, n_substep: int = 8, exposant: float = 2.0) -> None:
        super().__init__()
        # Sous-pas internes. Le pas journalier suffit tant que le temps de réponse de la
        # nappe dépasse quelques jours, mais la loi non linéaire se raidit quand la charge
        # est forte : la convergence se VÉRIFIE au banc au lieu d'être supposée.
        self.n_substep = int(n_substep)
        self.exposant = float(exposant)

    def forward(self, z: Tensor, recharge: Tensor, sy: Tensor, k_b: Tensor, z_riv: Tensor,
                h_ref: Tensor, e_max: Tensor | None = None, z_ext: Tensor | None = None,
                prelevement: Tensor | None = None, dt: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
        """Un pas de temps.

        Args:
            z: profondeur de la surface libre sous le sol (m), positive vers le bas.
            recharge: apport depuis la colonne au-dessus (m/j), positif vers le bas.
            sy: porosité de drainage (-), de 0,01 dans le roc fracturé à 0,30 dans les sables.
            k_b: débit de base quand la charge vaut h_ref (m/j).
            z_riv: profondeur du lit du cours d'eau sous le sol (m).
            h_ref: charge de référence (m), échelle de la loi stock-débit.
            e_max: évapotranspiration maximale depuis la zone saturée (m/j), None pour aucune.
            z_ext: profondeur d'extinction de cette évapotranspiration (m).
            prelevement: pompage net (m/j), positif quand on retire de l'eau.
            dt: durée du pas (j).

        Returns:
            z_new: profondeur mise à jour (m).
            q_base: débit de base moyen sur le pas (m/j).
            e_nappe: évapotranspiration prélevée sur la nappe, moyenne sur le pas (m/j).
        """
        sous_dt = dt / self.n_substep
        q_cumul = torch.zeros_like(z)
        e_cumul = torch.zeros_like(z)
        for _ in range(self.n_substep):
            h = torch.clamp(z_riv - z, min=0.0)
            q = k_b * (h / h_ref).pow(self.exposant)
            if e_max is not None:
                e = e_max * torch.clamp(1.0 - z / z_ext, min=0.0, max=1.0)
            else:
                e = torch.zeros_like(z)
            sortie = q + e
            if prelevement is not None:
                sortie = sortie + prelevement
            # La nappe ne peut pas rendre plus que ce qu'elle contient au-dessus du lit :
            # sans ce plafond une charge résiduelle produirait un débit à stock nul.
            z = z - (recharge - sortie) * sous_dt / sy
            z = torch.clamp(z, min=0.0)
            q_cumul = q_cumul + q * sous_dt
            e_cumul = e_cumul + e * sous_dt
        return z, q_cumul / dt, e_cumul / dt

    @staticmethod
    def reponse_analytique(k: float, periode: float = 365.25) -> tuple[float, float]:
        """Amplitude et retard d'un réservoir LINÉAIRE sous recharge sinusoïdale.

        Pour dS/dt = R0 (1 + a cos(wt)) - k S, la composante oscillante du stock a pour
        amplitude a R0 / sqrt(k^2 + w^2) et pour retard atan(w / k) / w. Les deux sont
        gouvernés par le même k : gagner de l'amplitude coûte du retard, exactement dans
        cette proportion. C'est la borne que la loi non linéaire doit franchir.

        Returns:
            gain: amplitude du stock par unité d'amplitude de recharge (jours).
            retard: décalage du maximum du stock sur celui de la recharge (jours).
        """
        import math

        w = 2.0 * math.pi / periode
        return 1.0 / math.sqrt(k * k + w * w), math.atan2(w, k) / w
