"""Hydrological state management — save, restore, and convert state dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class HydroState:
    """Full per-node hydrological state at a single timestep.

    All tensors have shape (n_nodes,).

    Soil moisture (volumetric water content, m3/m3):
        theta1  Layer 1  0-30 cm
        theta2  Layer 2  30-100 cm
        theta3  Layer 3  100-200 cm

    Snow and frost:
        swe     Snow water equivalent (mm)
        t_soil  Soil surface temperature (C)

    Canopy and wetland:
        canopy_storage  Intercepted water (mm)
        wetland_storage Wetland storage (mm)

    Groundwater:
        S_gw    Groundwater storage (mm)

    Stream temperature:
        T_water Stream water temperature (C)
    """

    theta1: Tensor
    theta2: Tensor
    theta3: Tensor
    swe: Tensor
    t_soil: Tensor
    canopy_storage: Tensor
    wetland_storage: Tensor
    S_gw: Tensor
    T_water: Tensor
    # Optional: cold content (mm équivalent eau). Énergie nécessaire pour
    # réchauffer le pack à 0°C avant fonte. Empêche les redoux mid-winter
    # de fondre toute la neige. Défaut zeros pour rétrocompatibilité.
    cold_content: Tensor | None = None
    # Growing degree days cumulés depuis le 1er janvier (°C·j). Reset chaque
    # année. Utilisé par PhenologyModulator (IHI Phase B) pour moduler K_c
    # selon le stade phénologique. Défaut zeros pour rétrocompatibilité.
    gdd_cum: Tensor | None = None
    # Optional: upper-zone fast-reservoir storage (mm) for the HBV-EC threshold
    # quickflow (K0/UZL/K1). Réservoir rapide (vidé en quelques jours), non
    # sérialisé dans to_tensor (reset à zéro sans effet aux allers-retours de
    # persistance). Défaut zeros pour rétrocompatibilité.
    S_uz: Tensor | None = None
    # Optional: états de la cascade de Nash de l'hydrogramme de versant (2
    # réservoirs). Non sérialisés (comme S_uz), reset à zéro sans effet aux
    # allers-retours de persistance (réservoirs rapides).
    uh_s1: Tensor | None = None
    uh_s2: Tensor | None = None
    uh_s3: Tensor | None = None
    uh_s4: Tensor | None = None

    def __post_init__(self) -> None:
        if self.cold_content is None:
            self.cold_content = torch.zeros_like(self.swe)
        if self.gdd_cum is None:
            self.gdd_cum = torch.zeros_like(self.swe)
        if self.S_uz is None:
            self.S_uz = torch.zeros_like(self.swe)

    @property
    def n_nodes(self) -> int:
        return self.theta1.shape[0]

    def to_tensor(self) -> Tensor:
        """Stack all state variables into a single (n_nodes, n_state_vars) tensor."""
        return torch.stack(
            [
                self.theta1,
                self.theta2,
                self.theta3,
                self.swe,
                self.t_soil,
                self.canopy_storage,
                self.wetland_storage,
                self.S_gw,
                self.T_water,
                self.cold_content,
                self.gdd_cum,
            ],
            dim=-1,
        )

    @classmethod
    def from_tensor(cls, x: Tensor) -> "HydroState":
        """Reconstruct HydroState from a (n_nodes, 7/8/9/10) tensor.

        Handles backward compatibility: old 7-column tensors get S_gw=0
        and T_water=10; 8-column tensors get T_water=10; 9-column tensors
        get cold_content=0.
        """
        n = x.shape[0]
        device = x.device

        # Pad missing columns for backward compatibility
        if x.shape[1] == 7:
            x = torch.cat([x, torch.zeros(n, 2, device=device)], dim=1)
            x[:, 8] = 10.0  # T_water default
        if x.shape[1] == 8:
            x = torch.cat([x, torch.full((n, 1), 10.0, device=device)], dim=1)
        if x.shape[1] == 9:
            x = torch.cat([x, torch.zeros(n, 1, device=device)], dim=1)
        if x.shape[1] == 10:
            x = torch.cat([x, torch.zeros(n, 1, device=device)], dim=1)

        return cls(
            theta1=x[:, 0],
            theta2=x[:, 1],
            theta3=x[:, 2],
            swe=x[:, 3],
            t_soil=x[:, 4],
            canopy_storage=x[:, 5],
            wetland_storage=x[:, 6],
            S_gw=x[:, 7],
            T_water=x[:, 8],
            cold_content=x[:, 9],
            gdd_cum=x[:, 10],
        )

    @classmethod
    def zeros(cls, n_nodes: int, device: torch.device | None = None) -> "HydroState":
        """Initialise state to zero (cold start)."""
        z = torch.zeros(n_nodes, device=device)
        return cls(
            theta1=z.clone(),
            theta2=z.clone(),
            theta3=z.clone(),
            swe=z.clone(),
            t_soil=z.clone(),
            canopy_storage=z.clone(),
            wetland_storage=z.clone(),
            S_gw=z.clone(),
            T_water=torch.full((n_nodes,), 10.0, device=device),
            cold_content=z.clone(),
            gdd_cum=z.clone(),
        )

    @classmethod
    def default_warm(
        cls, n_nodes: int, device: torch.device | None = None
    ) -> "HydroState":
        """Physically plausible warm start (avoids spin-up artefacts in tests)."""
        return cls(
            theta1=torch.full((n_nodes,), 0.3, device=device),
            theta2=torch.full((n_nodes,), 0.25, device=device),
            theta3=torch.full((n_nodes,), 0.2, device=device),
            swe=torch.zeros(n_nodes, device=device),
            t_soil=torch.full((n_nodes,), 5.0, device=device),
            canopy_storage=torch.zeros(n_nodes, device=device),
            wetland_storage=torch.zeros(n_nodes, device=device),
            S_gw=torch.full((n_nodes,), 10.0, device=device),
            T_water=torch.full((n_nodes,), 8.0, device=device),
            cold_content=torch.zeros(n_nodes, device=device),
            gdd_cum=torch.zeros(n_nodes, device=device),
        )

    # ---- Persistence ----

    def save(self, path: str) -> None:
        torch.save(self.to_tensor(), path)

    @classmethod
    def load(cls, path: str, device: torch.device | None = None) -> "HydroState":
        x = torch.load(path, map_location=device)
        return cls.from_tensor(x)

    def detach(self) -> "HydroState":
        """Return a new HydroState with all tensors detached from the computation graph.

        Used for truncated BPTT: breaks the gradient chain across chunk boundaries
        while keeping the state values for the next chunk's forward pass.
        """
        return HydroState(
            theta1=self.theta1.detach(),
            theta2=self.theta2.detach(),
            theta3=self.theta3.detach(),
            swe=self.swe.detach(),
            t_soil=self.t_soil.detach(),
            canopy_storage=self.canopy_storage.detach(),
            wetland_storage=self.wetland_storage.detach(),
            S_gw=self.S_gw.detach(),
            T_water=self.T_water.detach(),
            cold_content=self.cold_content.detach(),
            gdd_cum=self.gdd_cum.detach(),
            S_uz=self.S_uz.detach(),
            uh_s1=self.uh_s1.detach() if self.uh_s1 is not None else None,
            uh_s2=self.uh_s2.detach() if self.uh_s2 is not None else None,
            uh_s3=self.uh_s3.detach() if self.uh_s3 is not None else None,
            uh_s4=self.uh_s4.detach() if self.uh_s4 is not None else None,
        )

    # Number of state variables (used by residual corrector)
    N_VARS: int = 11
