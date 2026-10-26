"""Le domaine : tout ce que le modèle sait du monde, et rien d'autre.

Un `Domain` porte le réseau, les attributs par tronçon, le forçage, les observations et l'état
initial, sous une forme que la physique, le routage et l'entraînement consomment directement.
Il est produit par un adaptateur (`meandre.adapters.*`), jamais par le modèle ni par le pilote,
et il ne connaît ni PHYSITEL, ni le Québec, ni une machine. Son manifeste dit d'où vient chaque
pièce.

`validate` refuse un domaine incohérent avant qu'il atteigne le modèle : réseau qui n'est pas
une forêt, attributs hors vocabulaire ou non finis, forçage dont les dimensions ne concordent
pas avec le réseau ou l'axe de temps, observation qui vise un tronçon inexistant.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from meandre.domain.manifest import Manifest
from meandre.domain.vocabulary import check_columns
from meandre.routing.graph import RiverGraph
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures
from meandre.utils.state import HydroState


@dataclass
class Observation:
    """Une observation d'un observable, sur l'axe de temps du domaine.

    `values` : (n_times, n_sites) ou (n_times, n_nodes) ou (n_times,) selon la portée ;
    NaN = absent. `sites` : identifiants des colonnes quand la portée est par site ;
    `nodes` : tronçon de chaque colonne (par site) ; `scope` : "site", "node" ou "domain".
    `weight` : poids optionnel de même forme que `values` (par exemple hors neige).
    """
    name: str
    values: Tensor
    scope: str
    sites: list[str] = field(default_factory=list)
    nodes: Tensor | None = None
    weight: Tensor | None = None
    origin: str = ""


@dataclass
class Domain:
    name: str
    node_ids: list[int]
    node_coords: Tensor                 # (n_nodes, 2) lon, lat
    graph: RiverGraph
    attributes: TerritorialFeatures
    attribute_columns: list[str]        # toutes les colonnes de la table, champ et physique
    forcing: Tensor                     # (n_times, n_nodes, n_variables)
    forcing_variables: list[str]
    times: pd.DatetimeIndex
    initial_state: HydroState | None
    withdrawals: WithdrawalData | None
    observations: dict[str, Observation]
    manifest: Manifest
    # Pièces livrées par le paysage de données au-delà de la table d'attributs ; optionnelles.
    # Elles entreront dans `attributes` et dans le catalogue des priors à l'étape 3.
    land_cover: dict[str, Tensor] | None = None          # fractions brutes par classe, géométrie des milieux humides
    phenology: dict | None = None                        # calendrier et racines par classe de couvert
    lake_geometry: dict[str, Tensor] | None = None       # area_km2, k, beta, valid (lacs routés)
    texture_conductivity: dict | None = None             # K_sat_1..3 de la texture (m/j) et source par tronçon
    drainage_class: np.ndarray | None = None             # classe de drainage, 1 à 7, NaN si absente
    groundwater_conductivity: Tensor | None = None       # k_gw par tronçon, des récessions des jauges
    hgm_kernel: np.ndarray | None = None                 # noyau de versant (n_nodes, L), lignes sommant à 1
    texture_fractions: dict | None = None                # par couche : (fraction de sable, fraction d'argile), NaN si absente
    deposit_thickness_m: np.ndarray | None = None        # épaisseur des dépôts meubles, NaN si absente

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def day_of_year(self) -> Tensor:
        return torch.tensor(self.times.dayofyear.values, dtype=torch.float32, device=self.forcing.device)

    def validate(self) -> list[str]:
        """Incohérences, en phrases ; vide si le domaine est sain."""
        issues = []
        n = self.n_nodes
        if self.node_coords.shape != (n, 2):
            issues.append(f"coordonnées de forme {tuple(self.node_coords.shape)} pour {n} tronçons")
        elif not torch.isfinite(self.node_coords).all():
            issues.append("coordonnées non finies")
        ei = self.graph.edge_index
        if ei.numel():
            src = ei[0].cpu().numpy()
            if src.max() >= n or ei[1].max().item() >= n or src.min() < 0:
                issues.append("un arc du réseau vise un tronçon inexistant")
            if len(np.unique(src)) != len(src):
                issues.append("le réseau n'est pas une forêt : un tronçon a plusieurs avals")
        issues += check_columns(self.attribute_columns)
        if not torch.isfinite(self.attributes.data).all():
            issues.append("attributs du champ non finis")
        for k, v in self.attributes.physical.items():
            if not torch.isfinite(v).all():
                issues.append(f"attribut physique non fini : {k}")
        T = len(self.times)
        if self.forcing.shape[:2] != (T, n):
            issues.append(f"forçage de forme {tuple(self.forcing.shape)} pour {T} pas et {n} tronçons")
        if self.forcing.shape[-1] != len(self.forcing_variables):
            issues.append(f"{self.forcing.shape[-1]} variables de forçage pour {len(self.forcing_variables)} noms")
        for name, o in self.observations.items():
            if o.values.shape[0] != T:
                issues.append(f"observation {name} : {o.values.shape[0]} pas de temps pour {T}")
            if o.scope == "node" and o.values.ndim == 2 and o.values.shape[1] != n:
                issues.append(f"observation {name} : {o.values.shape[1]} colonnes pour {n} tronçons")
            if o.scope == "site":
                if o.nodes is None or len(o.nodes) != o.values.shape[1]:
                    issues.append(f"observation {name} : tronçons des sites absents ou de mauvaise taille")
                elif o.nodes.numel() and (o.nodes.max().item() >= n or o.nodes.min().item() < 0):
                    issues.append(f"observation {name} : un site vise un tronçon inexistant")
        return issues

    def summary(self) -> str:
        obs = ", ".join(f"{k} ({v.values.shape[1] if v.values.ndim == 2 else 1})" for k, v in self.observations.items()) or "aucune"
        return (f"domaine {self.name} : {self.n_nodes} tronçons, {len(self.times)} jours du {self.times[0].date()} au {self.times[-1].date()}, "
                f"{len(self.forcing_variables)} variables de forçage, {len(self.attributes.columns)} attributs au champ ; observations : {obs}")
