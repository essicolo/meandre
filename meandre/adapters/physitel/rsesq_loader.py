"""Niveaux de nappe mesurés du réseau de suivi, appariés aux nœuds d'un domaine.

Pourquoi ce jeu. C'est la SEULE observation directe et non circulaire de l'eau souterraine
disponible ici. Les cartes de recharge régionales sont écartées par le tableau des lignes
rouges du registre, étant calées sur le débit de base, la variable même que le modèle ajuste.
La gravimétrie contraint le stock total à quelque trois cents kilomètres de résolution, donc
la moyenne régionale et non sa structure. Les puits, eux, mesurent la position de la surface
libre en un point, au pas journalier, sur deux décennies.

Recevabilité. Deux filtres, pour des raisons de physique et non de qualité de donnée. Un
puits CAPTIF mesure une charge transmise à travers une couche imperméable, pas le remplissage
d'un réservoir : sa dynamique n'a aucune raison de ressembler à celle d'une nappe libre, et
le mélanger aux autres dilue le signal dans les deux sens, ce qui a été mesuré le 2026-09-18
(corrélation saisonnière de 0,45 tous puits confondus contre 0,12 sur les seuls puits libres).
Un puits INFLUENCÉ par un pompage voisin mesure ce pompage.

Usage. Contrainte de FORME, en anomalies réduites, comme pour l'évapotranspiration satellitaire
et la gravimétrie. Jamais de niveau absolu : la profondeur simulée dépend d'une porosité de
drainage et d'une altitude de référence que rien n'identifie encore.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WellTargets:
    """Niveaux mensuels appariés, prêts pour le terme de perte.

    puits : (W,) identifiants.
    node_idx : (W,) indice du nœud portant chaque puits, dans l'ordre du domaine.
    niveau : (M, W) profondeur mesurée sous le repère du tubage, en mètres, moyenne mensuelle.
    masque : (M, W) booléen, vrai là où la mesure existe.
    mois : (M,) premier jour de chaque mois couvert, en datetime64.
    distance_km : (W,) distance du puits au centre du tronçon apparié.
    """
    wells: np.ndarray
    node_idx: np.ndarray
    level: np.ndarray
    mask: np.ndarray
    months: np.ndarray
    distance_km: np.ndarray

    @property
    def n_wells(self) -> int:
        return len(self.wells)

def _path_default() -> str:
    root = os.environ.get("MEANDRE_DERIVES")
    if not root:
        from meandre.utils import paths as _paths

        root = _paths.DERIVED_ROOT
    return f"{root}/auxiliaires"


def read_rsesq(region: str, times, file_path: str | None = None, unconfined_only: bool = True,
               uninfluenced: bool = True, distance_max_km: float = 10.0,
               min_months: int = 24) -> WellTargets:
    """Cibles de nappe pour un domaine, agrégées au mois sur la période de `times`.

    times : index de dates journalières de la simulation. La grille mensuelle de sortie
    couvre exactement les mois qu'il contient, si bien que la perte peut indexer sans
    réalignement.
    """
    base = file_path or _path_default()
    wells = pd.read_parquet(f"{base}/rsesq-puits.parquet")
    wells = wells[wells.region.str.lower() == region.lower()]
    if unconfined_only:
        wells = wells[wells.confinement == "Libre"]
    if uninfluenced:
        wells = wells[wells.influence != "Oui"]
    wells = wells[wells.distance_km <= distance_max_km]

    levels = pd.read_parquet(f"{base}/rsesq-niveaux-journaliers.parquet",
                              columns=["puits", "date", "niveau_m"])
    levels = levels[levels.puits.isin(wells.puits)]
    levels = levels.assign(months=pd.to_datetime(levels.date).values.astype("datetime64[M]"))
    table = levels.groupby(["months", "puits"]).niveau_m.mean().unstack()

    months = pd.to_datetime(times).values.astype("datetime64[M]")
    grid = np.unique(months)
    table = table.reindex(index=pd.DatetimeIndex(grid))

    # Un puits trop court ne porte aucune information de forme : on l'écarte ici plutôt que
    # de laisser le terme de perte le découvrir à chaque pas.
    assez = table.notna().sum(axis=0) >= min_months
    table = table.loc[:, assez[assez].index]
    order = [p for p in wells.puits if p in table.columns]
    table = table[order]
    meta = wells.set_index("puits").loc[order]

    return WellTargets(wells=np.asarray(order, dtype=object),
                       node_idx=meta.node_idx.to_numpy(dtype=np.int64),
                       level=table.to_numpy(dtype=np.float32),
                       mask=table.notna().to_numpy(),
                       months=grid,
                       distance_km=meta.distance_km.to_numpy(dtype=np.float32))


def index_monthly(times) -> np.ndarray:
    """Indice du mois de chaque pas journalier dans la grille mensuelle de `read_rsesq`."""
    months = pd.to_datetime(times).values.astype("datetime64[M]")
    grid = np.unique(months)
    return np.searchsorted(grid, months)
