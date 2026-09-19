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
class CiblesNappe:
    """Niveaux mensuels appariés, prêts pour le terme de perte.

    puits : (W,) identifiants.
    node_idx : (W,) indice du nœud portant chaque puits, dans l'ordre du domaine.
    niveau : (M, W) profondeur mesurée sous le repère du tubage, en mètres, moyenne mensuelle.
    masque : (M, W) booléen, vrai là où la mesure existe.
    mois : (M,) premier jour de chaque mois couvert, en datetime64.
    distance_km : (W,) distance du puits au centre du tronçon apparié.
    """
    puits: np.ndarray
    node_idx: np.ndarray
    niveau: np.ndarray
    masque: np.ndarray
    mois: np.ndarray
    distance_km: np.ndarray

    @property
    def n_puits(self) -> int:
        return len(self.puits)

    def resume(self) -> str:
        n = self.masque.sum(axis=0)
        return (f"{self.n_puits} puits, {int(self.masque.sum())} couples mois-puits, "
                f"{int(np.median(n))} mois en médiane par puits, "
                f"distance médiane {float(np.median(self.distance_km)):.1f} km")


def _chemin_defaut() -> str:
    racine = os.environ.get("MEANDRE_DERIVES")
    if not racine:
        from meandre.utils import paths as _paths

        racine = _paths.DERIVED_ROOT
    return f"{racine}/auxiliaires"


def read_rsesq(region: str, times, chemin: str | None = None, libres_seulement: bool = True,
               sans_influence: bool = True, distance_max_km: float = 10.0,
               mois_minimum: int = 24) -> CiblesNappe:
    """Cibles de nappe pour un domaine, agrégées au mois sur la période de `times`.

    times : index de dates journalières de la simulation. La grille mensuelle de sortie
    couvre exactement les mois qu'il contient, si bien que la perte peut indexer sans
    réalignement.
    """
    base = chemin or _chemin_defaut()
    puits = pd.read_parquet(f"{base}/rsesq-puits.parquet")
    puits = puits[puits.region.str.lower() == region.lower()]
    if libres_seulement:
        puits = puits[puits.confinement == "Libre"]
    if sans_influence:
        puits = puits[puits.influence != "Oui"]
    puits = puits[puits.distance_km <= distance_max_km]

    niveaux = pd.read_parquet(f"{base}/rsesq-niveaux-journaliers.parquet",
                              columns=["puits", "date", "niveau_m"])
    niveaux = niveaux[niveaux.puits.isin(puits.puits)]
    niveaux = niveaux.assign(mois=pd.to_datetime(niveaux.date).values.astype("datetime64[M]"))
    table = niveaux.groupby(["mois", "puits"]).niveau_m.mean().unstack()

    mois = pd.to_datetime(times).values.astype("datetime64[M]")
    grille = np.unique(mois)
    table = table.reindex(index=pd.DatetimeIndex(grille))

    # Un puits trop court ne porte aucune information de forme : on l'écarte ici plutôt que
    # de laisser le terme de perte le découvrir à chaque pas.
    assez = table.notna().sum(axis=0) >= mois_minimum
    table = table.loc[:, assez[assez].index]
    ordre = [p for p in puits.puits if p in table.columns]
    table = table[ordre]
    meta = puits.set_index("puits").loc[ordre]

    return CiblesNappe(puits=np.asarray(ordre, dtype=object),
                       node_idx=meta.node_idx.to_numpy(dtype=np.int64),
                       niveau=table.to_numpy(dtype=np.float32),
                       masque=table.notna().to_numpy(),
                       mois=grille,
                       distance_km=meta.distance_km.to_numpy(dtype=np.float32))


def index_mensuel(times) -> np.ndarray:
    """Indice du mois de chaque pas journalier dans la grille mensuelle de `read_rsesq`."""
    mois = pd.to_datetime(times).values.astype("datetime64[M]")
    grille = np.unique(mois)
    return np.searchsorted(grille, mois)
