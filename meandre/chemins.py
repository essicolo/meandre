"""Racines de donnees, en UN seul endroit et surchargeables par l'environnement.

Le depot portait 24 chemins absolus codes en dur (`D:/meandre-data/...`,
`C:/Users/parse01/...`), ce qui rendait toute execution ailleurs impossible : autre
poste, serveur de calcul loue, conteneur, machine d'un collegue. Les valeurs par
defaut sont celles du poste d'origine, donc rien ne change sans action explicite.

  MEANDRE_DATA         racine des donnees derivees (caches DuckDB, forcages, champs)
  MEANDRE_PLATEFORMES  plateformes Hydrotel (projets PHYSITEL, calages)
  MEANDRE_RQH          sorties de reference du RQH (zarr de post-traitement)

Note de gouvernance : MEANDRE_RQH et MEANDRE_PLATEFORMES pointent des donnees
INTERNES. Si des entrainements sont deportes sur un serveur externe, ces deux
racines ne doivent pas suivre : la comparaison a l'ensemble Hydrotel reste locale.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")
PLATEFORMES = os.environ.get(
    "MEANDRE_PLATEFORMES",
    "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel")
RQH = os.environ.get(
    "MEANDRE_RQH",
    "C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data")


def data(*parties: str) -> str:
    """Chemin sous la racine des donnees derivees."""
    return str(Path(DATA).joinpath(*parties)).replace("\\", "/")


def plateforme(*parties: str) -> str:
    """Chemin sous les plateformes Hydrotel (donnees INTERNES)."""
    return str(Path(PLATEFORMES).joinpath(*parties)).replace("\\", "/")


def rqh(*parties: str) -> str:
    """Chemin sous les sorties de reference du RQH (donnees INTERNES)."""
    return str(Path(RQH).joinpath(*parties)).replace("\\", "/")
