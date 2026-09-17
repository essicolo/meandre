"""Ingestion standard des données auxiliaires.

Deux étages séparés. L'ingestion transforme une source brute, décrite par son `source.toml`,
en une table par tronçon qui ne sait rien de l'usage qu'on en fera : valeurs, couverture et
nature de la donnée (observée, interpolée, prédite). Le rôle de chaque variable, descripteur,
ancre, a priori, cible ou contrôle, se déclare ensuite dans la recette d'une expérience.
"""
from meandre.data.auxiliary.catalog import Source, load_source

__all__ = ["Source", "load_source"]
