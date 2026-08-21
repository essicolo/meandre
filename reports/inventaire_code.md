# Inventaire du code — état au 2026-08-21

Demandé par Essi : retirer ce qui a été écrit sur des hypothèses réfutées, en gardant une trace de la méthodologie. Ce document est l'inventaire préalable. Rien n'est supprimé sans qu'une ligne le justifie ici.

## Volumétrie

| ensemble | lignes | fichiers |
|---|---|---|
| `meandre/` (le paquet) | 22 178 | — |
| `.runs/` (scripts d'expérience) | 21 768 | **189** |
| `hydrotel_clone/` (clones fidèles) | 3 316 | — |

Le gisement principal n'est pas le paquet, c'est `.runs/` : presque autant de lignes, réparties sur 189 scripts dont la plupart ont servi une fois.

## Composants issus d'hypothèses RÉFUTÉES

Références mesurées dans le dépôt (paquet, scripts, tests) et nombre de configurations qui les activent.

| composant | paquet | scripts | tests | configs ON | statut |
|---|---|---|---|---|---|
| encodeur GRU temporel | 0 | 0 | 0 | 0 | **déjà retiré** |
| correcteur résiduel | 0 | 0 | 0 | 0 | **déjà retiré** |
| `TemporalModulator` | 1 | 0 | 0 | 0 | **MORT** |
| bruit d'état AR(1) | 4 | 0 | 0 | 0 | dormant |
| tête sigma hétéroscédastique | 7 | 0 | 0 | 0 | dormant, supplantée par la tête quantile |
| mélange de densités | 3 | 0 | 0 | 0 | dormant |
| attention sur le temps de trajet | 10 | 0 | 6 | 0 | dormant, mais testé |
| `ConcreteDropout` | 8 | 4 | 0 | **20** | à traiter avec prudence |
| réservoir quickflow à seuil | 1 | 0 | 0 | 1 | quasi mort (falsifié, 2 nulls) |
| modulateur de phénologie | 9 | 13 | 0 | 2 | **VIVANT** |

`ConcreteDropout` est le cas délicat : la mémoire du projet le dit abandonné au profit du probabiliste prédictif, mais **20 configurations l'activent encore**. Ce sont vraisemblablement d'anciennes configurations SLSO. À vérifier une par une avant de toucher au code.

## Modules jamais nommés dans un import (1 123 lignes)

ATTENTION : « jamais importé » ne veut pas dire « à supprimer ». Il faut distinguer deux familles.

**Outils d'ingestion à usage unique** — ils ne sont pas importés par le pipeline mais servent à RECONSTRUIRE les bases. Les supprimer rendrait les données non reproductibles. À GARDER, éventuellement à regrouper sous un nom explicite.

`hydat_loader` (154), `withdrawal_parquet_loader` (190), `withdrawals_loader` (110), `station_obs` (117), `pygmet_loader` (107), `territorial_loader` (104), `grace_multimascon` (101), `ghcn_loader` (90), `esa_cci_sm_loader` (82), `physitel_cache` (8).

**Vraiment mort** : `temporal_modulator` (60 lignes).

## Méthode proposée

Par étapes, chacune vérifiée par la suite de tests, chacune commitée séparément pour être annulable.

1. **Archiver** ce qui est mort dans `archive/` avec un `LISEZMOI.md` qui dit pour chaque pièce quelle hypothèse elle servait et par quelle mesure elle est tombée. Le code reste lisible dans l'historique ET sur disque.
2. **Clarifier** les outils d'ingestion : les regrouper et documenter qu'ils sont à usage unique, pas morts.
3. **Trancher `ConcreteDropout`** après audit des 20 configurations qui l'activent.
4. **Élaguer `.runs/`** : c'est le plus gros gisement. Classer les 189 scripts par chantier, archiver ceux dont l'hypothèse est réfutée au registre, garder ceux qui produisent des données ou des figures.
5. **Les 13 sorties mortes du champ spatial** (dette #5 du registre) : à retirer au moment de la reconstruction des caches, parce que ça casse la compatibilité des points de reprise.

## Ce qu'il ne faut PAS supprimer

Les clones fidèles de `hydrotel_clone/`, y compris les harnais de validation, même inutilisés au quotidien : ils sont la preuve que chaque pièce a été confrontée au binaire C++. Et la formulation d'Hydrotel du milieu humide (`conservatif=False`), gardée exprès depuis le 2026-08-20 pour les comparaisons module par module.
