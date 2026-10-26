# Répertoire des barrages du CEHQ

Récolte, mise en table et instruction du dossier « barrages » pour meandre, à partir du [répertoire des barrages](https://www.cehq.gouv.qc.ca/barrages/default.asp) du Centre d'expertise hydrique du Québec, qui recense tous les barrages du Québec d'une hauteur de 1 m et plus.

Le code vit ici, les données vivent dans `D:/meandre-data/barrages/` (`data/` pour les tables, `cache/` pour les 6 115 fiches HTML brutes). Rien n'est versionné : tout se régénère depuis le site, et `02_fiches.py` saute ce qui est déjà en cache.

Deux dépendances externes. Le dépôt io-eau reste l'unique responsable du réseau de tronçons (`troncons.parquet`), des toponymes GRHQ, du découpage administratif BDAT et de la climatologie de débit ; la constante `IOEAU` des scripts 04 et 05 y pointe. Les bases régionales de meandre (`D:/meandre-data/quebec/*.duckdb`) fournissent la topologie et les stations, et `D:/meandre-data/hydat/Hydat.sqlite3` la base fédérale.

## Chaîne de traitement

Récolte et vérification :

| script | rôle | sortie |
|---|---|---|
| `01_liste.py` | énumère les barrages MRC par MRC (le formulaire refuse une requête sans critère) | `liste.csv` |
| `01b_recoupement_liste.py` | contre-énumération par municipalité, partition indépendante | `liste-municipalites.csv` |
| `01c_sondage_numeros.py` | sonde chaque numéro manquant du bloc dense X0000462-X0008036 ; une fiche inconnue renvoie une erreur 500 | `numeros-orphelins.csv` |
| `02_fiches.py` | télécharge la fiche technique de chaque barrage | `cache/<no>.html` |
| `03_parse.py` | met les fiches en table | `barrages.parquet`, `barrages-hydrographie.parquet` |
| `04_mapping.py` | rattache chaque barrage à un tronçon et juge la qualité du rattachement | `mapping-barrages-troncons.csv`, `barrages.geojson` |
| `05_verification.py` | complétude, cohérence interne, exactitude du rattachement | `RAPPORT-VERIFICATION.md`, `anomalies.csv` |

Instruction du dossier de modélisation :

| script | question | sortie |
|---|---|---|
| `06_candidats_reservoirs.py` | quelles retenues pèsent assez pour mériter un nœud de stockage | `candidats-reservoirs.csv` |
| `07_amont_stations.py` | lesquelles ont une station en aval, donc peuvent être calées | `candidats-reservoirs-stations.csv` |
| `08_stations_regulees.py` | quelles stations d'entraînement portent une régularisation non modélisée | `stations-regulees.csv` |
| `09_gain_hydat.py` | ce que le réseau fédéral ajouterait sur 2001-2024 | `gain-hydat.csv` |
| `10_fenetre_historique.py` | ce qu'il ajoute sur 1980-1994 | `fenetre-historique.csv` |

Les scripts se relancent dans l'ordre. `04` et au-delà supposent que io-eau et les bases régionales sont en place.

## Deux pièges du site

Le formulaire est servi par un ASP classique en windows-1252. Un corps de requête envoyé en UTF-8 ne renvoie aucune ligne, silencieusement : une première récolte a ainsi perdu 36 MRC sur 103, toutes celles dont le nom porte un accent ou une apostrophe. Le corps est donc encodé en cp1252, et les valeurs des `<option>` sont déséchappées avant d'être renvoyées. Une requête sans aucun critère est redirigée vers la page d'accueil plutôt que de renvoyer le répertoire complet, d'où l'énumération par MRC.

Surtout, l'énumération par MRC n'est pas exhaustive : 271 fiches sont servies par `detail.asp` sans jamais apparaître dans les résultats de recherche, dont les 264 barrages de la Jamésie, MRC pour laquelle la recherche renvoie zéro ligne. C'est pourquoi `01c` interroge les fiches une par une plutôt que de faire confiance à l'index. Le bloc clairsemé X2000000 et plus n'a pas été sondé (103 000 numéros) : il peut encore cacher des fiches non indexées.

Rien de tout cela n'existe en données ouvertes. Vérifié le 2026-08-27 : les 129 jeux du MELCC sur Données Québec ne contiennent aucun répertoire de barrages, seulement les ouvrages de protection contre les inondations.

## Ce que le rattachement peut et ne peut pas dire

Le réseau meandre compte 28 121 tronçons pour l'ensemble du Québec, alors que le barrage médian du répertoire retient un bassin de 4,4 km². La plupart des barrages sont donc posés sur un cours d'eau qui n'existe pas dans le réseau : `sjoin_nearest` les rattache malgré tout, au ruisseau modélisé le plus proche, ce qui est un artefact et non une localisation. `04_mapping.py` rend ce diagnostic explicite avec une colonne `qualite` :

- `assuré` : tronçon à moins de 150 m et confirmé soit par le toponyme, soit par la superficie du bassin versant ;
- `sous-maille` : le tronçon draine plus de cinq fois le bassin déclaré, ou, à défaut de débit simulé, le bassin déclaré fait moins de 10 km² ;
- `hors couverture` : le point ne tombe dans aucune des 15 régions hydrographiques modélisées ;
- `douteux` : le reste, à examiner.

Le contrôle décisif n'est pas la distance mais la superficie : la fiche donne le bassin versant amont, ce qui permet de confronter le rattachement au débit climatologique simulé sur le tronçon retenu. Sur les rattachements assurés, le débit spécifique implicite vaut 21,6 L/s/km², la valeur attendue au Québec méridional. Voir `RAPPORT-VERIFICATION.md`.

## Suite

[PROMPT-INGESTION.md](PROMPT-INGESTION.md) propose la méthodologie d'ingestion dans meandre, avec les verrous établis par les scripts 06 à 10 et les portes de décision.
