# Vérification du répertoire des barrages (CEHQ)
Récolte du 2026-08-28, source <https://www.cehq.gouv.qc.ca/barrages/default.asp>.
## A. Complétude de la récolte

- barrages listés (partition par MRC) : **5844**
- fiches techniques analysées : **6115**
- listés sans fiche : **0**
- fiches hors liste : **0**
- numéros uniques : **6115**
- contre-énumération par municipalité (`01b_recoupement_liste.py`) : interrompue à 300 municipalités sur 767, le moteur de recherche du site cessant de répondre sous la charge. Le sondage direct des numéros ci-dessous répond à la même question plus vite et plus franchement, en interrogeant les fiches et non l'index.
- sondage direct des 3014 numéros manquants du bloc dense X0000462-X0008036 : **271** fiches servies par `detail.asp` mais absentes du moteur de recherche du site

  Ces fiches sont réelles et ont été ajoutées à la récolte. L'écart ne vient pas du grattage : réinterrogée, la recherche par MRC rend exactement la même liste que celle enregistrée (Charlevoix-Est, 108 lignes des deux côtés), sans le barrage X0001110 que `detail.asp` sert pourtant.

  Le gros du lot est concentré : **264** des 271 sont dans la MRC de Jamésie, la seule dont la recherche renvoie zéro ligne alors que le territoire porte des barrages. Les 7 autres sont dispersés (Vaudreuil-Soulanges, Charlevoix-Est, Caniapiscau) et relèvent d'un index de recherche incomplet.

## B. Exactitude interne des fiches

- municipalité divergente entre la liste et la fiche : **0**
- sans coordonnées : **1**
- coordonnées hors de l'emprise du Québec : **0**
- barrages partageant une coordonnée exacte avec un autre : **51** (sur 25 points distincts)
- hauteur de la retenue supérieure à la hauteur du barrage : **29**
- année de construction absurde : **0**
- modification antérieure à la construction : **8**
- capacité de retenue dépassant le majorant géométrique hauteur de la retenue × superficie du réservoir : **139** sur 3024 testables (58 % des fiches posent l'égalité, les autres restent en dessous, ce qui est le cas normal)
- catégorie administrative non conforme à la loi : **0** sur 5964 testables
- fiches citant un barrage amont/aval absent du répertoire récolté : **121**

Confrontation de la coordonnée au découpage administratif (BDAT), qui ne vient pas du CEHQ :

- coordonnée hors de toute municipalité : **3**
- municipalité déclarée contredite par la coordonnée : **49** sur 6112 (0.8 %)
- MRC déclarée contredite par la coordonnée : **302** (4.9 %), pour l'essentiel des libellés et non des erreurs de position : Jamésie porte la mention « (terr. conventionné) » côté CEHQ, Beauce-Centre est le nouveau nom de Robert-Cliche

| MRC déclarée | MRC du polygone | n |
|---|---|---|
| Jamésie (terr. conventionné) | Jamésie | 264 |
| Beauce-Centre | Robert-Cliche | 20 |
| Administration régionale Kativik | Kativik | 3 |
| Eeyou Istchee | Nouveau toponyme à venir | 2 |
| Roussillon | Les Jardins-de-Napierville | 2 |
| Vaudreuil-Soulanges | Beauharnois-Salaberry | 1 |
| Brome-Missisquoi | La Haute-Yamaska | 1 |
| Argenteuil | La Rivière-du-Nord | 1 |

## C. Exactitude du rattachement aux tronçons

- barrages rattachés : **6114**
- distance au tronçon : médiane **499 m**, p90 **2729 m**, max **987326 m**
- au-delà de 500 m : **3054** ; au-delà de 5 km : **302**
- relocalisés vers un tronçon-lac : **295**
- toponyme du tronçon en accord avec la fiche : **2394**, en désaccord : **3578**, indécidable (un nom manque) : **142**

Contrôle indépendant par le débit : la superficie du bassin versant déclarée sur la fiche est confrontée au débit moyen climatologique simulé sur le tronçon retenu.

- couples testables (superficie déclarée et débit simulé) : **1789**
- débit spécifique implicite sur les rattachements assurés (n = 518) : **21.6 L/s/km²**, soit la valeur attendue au Québec méridional (15 à 30 L/s/km²) : les deux sources, indépendantes, se recoupent
- sur l'ensemble des couples testables : **307 L/s/km²**, valeur aberrante qui mesure exactement le défaut de résolution du réseau plutôt qu'une erreur des fiches
- tronçon drainant plus de 5 fois le bassin déclaré : **1085** (61 %)
- tronçon drainant moins du cinquième du bassin déclaré : **12**

Verdict par barrage :

| qualité | n | superficie médiane du bassin (km²) | distance médiane (m) | toponyme en accord |
|---|---|---|---|---|
| sous-maille | 2696 | 1.5 | 858 | 32% |
| douteux | 1800 | 30.25 | 614 | 26% |
| assuré | 1282 | 74.3 | 15 | 86% |
| hors couverture | 336 | 7449.0 | 227341 | 2% |

## Anomalies

5206 signalements sur 4065 barrages, détaillés dans [data/anomalies.csv](data/anomalies.csv).

| code | n |
|---|---|
| toponyme_en_desaccord | 3578 |
| troncon_trop_grand | 1085 |
| rattachement_lointain | 302 |
| volume_impossible | 139 |
| municipalite_contredite | 49 |
| hauteur_retenue_superieure | 29 |
| troncon_trop_petit | 12 |
| modification_avant_construction | 8 |
| coordonnee_hors_polygone | 3 |
| coordonnees_absentes | 1 |

