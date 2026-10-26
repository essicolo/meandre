# Résultats du plan, au fil de l'eau

EN COURS, NE PAS SUPPRIMER. Plan dans PLAN.md.

## 0.1 Routage operator contre operator-lagged, Outaouais, poids gelés (2026-10-23, matin)

Modèle provincial gelé (best-quebec-etl-qv848-e32, entraîné 2001-2018 en mode operator-lagged), passe avant sans entraînement sur l'Outaouais, 16 stations, 2019-2024. Seule différence : routing_mode dans le TOML (experiments/plan_ml/modele-provincial-operator.toml). Journaux et séries dans D:/meandre-data/experiments/plan_ml ; comparaison par comparer_routage.py.

| Mesure, 2019-2024, 16 stations | operator-lagged | operator |
|---|---|---|
| Perte complète du pilote (termes de débit pondérés) | 0,884 | 0,898 |
| KGE médian du pilote | 0,710 | 0,686 |
| KGE médian recalculé sur tous les jours mesurés (comparer_routage.py) | 0,675 | 0,680 |
| Stations en retard d'au moins un jour (corrélation croisée) | 8 | 7 |
| Délai de réponse à la pluie, simulé moins observé, moyenne | +0,75 j | +0,31 j |
| Écart de volume médian | 6,3 % | 6,4 % |
| Erreur d'étiage médiane (log du rapport des minimums de 7 jours) | 0,284 | 0,284 |
| Variation journalière relative médiane (observée 0,139) | 0,081 | 0,086 |

Par nombre de lacs en série vers la station (délai de réponse simulé moins observé, jours) : 0 lac, 5 stations, −0,60 dans les deux modes ; 1 à 2 lacs, 3 stations, +0,67 contre −0,33 ; 3 à 4 lacs, 5 stations, +1,40 contre +1,20 ; 5 lacs et plus, 3 stations, +2,00 contre +1,00. KGE médian par classe inchangé sauf 5 lacs et plus, 0,55 contre 0,50.

Lecture. Le mode lagged porte bien une partie du retard : passer à operator retire 0,4 jour de délai en moyenne, et un jour entier aux stations à cinq lacs et plus. Mais à poids gelés la perte ne s'améliore pas (0,884 contre 0,898), le volume et l'étiage sont identiques, et l'hydrogramme reste aussi amorti (variation journalière 0,086 contre 0,139 observée). Deux explications non départagées : les paramètres de routage et de lacs ont été appris sous le mode lagged et compensent déjà une partie du retard (K dans [4, 48] h, k et β des lacs) ; ou l'amortissement vient d'ailleurs que du retard, ce que les stations sans lac suggèrent (délai juste, hydrogramme amorti quand même).

Ce que seul un entraînement dirait : si les paramètres de routage se réadaptent au mode operator et si la perte descend alors sous 0,884. Coût : affinage court sur l'Outaouais à 1e-5, quatre époques, environ deux heures. Non lancé ce matin : l'amortissement aux stations sans lac ne sera pas résolu par ce changement, et c'est lui qui porte l'erreur quadratique (70 % sur les 10 % de jours les plus forts, R389). La décision est reportée après la candidate 5.6 (forme de la réponse rapide), qui vise directement cet amortissement.

Verdict provisoire : le retard des lacs en série est pour moitié un artefact du mode lagged, pour moitié dans le tarage ou le routage lui-même ; le changement de mode seul ne rapporte rien à poids gelés. À consigner au registre.

## 0.2 Banc de sensibilité sur la physique d'octobre, Montérégie (2026-10-23, 10 h 42)

`sensibilite_des_observables.py`, déporté sur Ubuntu, modèle provincial gelé, zéro époque, 365 jours en 25 blocs de 15, un multiplicateur unité par champ du réseau spatial (43). Septembre avait mesuré trois directions sur cette physique d'avant le sol semi-implicite et la couche 3 déclarée ; voici la mesure à jour.

| Mesure, Montérégie, 43 champs | valeur |
|---|---|
| directions de paramètres distinguées par l'ensemble des six observables (débit, évapotranspiration, gravimétrie, nappe, neige, recharge) | 4 |
| directions du témoin de bruit, même forme, mêmes normes de ligne | 19 |
| part de participation de l'ensemble | 2,7 sur 43 |
| directions par observable pris seul | 1 pour chacun des six |
| champs qu'aucun observable ne touche | 25 sur 43 |

Le même banc sur l'Outaouais donne le même compte : 4 directions sur 43, participation 3,1, gravimétrie seule à en distinguer 2, neige à 1, et les mêmes 25 champs muets, mot pour mot. La hiérarchie des champs vivants est aussi la même : theta_fc_3 et porosity_2 d'abord, puis porosity_3, porosity_1 et Z3. Deux territoires de taille et d'occupation très différentes donnent donc la même lecture.

Les quatre directions, nommées. Le débit porte porosity_3 contre theta_fc_3 (charges −0,71 et +0,62). L'évapotranspiration porte K_c (−0,97) : c'est bien la seule chose qu'elle regarde. La gravimétrie porte theta_fc_3 contre porosity_2 (−0,74 et −0,63). La neige porte C_f (+0,98). La nappe et la recharge reprennent la direction du débit et du stock, sans en ajouter une.

Les champs muets : theta_fc_2, theta_wp_2, theta_wp_3, f_root_1, f_root_2, f_root_3, f_vert_1, f_vert_2, f_vert_3, f_wetland, frost_alpha, manning_n, interception_capacity, T_snow, alpha_T, K_atm, dT_canopee_conif, dT_canopee_feu, krec, vsa_b, rain_hours, k_sub, k_gw, T_gw, vg_n.

Ce que cela change pour le plan. La candidate 5.3 ne peut pas s'appuyer sur le champ vsa_b tel qu'il est : rien ne le lit, et le banc le confirme par une sensibilité nulle. Si l'exposant doit devenir spatial, il faut d'abord le brancher là où bv3c2 lit la part, sinon il n'aura pas de gradient. Les candidates 5.2 (k_gw, T_gw muets) et 5.5 (manning_n muet) restent sans juge dans l'espace des paramètres, ce qui est cohérent avec leurs verdicts. Le nombre de directions utiles, 4, se lit contre les 19 du témoin de bruit : une matrice sans aucune structure, de mêmes dimensions et de mêmes normes de ligne, en distinguerait 19. Les sensibilités réelles sont donc fortement colinéaires, et six observations ne suffisent pas à séparer les quarante-trois champs.

## 0.3 Audit des sorties du champ spatial sans lecteur (2026-10-23, matin)

Comptage des lectures de chaque sortie de `SpatialParams` (43 champs) hors de `field_network.py`, par attribut et par nom en chaîne, dans `meandre/`, `hydrotel_clone/`, le pilote et le banc, et croisement avec la recette provinciale.

Sorties que RIEN ne lit, en aucun mode : `f_root_1`, `f_root_2`, `f_root_3`, `T_snow`, `interception_capacity`, `frost_alpha`, `f_vert_1`, `f_vert_2`, `f_vert_3`, `alpha_T`, `vg_n`, `rain_hours`, `vsa_b`, `f_wetland` (la colonne lit la fraction brute du projet PHYSITEL, pas le champ). Quatorze sorties sur 43 ; aucune n'a de gradient, et le réseau dépense une partie de sa dernière couche à les produire.

Sorties lues seulement par un module inactif dans la recette provinciale : `T_gw` et `K_atm` (thermie du cours d'eau, hors perte) ; `manning_n` (niveau d'eau pour SWOT, retourne None faute de largeur de lit dans les bases) ; `k_sub` (plafond de percolation nommé par chaîne dans `[soil]`, mais la recette provinciale emploie la règle par classe de drainage, donc `k_sub` n'est pas lu).

Sorties vivantes : K_sat_1/2/3, porosity_1/2/3, theta_fc_1/2/3 et theta_wp_1/2/3 (la colonne les passe toutes au sol ; la sensibilité nulle de theta_fc_2, theta_wp_2 et theta_wp_3 mesurée en septembre vient de la physique de bv3c2 qui ne consomme pas ces valeurs, pas d'un défaut de branchement), C_f, T_melt, k_gw, K_musk_hours, x_musk, K_c, Z2, Z3, krec, diff_gel, fs_neige, dT_canopee_feu, dT_canopee_conif.

Conséquences pour le plan : la candidate 5.3 (aires saturées) doit se brancher à l'endroit où `bv3c2` lit l'exposant, et `vsa_b` est le champ naturel à y connecter ; aucune candidate ne doit s'appuyer sur les quatorze sorties mortes sans les brancher d'abord. Retirer ces sorties de la tête du réseau changerait la dimension de sortie et romprait la lecture des points de reprise existants ; décision à prendre par Essi, hors de ce plan.

## 5.2 Loi stock-débit de la nappe : épreuve 3.3 hors ligne sur les récessions observées (2026-10-23, matin)

Débits observés 2000-2024 des bases, jours mesurés seulement, 155 stations, 17 350 points d'enveloppe inférieure de décrue franche (Brutsaert-Nieber), juin à novembre. Cible : logarithme de −dQ/dt en débit spécifique, en fonction du logarithme de Q centré sur la médiane de la station (`nappe_recession.py`). Quatre lois, de la plus contrainte à la plus libre, chacune avec une échelle propre par station.

| Loi | A. années retirées, stations vues | B. stations retirées, échelle estimée sur 2000-2012, forme venue des autres, évaluation 2013-2024 |
|---|---|---|
| puissance par station, exposant propre | 45,5 % | 45,9 % |
| puissance à exposant commun | 41,6 % | 42,1 % |
| forme commune monotone libre (36 nœuds) | 42,3 % | 42,9 % |
| forme commune conditionnée par les attributs du tronçon | 41,3 % | 44,9 % |

Les chiffres sont l'erreur relative médiane sur −dQ/dt. Exposant b par station : médiane 1,50, quartiles 1,32 à 1,72 (1,60 au registre de septembre, mesuré sur toute l'année). La forme commune apprise a une pente locale (exposant b local) de 1,2 à 1,6 autour de la médiane, qui tombe sous 1 aux très bas débits et au-dessus de la médiane haute, mais cette courbure n'améliore rien hors échantillon.

Lecture. La loi de puissance à exposant COMMUN est la meilleure des quatre aux stations retirées. L'exposant propre à chaque station est du bruit (45,9 contre 42,1 %), ce qui pèse aussi sur l'idée d'un exposant par territoire. La forme non linéaire libre ne bat pas la puissance (42,9 contre 42,1 %), et la conditionner par les attributs du tronçon aide aux stations vues (41,3 %) et nuit aux stations retirées (44,9 %) : même signature d'ajustement que pour le correcteur aval. Le carré de Boussinesq (b = 1,5, n = 2) actuellement codé est la valeur mesurée.

Verdict : candidate 5.2 arrêtée à l'épreuve 3.3 ; il n'y a pas de forme commune à apprendre au-delà de la puissance, et la loi actuelle est juste. Ce qui reste à expliquer dans l'étiage (conductance spatiale, extraction) n'est pas une question de forme de la loi.

## 5.1 Fonte : épreuve 3.3 hors ligne contre l'ablation mesurée par CanSWE (2026-10-23, matin)

Cible : baisse de masse du manteau entre deux relevés à un jour d'intervalle, un jour sans précipitation CaSR, bornée à zéro ; Outaouais, Saguenay, Saint-Laurent nord-ouest (la Gaspésie n'a que 19 jours-sites recevables). Sur toute l'année, la cible est bruitée : 57 % des « jours de fonte » ont une température moyenne négative et 17 % des jours sous −10 °C montrent une baisse de plus de 0,5 mm (vent, tassement, mesure) ; une loi apprise sur tout apprend ce bruit. L'épreuve retenue porte sur la SAISON DE FONTE, mars à juin, stock de plus de 10 mm : 10 059 jours-sites secs, 50 sites, 56 % de jours de fonte, ablation moyenne 7,19 mm/j les jours où il fond ; 10 % seulement de jours bloqués (aucune variation par T > 2 °C). Scripts : `fonte_observee.py fonte` et compléments dans le journal `fonte_observee-saison.log`.

Ce que la cible montre avant tout modèle : il fond un jour sur trois sous −5 °C de moyenne (3,2 mm/j) et un jour sur deux entre −5 et −2 °C (4,0 mm/j), sous un rayonnement net de 8 à 9 ; au-dessus de 0 °C l'ablation croît à peu près linéairement, 7,7 mm/j à 1 °C, 12,4 à 9 °C. La corrélation de l'ablation avec le rayonnement cumulé sur sept jours (0,45) dépasse celle avec la température du jour (0,37). Une loi en max(T − T_melt, 0) ne peut pas rendre la fonte radiative par temps froid, et le réajustement la pousse à T_melt = −5 °C, sa borne.

Écart absolu moyen de l'ablation les jours de fonte, en mm/j (moyenne observée 7,19) :

| Loi | Sites retirés | Années retirées | Territoire retiré |
|---|---|---|---|
| degrés-jours d'Hydrotel réajustée (cf, T_melt), plafonnée au stock | 4,24 | 4,22 | 4,23 |
| température plus rayonnement, bornée, 3 paramètres | 4,10 | | 4,10 |
| idem plus température cumulée sur 7 jours, 4 paramètres | 3,69 | | 3,71 |
| arbres monotones en T et rayonnement, variables du jour | 3,38 | 3,78 | 3,46 |
| arbres monotones plus cumuls 3 et 7 jours | 3,15 | 3,34 | 3,17 |
| idem plus attributs du site (couvert, altitude, latitude) | 3,25 | 3,00 | 3,59 |
| idem plus coordonnées | 3,22 | 3,02 | 3,50 |

Lecture. Le seuil d'arrêt posé d'avance (15 % de gain sur la loi réajustée aux sites retirés) est franchi : la fonction monotone apprise réduit l'erreur de 26 % (4,24 à 3,15), et de 25 % au territoire jamais vu. La moitié de ce gain est accessible à une forme PHYSIQUE à quatre paramètres, température, rayonnement et état thermique du manteau par la température cumulée sur sept jours (3,69), ce qui est la route préférée ; l'autre moitié (3,69 à 3,15) demande une forme libre, monotone. Les cumuls comptent (3,46 à 3,17 : l'état du manteau, pas la mémoire au sens d'un GRU). Pour la troisième fois en deux jours, le conditionnement par les attributs ou par les coordonnées aide quand les lieux sont vus et NUIT au territoire retiré (3,59 et 3,50 contre 3,17) : une loi de fonte commune transfère, une loi spatialisée n'y gagne rien.

Suite proposée pour 5.1, dans l'ordre : (a) remplacer la loi degrés-jours par la forme physique bornée à quatre paramètres (T, rayonnement net, température cumulée sept jours, seuil), déclarée au catalogue de la neige, initialisée sur la loi actuelle ; (b) sonde de gradient et passe avant sur l'Outaouais et la Gaspésie ; (c) seulement si le reste de l'écart compte pour le débit d'avril, la forme libre monotone, à poids partagés, sans conditionnement spatial. L'épreuve 3.6 (GRU contre cumuls) n'est pas faite ; les cumuls portent déjà l'état thermique et les deux comparaisons précédentes ont donné le GRU perdant, elle passe derrière (a) et (b).

## 5.4 Porte de gel : épreuve 3.1, réponse observée aux pluies d'automne selon l'état de gel, et témoin simulé (2026-10-23, matin)

`sol_gele.py`, six territoires, 117 stations, 9 844 pluies de novembre-décembre d'au moins 5 mm par température moyenne positive, jours mesurés, forçage CaSR brut. Réponse : hausse du débit sur trois jours rapportée à la pluie, puis rapportée à la réponse médiane de la même station sur sol non gelé (indice de gel sous 5 °C·jour). Indice de gel : degrés-jours négatifs cumulés depuis le 15 octobre.

| Indice de gel (°C·jour) | Pluies | Réponse relative observée (médiane) | Réponse relative simulée, modèle provincial gelé, 2019-2024 |
|---|---|---|---|
| 0 à 5 | 3 608 | 1,00 | 1,00 |
| 5 à 20 | 2 553 | 1,16 | 0,75 |
| 20 à 50 | 2 113 | 1,44 | 1,18 |
| 50 à 100 | 1 126 | 1,97 | 1,87 |
| 100 à 200 | 367 | 2,74 | 1,77 |
| 200 et plus | 77 | 1,83 (37 stations, manteau de 74 mm) | 1,51 |

(Colonne simulée : 131 stations, 3 239 pluies de 2019-2024, mêmes classes ; l'observé sur ces mêmes pluies vaut 1,00, 0,99, 1,32, 2,43, 2,61, 2,63.)

Le manteau amplifie encore la réponse à gel égal : à 20-50 °C·jour, 1,22 sans neige, 2,12 avec 5 à 30 mm, 3,19 au-delà de 30 mm. Par territoire, sous gel de 20 à 100 : Outaouais 2,05 (91 % des stations répondent plus sous gel), Montérégie 1,88 (100 %), Saint-Laurent nord-ouest 1,61 (80 %), Saguenay 1,08 (53 %), Gaspésie 1,34 (73 %), Saint-Laurent sud-ouest 1,52 (88 %). Aucun attribut amont ne prédit la réponse sous gel entre stations (corrélations de rang de −0,12 à +0,08).

Lecture. La réponse croît de façon CONTINUE et monotone avec l'indice de gel, sans palier : c'est la signature d'une porte continue, et elle est commune aux territoires forestiers et agricoles. Le modèle actuel (porte continue, rampe linéaire en profondeur de gel) reproduit le sens de l'effet mais SATURE : réponse simulée stable à 1,5 à 1,9 là où l'observée atteint 2,4 à 2,6 ; et par territoire il répond trop en Outaouais, Gaspésie et Saguenay (1,91, 1,45, 1,21 contre 1,52, 1,22, 0,70 observés) et pas assez en Montérégie et au Saint-Laurent sud-ouest (2,20 et 1,16 contre 2,96 et 1,73). Le témoin porte donc le défaut, d'amplitude et de forme, pas de signe.

Verdict : candidate 5.4 passe l'épreuve 3.1. La forme à essayer d'abord est PHYSIQUE : une porte qui ne sature pas à la profondeur de la couche 1 et qui combine gel et manteau (la pluie sur neige mobilise aussi de la fonte). Épreuves suivantes : sonde de gradient (3.2), passe avant avec la même statistique sur le simulé (3.4). Une forme apprise n'est justifiée que si cette forme physique laisse encore l'écart d'amplitude.

Note : la table d'attributs du dossier gru_residual portait des valeurs TRANSFORMÉES pour le réseau (l'aire y valait 1,3 pour 1 300 km²) ; les tests précédents n'en dépendaient que par des rapports par station, non affectés ; `aire_km2` est désormais lue de la table des stations.

## 5.3 Aires saturées : épreuve 3.1, part ruisselée des orages d'été selon l'humidité antécédente, observée et simulée (2026-10-23, matin)

`orages_ete.py`, 5 357 orages de juin à septembre (pluie de bassin d'au moins 10 mm après deux jours à moins de 2 mm), 144 stations, 2019-2024, débit simulé du modèle provincial gelé. Coefficient d'écoulement : hausse cumulée du débit spécifique sur cinq jours au-dessus du débit de la veille, rapportée à la pluie des cinq jours. Humidité antécédente : indice de précipitation antérieure (décroissance 0,9 par jour), rapporté à la médiane de la station.

| Humidité antécédente (API relatif) | Orages | Coefficient observé (médiane) | Coefficient simulé | Simulé / observé |
|---|---|---|---|---|
| sous 0,5 | 537 | 0,025 | 0,035 | 1,43 |
| 0,5 à 0,8 | 1 128 | 0,033 | 0,037 | 1,12 |
| 0,8 à 1,2 | 1 962 | 0,040 | 0,038 | 0,94 |
| 1,2 à 1,6 | 973 | 0,054 | 0,041 | 0,76 |
| 1,6 à 2,5 | 691 | 0,065 | 0,037 | 0,57 |

| Pluie de l'événement (mm sur cinq jours) | Orages | Observé | Simulé | Simulé / observé |
|---|---|---|---|---|
| 10 à 15 | 415 | 0,008 | 0,022 | 2,87 |
| 15 à 25 | 1 430 | 0,017 | 0,027 | 1,65 |
| 25 à 40 | 1 766 | 0,039 | 0,038 | 0,99 |
| plus de 40 | 1 746 | 0,096 | 0,058 | 0,60 |

Le coefficient médian global est presque juste (0,041 observé, 0,038 simulé ; rapport médian par station 0,90) : le défaut n'est pas de niveau, il est de FORME. Le coefficient observé croît de 0,025 à 0,065 du sol sec au sol humide ; le simulé reste à 0,035 à 0,041 quelle que soit l'humidité antécédente. Le coefficient observé croît de 0,008 à 0,096 des petites aux grosses pluies ; le simulé va de 0,022 à 0,058, soit trois fois trop de réponse aux petites pluies et 40 % trop peu aux grosses. C'est exactement l'amortissement mesuré hier (variation journalière 0,16 contre 0,22) et le rapport des pointes en Montérégie.

Par territoire, corrélation de rang du coefficient observé avec l'humidité antécédente : Montérégie +0,32 (simulé +0,10), Saint-Laurent sud-ouest +0,21 (+0,09), Labrador +0,36 ; nulle en Outaouais, Saguenay et Abitibi, où les lacs amortissent (le coefficient observé entre stations décroît avec la fraction de lacs, corrélation de rang −0,34 ; aucun autre attribut ne le prédit). Le rapport simulé / observé n'est prédit par aucun attribut (de −0,21 à +0,09).

Lecture. La part ruisselée dépend fortement de l'état du sol là où il n'y a pas de lacs, et le modèle ne le rend pas : sa réponse rapide est insensible à l'humidité antécédente et à la taille de la pluie, comme si la part rapide était une fraction fixe. Deux causes possibles, à départager par la sonde et les diagnostics : la loi (theta1/thetas1)^8 reçoit une humidité de couche 1 qui ne varie presque pas (le registre notait une teneur en eau qui ne bouge pas de plus de trois pour cent avant le drainage de la couche 3), ou la réponse rapide passe par un autre chemin (écoulement latéral de couche 2 en un jour, R379) qui ne lit pas l'état. Dans les deux cas le témoin porte le défaut, et une fonction apprise de la part ruisselée aurait la bonne cible (ce coefficient par événement, 5 357 points, sans simulation) ; mais la première chose à vérifier est que la variable d'entrée de la loi, l'humidité de surface, varie réellement dans la colonne.

Verdict : candidate 5.3 passe l'épreuve 3.1 avec un défaut de forme net et mesuré. Suite : diagnostics de theta1 et de la part rapide sur un sous-bassin de Montérégie (banc, trois minutes), sonde de gradient de l'exposant, puis forme physique (exposant plus bas ou saturation par aire contributive dépendant de theta2) avant toute forme apprise.

## 5.3, épreuve 3.4 : l'exposant des aires saturées porte-t-il la forme ? (2026-10-23, 11 h 20)

Balayage de l'exposant b de la part (theta1/thetas1)^b sur le pilote, modèle provincial gelé, zéro époque, tout le reste figé (variable d'environnement MEANDRE_VSA_B, qui prime sur le fichier déclaré). Trois valeurs, trois territoires (Montérégie, Saint-Laurent sud-ouest, Outaouais), même statistique d'orage que l'épreuve 3.1, recalculée sur les séries du pilote par `balayage_vsa.py`.

Coefficient d'écoulement médian par classe d'humidité antécédente, 2 748 orages, 68 stations :

| b | coefficient médian simulé | observé | du sol sec au sol humide, simulé | du sol sec au sol humide, observé | corrélation de rang simulée avec l'humidité | volume simulé / observé |
|---|---|---|---|---|---|---|
| 2 | 0,111 | 0,044 | 0,121 puis 0,082 | 0,020 puis 0,077 | −0,13 | 1,07 |
| 8 (actuel) | 0,020 | 0,044 | 0,021 puis 0,014 | 0,020 puis 0,077 | −0,08 | 1,03 |
| 16 | 0,016 | 0,044 | 0,016 puis 0,007 | 0,020 puis 0,077 | −0,08 | 1,03 |

Effet sur la perte et sur ses termes, mêmes passes, 2019-2024 :

| territoire | b | KGE médian | perte totale | étiage | volume (pbias) |
|---|---|---|---|---|---|
| Outaouais | 8 | 0,6874 | 0,9316 | 0,4266 | 0,0312 |
| Outaouais | 2 | 0,7106 | 1,1111 | 0,5891 | 0,0416 |
| Saint-Laurent sud-ouest | 8 | 0,6285 | 1,1202 | 0,4648 | 0,0509 |
| Saint-Laurent sud-ouest | 2 | 0,6871 | 1,4432 | 0,8796 | 0,0509 |
| Saint-Laurent sud-ouest | 16 | 0,6028 | 1,1389 | 0,4622 | 0,0510 |
| Montérégie | 8 | 0,5728 | 1,4291 | 0,7145 | 0,0752 |
| Montérégie | 2 | 0,5883 | 1,7748 | 1,1474 | 0,0665 |
| Montérégie | 16 | 0,5408 | 1,4549 | 0,7161 | 0,0755 |
| Outaouais | 16 | 0,6795 | 0,9370 | 0,4290 | 0,0311 |

Lecture. L'exposant ne porte pas la forme. À b = 2, 8 et 16, la réponse simulée au coefficient d'écoulement reste PLATE ou DÉCROISSANTE avec l'humidité antécédente (corrélation de rang de −0,08 à −0,13) là où l'observé croît (+0,20), et ce n'est pas un problème d'échelle : à b = 2 le coefficient simulé vaut 2,5 fois l'observé et la pente reste négative. L'exposant déplace le NIVEAU (le volume passe de 1,07 à 0,99 de l'observé), jamais le SENS de la dépendance.

Le coût est en outre défavorable dans le seul sens qui augmente l'écoulement : à b = 2 le KGE médian monte (Outaouais 0,6874 à 0,7106, Saint-Laurent sud-ouest 0,6285 à 0,6871) mais la PERTE se dégrade partout (0,9316 à 1,1111, 1,1202 à 1,4432, 1,4291 à 1,7748), parce que l'étiage paie la double ration (0,4266 à 0,5891 ; 0,4648 à 0,8796 ; 0,7145 à 1,1474). L'objectif est la perte, pas le KGE : b = 2 est écarté.

Verdict pour 5.3. La forme physique « exposant plus bas », annoncée dans la suite de l'épreuve 3.1, est RÉFUTÉE comme levier : l'exposant ne fait pas dépendre le ruissellement de l'état du sol dans le sens observé. Le défaut de forme vient donc d'un chemin rapide qui ne lit pas la saturation de la couche 1, comme le suggérait déjà le registre (écoulement latéral de couche 2 en un jour). La prochaine mesure doit nommer ce chemin, et elle demande les diagnostics journaliers par tronçon du pilote, que rien n'expose aujourd'hui : `ETL_DUMP_REACH` ne donne que des moyennes mensuelles et ne porte ni le ruissellement de surface ni l'écoulement latéral.

## 5.3, suite : quel chemin porte la réponse rapide (2026-10-23, 11 h 45)

Le verdict de l'épreuve 3.4 laissait une question : si l'exposant ne fait rien, quel chemin produit l'eau rapide ? Pour y répondre, le pilote écrit désormais les diagnostics JOURNALIERS aux seuls nœuds portant une station (`ETL_DUMP_STATIONS`, ajouté ce matin ; sans `MEANDRE_DIAG_CPU=1` les diagnostics restent sur la carte, ce qui évite le ralentissement d'un transfert par jour). Montérégie, modèle provincial gelé, zéro époque, 2019-2024, 28 stations, 1 225 orages ; le débit de la même passe est sauvegardé à part. `chemin_rapide.py` rapporte à la pluie la production de surface, l'écoulement hypodermique et le débit latéral de la fenêtre de cinq jours.

Ce que la décomposition montre d'abord, à l'échelle de la journée : la production de surface moyenne d'été vaut 0,70 mm/j et l'écoulement hypodermique 0,06 mm/j, soit un rapport de douze. Le chemin rapide est donc la surface, pas la couche 2, ce qui écarte l'hypothèse d'un écoulement latéral porteur.

Deuxièmement, la part de surface monte avec la PLUIE de l'événement (0,053, 0,063, 0,078 puis 0,189 des classes 10-20, 20-35, 35-60 et plus de 60 mm) : c'est une signature d'excès d'infiltration, pas d'excès de saturation. L'humidité de surface theta1 ne fait que suivre la pluie antérieure (corrélation de rang 0,49 avec l'indice de précipitation antérieure, contre 0,46 pour theta2 et 0,02 pour le stock souterrain), et à une teneur médiane de 0,24 m³/m³, une porosité de l'ordre de 0,45 donne une saturation de 0,5, donc une part (theta/thetas)^8 de 0,004 : le terme de saturation ne produit presque rien.

Troisièmement, la corrélation entre la part de surface et theta1 vaut −0,44 si l'on met toutes les stations ensemble, mais +0,35 EN MOYENNE À L'INTÉRIEUR d'une station (quartiles 0,11 à 0,56). Le signe négatif global est un effet de composition : trois stations rendent presque toute leur pluie immédiatement (030314 : 1,00 ; 030921 : 0,93 ; 030919 : 0,63 de la pluie en production de surface), et ce sont les plus sèches. Le cache par tronçon les explique : leurs sols sont les plus fins du territoire, porosité de la couche 1 de 0,258, 0,265 et 0,325 contre une médiane de 0,496, et conductivité de 0,274, 0,149 et 0,065 m/jour contre une médiane de 0,248 (030919 est au cinquième centile). À 0,065 m/jour sur deux à huit heures d'orage, la capacité d'infiltration vaut 5 à 21 mm : tout le reste ruisselle, quel que soit l'état du sol. C'est de l'excès d'infiltration pur, et c'est ce qui domine le coefficient réuni.

CORRECTION (2026-10-23, 13 h 05, vérifiée sur le code et sur les diagnostics de la même passe). L'explication par l'excès d'infiltration est fausse. La colonne répartit la pluie du jour uniformément sur 24 heures (le ruissellement hortonien sous-journalier, `use_hortonian`, est éteint dans la recette, et le champ `rain_hours` n'est lu nulle part) : avec une conductivité de 0,065 m/j, il faudrait plus de 65 mm en un jour pour un excès d'infiltration. La saturation non plus : aux trois stations, la couche 1 n'atteint jamais la saturation en été (saturation médiane 0,25, 0,21 et 0,33, porosité réelle du tronçon et non supposée). La cause est l'OCCUPATION : ces trois bassins sont fortement urbanisés (fraction urbaine de la table territoriale à 8,1, 7,5 et 5,2 écarts-types au-dessus de la moyenne du territoire, contre −0,1 pour la Châteauguay), et la colonne envoie la pluie des surfaces imperméables directement en production de surface. L'effet de composition décrit plus haut reste juste ; son mécanisme est l'imperméabilisation. La porosité supposée de 0,45 et la saturation de 0,5 qui en découlait sont aussi à retirer : la saturation d'été médiane réelle aux 28 stations vaut 0,45 (quantiles 0,24 à 0,65), ce qui rend bien la part (theta/thetas)^8 très petite, mais pour les bassins non urbains seulement.

Quatrièmement, et c'est le point décisif : même en retirant les trois stations passantes, le coefficient d'écoulement simulé ne monte pas avec l'humidité antécédente (−0,08 sur 1 115 orages et 25 stations, la production de surface montant à +0,14) alors que l'observé monte (+0,32 en Montérégie). La production répond donc un peu à l'état du sol, mais le DÉBIT ne suit pas.

Lecture. Le défaut n'est ni dans la loi d'aires saturées ni dans le choix de la variable d'état : il est entre la production et la rivière. L'eau rapide produite sur le versant est étalée et retenue avant d'arriver au tronçon, ce que disaient déjà les deux mesures de la candidate 5.6 (propagation juste en retard mais trop peu atténuée dans le réseau ; réponse du versant démarrant le jour même et plus étalée que l'observé). Les trois candidates convergent donc sur le même endroit du modèle : le transfert versant vers tronçon, et non le partage ruissellement-infiltration.

Suite recommandée : traiter la candidate 5.6 comme la priorité, en vérifiant d'abord ce que sont les trois nœuds passants (une erreur d'aire locale ou de tarage y produirait ce comportement), puis en éprouvant une forme de réponse du versant à deux paramètres (loi gamma, à somme un) qui retarde d'un jour et concentre le pic.

## 0.4 Niveaux aux jauges du CEHQ (2026-10-23, matin)

Les fichiers de niveaux `NNNNNN_N.txt` existent au même dépôt que les débits pour une partie des stations seulement. Sur les 170 stations du modèle, 45 en ont un ; téléchargés dans `D:/meandre-data/sources/cehq-niveaux/` (27 Mo, journal et inventaire à côté). Ce sont des niveaux à la jauge, en mètres locaux (constante vers le géodésique donnée dans l'en-tête, sans valeur légale) : ils jugent la géométrie et le tarage au droit de la station, pas les lacs. Aucune table ni terme de perte n'est créé ; ingestion dans les bases à décider avec Essi.

Complément 5.1 (09 h 05). La forme physique « température plus rayonnement » existe déjà dans la colonne : `melt_mode = "eti"` (hydrotel_clone/snow.py, coefficients tf et srf de Pellicciotti, rayonnement réel lu au canal 6 du forçage), avec les forçages `forcing-<reg>-swin.nc` présents pour l'Abitibi, la Gaspésie, l'Outaouais, le Saguenay et le Saint-Laurent nord-ouest, et le contenu de froid du manteau y est déjà un état (chaleur_n). Le registre l'a jugée deux fois : sur la MASSE (août), meilleure formulation disponible sur les bassins au forçage fiable (Saguenay 0,81 à 0,89 du réseau de décembre à mai contre 0,60 à 0,77 pour les degrés-jours) ; sur le DÉBIT (juillet, SLSO, forçage krigé, pipeline d'avant les corrections du sol), rejetée (0,551 contre 0,651). Le résultat d'aujourd'hui contre l'ablation mesurée va dans le sens du verdict d'août. Mais le pilote fixe `melt_mode="degree_day"` en dur depuis le lot D du nettoyage (option « fonte ETI » retirée faute de fichier qui la pose) : l'épreuve 3.4 de la candidate 5.1 demande une clé déclarée `[snow] melt_mode`, puis une passe avant à poids gelés sur le Saguenay et l'Outaouais, avec la même comparaison CanSWE par mois. Demande transmise à la session de nettoyage.

Réponse de la session de nettoyage (09 h 10) : le retrait de l'option ETI au lot D était une décision d'Essi, à rouvrir avec son accord et en changement séparé après le commit en cours. Contrainte technique : le mode ETI lisait le rayonnement au canal 6 du forçage, que le modèle gelé emploie pour la demande évaporative apprise ; il faudrait un septième canal (le rayonnement net R_n du CaSR brut est déjà dans le fichier de forçage, variable 4 sur 6, et pourrait servir sans fichier -swin). Le bloc retiré se relit par `git show dcc71db:.runs/quebec/etl_run.py` (ETL_ETI) ; build_swin_region.py a été supprimé ce matin (commit 14cb8c2), les fichiers -swin construits restent. Décision à prendre par Essi : rouvrir la fonte au rayonnement par une clé déclarée et un canal dédié, ou en rester aux degrés-jours.

## 5.3, suite : diagnostics au banc de la Châteauguay (030905, 214 tronçons), 2011-2024 (2026-10-23, 09 h 20)

Banc de sous-bassin sur le processeur, point de reprise provincial chargé, forçage CaSR brut, mode bilan mensuel avec séries journalières (`series-mont-030905.parquet`). ATTENTION : cette première passe garde le réglage par défaut du banc, qui impose le calage de sol d'Hydrotel sauf K_sat (loi des ancrages) ; une seconde passe avec le sol du champ (`--sol libre`) est en cours pour juger le modèle provincial lui-même.

Ce que la première passe dit déjà. La saturation de la couche 1 varie bien en été (quantiles 5 et 95 % : 0,69 et 0,99 ; elle suit l'indice de précipitation antérieure à 0,73 de corrélation de rang, autant que le débit observé, 0,75). La production de surface répond fortement à cette saturation : par quartile de saturation la veille de l'orage, production sur pluie 0,005, 0,040, 0,154, 0,469 (la puissance 8 fait son travail). Mais seulement 35 % de cette production atteint la rivière dans les cinq jours (médiane du rapport hausse de débit sur production), et le débit simulé répond plus que l'observé à la saturation (corrélation de rang 0,51 contre 0,14) et moins à la pluie (0,48 contre 0,57). Composite des 92 orages : la production de surface du jour vaut 5,7 mm/j, la hausse du débit simulé 0,6 mm/j, la hausse observée 0,2 mm/j le jour même puis 0,9 le lendemain.

Lecture provisoire, à confirmer par la passe au sol libre : sur ce bassin agricole, le défaut n'est pas une humidité de surface figée mais un chemin entre la production et la rivière qui étale et retient (versant, milieux humides, routage), et une loi de part ruisselée qui réagit trop à la saturation et pas assez à la pluie. Cela pointe vers la forme de la réponse rapide (candidate 5.6) plutôt que vers la loi des aires saturées seule.

Seconde passe, sol du champ (`--sol libre`, 09 h 30). Le chargeur signale encore deux écarts avec la fiche du point de reprise : ETP Linacre au lieu de Penman et seuil pluie-neige à −3,07 °C au lieu de −0,8 (défauts du banc, pas du pilote). Le banc ne reproduit donc PAS le modèle provincial, et ses chiffres ne valent que pour la mécanique interne. Avec le sol du champ : production de surface 0,044 de la pluie (0,111 avec le sol d'Hydrotel), dont 44 % atteint la rivière en cinq jours ; coefficient simulé 0,020 contre 0,019 observé ; par quartile de saturation de la couche 1 la veille, simulé 0,019, 0,016, 0,020, 0,049 et observé 0,015, 0,018, 0,023, 0,022 ; sur ce seul bassin, l'observé ne dépend pas non plus de l'indice de précipitation antérieure (corrélation de rang 0,03), alors que sur les 23 stations de Montérégie il en dépend à +0,32. Conclusion pour 5.3 : le banc ne tranche pas, parce qu'il ne porte pas la recette du pilote et qu'une station seule ne porte pas le signal ; la mesure robuste reste celle des 144 stations (réponse simulée plate contre l'humidité antécédente et la taille de la pluie). Le diagnostic décisif passe par le pilote lui-même (diagnostics par tronçon, carte) et par la sonde de gradient de l'exposant, à faire quand la carte est libre.

## Calculs déportés sur Ubuntu : la référence de la machine (2026-10-23, 10 h 30)

Le poste de travail et le poste Ubuntu ne donnent pas les mêmes nombres pour le même modèle, et il faut le savoir avant de comparer quoi que ce soit entre les deux. Sur l'Outaouais, modèle provincial gelé, zéro époque, 2019-2024, 16 stations : KGE médian 0,7096 et perte 0,8841 sur Windows, 0,6874 et 0,9316 sur Ubuntu. Les entrées sont pourtant identiques au bit près : le forçage CaSR brut a le même empreinte sur les deux machines, la base de bassin aussi, et le point de reprise est le même fichier. Deux passes identiques sur Ubuntu redonnent exactement 0,6874 et 0,9316, donc la machine est reproductible ; l'écart vient des noyaux de calcul (compilation du sol et sous-pas adaptatifs), pas de la configuration. Les séries simulées diffèrent de 4 à 8 % d'écart quadratique relatif dès 2019 et le biais reste sous 1,5 % : une différence de noyaux, pas une dérive.

Vérification (2026-10-23, 13 h 40). Les fichiers de physique des deux machines sont identiques au caractère près (seules les fins de ligne diffèrent). Le poste Windows avait tourné avant le commit du lot E du nettoyage (100 617 paramètres) et Ubuntu après (99 136) ; la même passe relancée sur Ubuntu avec le code d'avant le lot E redonne exactement 0,6874 et 0,9316. Le lot E ne change donc pas les débits, et l'écart entre les machines est bien un écart de machine. Reste ouvert : un écart de 0,02 de KGE et de 4 à 8 % d'écart quadratique pour le même code est grand pour des différences d'arrondi, ce qui suggère une amplification numérique quelque part (schéma du sol, sous-pas, compilation) ; à mesurer un jour par une passe sur processeur des deux côtés.

Conséquence pratique : toute comparaison de variantes doit se faire sur une même machine. Le poste Ubuntu est donc la référence des essais déportés, et les chiffres du poste de travail ne se mélangent pas avec les siens.

## 5.6 Routage : épreuve 3.1, propagation entre stations emboîtées (2026-10-23, 09 h 35)

`propagation.py`, 58 paires de stations emboîtées (aire amont sous 80 % de l'aval), 2019-2024, jours communs, modèle provincial gelé. Retard amont-aval du simulé juste : 0,19 jour contre 0,17 observé, 46 paires sur 58 au même jour. Corrélation croisée au meilleur décalage 0,87 simulée contre 0,82 observée. Mais l'ATTÉNUATION le long du réseau est trop faible : la hausse spécifique des vingt plus fortes crues à l'aval vaut 0,49 fois celle de l'amont dans l'observé et 0,73 dans le simulé ; l'écart grandit avec la distance (150 tronçons et plus : 0,41 observé, 0,70 simulé) et il est le plus fort en Montérégie (0,57 contre 0,80) et au Saint-Laurent sud-ouest (0,46 contre 0,83) ; l'Outaouais est l'exception (0,33 contre 0,26, les lacs).

## 5.6, suite : forme de la réponse rapide, observée contre simulée (2026-10-23, 10 h 40)

`forme_reponse.py`, 4 699 orages observés et 4 919 simulés, 129 stations, 2019-2024, aucun calcul nouveau : la forme est lue sur les séries déjà enregistrées. Pour chaque orage, hausse journalière du débit spécifique au-dessus du débit de la veille, sur sept jours, et trois grandeurs : jour du maximum, part de la hausse des sept jours déjà acquise le jour de l'orage, le lendemain et le surlendemain, et hausse restante le septième jour.

| Médiane par station | observé | simulé |
|---|---|---|
| jour du maximum | 2 | 2 |
| part de la hausse acquise le jour de l'orage | 0,008 | 0,106 |
| part acquise le lendemain | 0,263 | 0,343 |
| part acquise au surlendemain | 0,519 | 0,534 |
| part de la hausse concentrée sur le seul jour du maximum | 0,365 | 0,296 |
| hausse restante au septième jour | 0,063 | 0,069 |

Lecture. La réponse observée est presque nulle le jour de l'orage (0,8 % de la hausse, avec des pluies de bassin d'au moins 10 mm), puis elle monte d'un coup le lendemain ; la simulée livre déjà 10,6 % le jour même, treize fois plus, et se rattrape ensuite. Le ressuage final est le même des deux côtés, donc ce n'est pas la queue qui diffère, c'est le DÉBUT : le modèle répond trop tôt et trop fort, puis les deux courbes se rejoignent au surlendemain. Par territoire, le maximum observé tombe toujours au deuxième jour ; le simulé tombe au premier en Montérégie et en Gaspésie, au deuxième au Saguenay, au Saint-Laurent sud-ouest et nord-ouest, et au quatrième en Outaouais (les lacs) alors que l'observé reste au deuxième.

L'écart n'est pas une moyenne portée par quelques bassins : sur les 129 stations, la part du jour de l'orage est plus forte dans le simulé que dans l'observé à 113 stations et plus faible à 15, et la concentration sur le jour du maximum est plus faible dans le simulé à 115 stations et plus forte à 14.

Ce que cela ajoute aux deux mesures précédentes. La propagation entre stations emboîtées est au bon jour et n'atténue pas assez, et l'hydrogramme aux stations est trop amorti. Cette troisième mesure montre où le défaut se loge : la crue démarre trop tôt, ce qui étale le sommet sur deux jours au lieu de le concentrer sur un, et abaisse le pic. La candidate 5.6 doit donc viser un retard et une concentration de la réponse du versant, pas seulement une atténuation plus forte.

Lecture, avec les résultats précédents. Le modèle propage au bon jour et n'atténue pas assez dans le réseau ; or ses hydrogrammes aux stations sont trop amortis. Les deux ne tiennent ensemble que si l'amortissement est en AMONT du réseau, à l'échelle du versant et des petits tronçons : réponse rapide trop étalée dès la production (au banc de la Châteauguay, 35 à 44 % de la production de surface atteint la rivière en cinq jours), puis un réseau qui transmet presque sans atténuer. Le temps de transfert de Muskingum (médiane 22 h sur la Châteauguay, bornes 4 à 48 h) n'est pas le levier : le retard est juste. La candidate 5.6 vise donc la forme de la réponse du versant (production vers tronçon), pas le canal ; une célérité apprise en fonction du débit n'est pas justifiée par cette mesure.

Correctif du 2026-10-23, après-midi. La part de hausse au jour de l'orage, lue ici comme médiane par orage (0,008 observée contre 0,106 simulée), est dominée par les orages à pluie étalée et par la petite taille des réponses ; le profil moyen donne 0,064 des deux côtés, et sur les orages à pluie d'un seul jour le modèle démarre plus TARD que l'observé. Voir la section suivante : le défaut n'est pas un démarrage trop précoce, c'est une forme trop peu variable d'un orage à l'autre.

## 5.6, épreuve 3.3 : noyau de réponse pluie-débit, forme commune et transfert (2026-10-23, après-midi)

`noyau_versant.py`, 2 151 orages de juin à septembre sur 144 stations et 14 territoires, 2019-2024, débits du modèle provincial gelé, aucune simulation nouvelle. La forme est mesurée de deux façons : profil moyen de la hausse sur sept jours, normalisé par orage ; puis déconvolution pluie-débit, chaque orage gardant une amplitude libre et le noyau commun étant positif et de somme un.

Profil moyen, part de chaque jour, et orages à pluie du jour 0 majoritaire (223 sur 2 151) :

| jour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | retard moyen | pic |
|---|---|---|---|---|---|---|---|---|---|
| pluie | 0,448 | 0,166 | 0,095 | 0,076 | 0,079 | 0,074 | 0,063 | 1,65 j | 0 |
| hausse observée | 0,064 | 0,209 | 0,188 | 0,155 | 0,140 | 0,128 | 0,115 | 2,95 j | 1 |
| hausse simulée | 0,064 | 0,147 | 0,174 | 0,172 | 0,159 | 0,146 | 0,137 | 3,20 j | 2 |
| observée, pluie d'un jour | 0,164 | 0,393 | 0,188 | 0,109 | 0,069 | 0,044 | 0,033 | 1,79 j | 1 |
| simulée, pluie d'un jour | 0,129 | 0,248 | 0,219 | 0,160 | 0,112 | 0,077 | 0,055 | 2,33 j | 1 |

Dispersion de la forme, 36 stations d'au moins vingt orages, distance L1 :

| mesure | observé | simulé |
|---|---|---|
| d'une station à la forme commune | 0,222 | 0,156 |
| d'un orage à la forme commune | 0,608 | 0,466 |
| d'un orage à la moyenne de sa station | 0,593 | |
| retard moyen du profil de station | 2,84 j | 3,14 j |

La forme commune observée et la simulée sont distantes de 0,146, soit moins que la dispersion entre stations (0,222) et quatre fois moins que la dispersion entre orages (0,608). L'observé est plus rapide que le simulé à 27 stations sur 36.

Transfert du profil commun, distance L1 à la forme observée des lieux retirés :

| retrait | commun observé | commun simulé | profil plat |
|---|---|---|---|
| stations, cinq plis | 0,221 | 0,240 | 0,347 |
| années, 2019-2021 vers 2022-2024 | 0,212 | | |
| Montérégie, 5 stations | 0,152 | 0,229 | |
| Outaouais, 4 stations | 0,246 | 0,309 | |
| Saguenay, 2 stations | 0,367 | 0,243 | |
| Saint-Laurent nord-ouest, 13 stations | 0,197 | 0,224 | |
| Saint-Laurent sud-ouest, 12 stations | 0,229 | 0,250 | |

Déconvolution, noyau commun :

| noyau | 0 | 1 | 2 | 3 | 4 | 5 | 6 | retard |
|---|---|---|---|---|---|---|---|---|
| observé | 0,174 | 0,437 | 0,165 | 0,098 | 0,064 | 0,037 | 0,026 | 1,65 j |
| simulé | 0,129 | 0,268 | 0,231 | 0,158 | 0,104 | 0,068 | 0,041 | 2,21 j |

Erreur de forme sur les orages retirés, médiane : stations 0,591 pour le noyau observé, 0,473 pour le simulé, 0,632 pour un simple retard d'un jour ; années 0,444, 0,469 et 0,622 ; un territoire à la fois, observé contre simulé, Gaspésie 0,501 contre 0,387, Montérégie 0,476 contre 0,557, Outaouais 0,474 contre 0,357, Saguenay 0,446 contre 0,398, SLNO 0,588 contre 0,477, SLSO 0,479 contre 0,541.

Lecture. Trois faits.

1. Le modèle n'est pas trop précoce en moyenne : il est plus LENT que l'observé (retard moyen 3,20 contre 2,95 jours, pic au jour 2 contre jour 1) et plus ÉTALÉ, et l'écart est encore plus net sur les orages à pluie d'un seul jour (2,33 contre 1,79). La lecture du matin venait de la médiane par orage de la part au jour de l'orage, tirée vers le bas par les petits orages à pluie étalée ; le profil moyen, qui pèse chaque orage pareil, donne 0,064 de part au jour 0 des deux côtés. Les deux statistiques mesurent des choses différentes : la distribution est très asymétrique.
2. La forme observée est bien plus variable que la simulée. D'un orage à l'autre, la distance à la forme commune vaut 0,608 dans l'observé et 0,466 dans le simulé, et la dispersion interne d'une station (0,593) est du même ordre : la réponse dépend de l'état de l'orage, pas seulement du lieu. Le modèle, lui, produit presque toujours la même forme. C'est le défaut, et il rejoint la mesure des orages d'été (R394 : le coefficient d'écoulement observé croît de 0,025 à 0,065 avec l'humidité antécédente, le simulé reste plat).
3. Aucun noyau linéaire commun ne transfère. La déconvolution laisse 46 à 59 % d'erreur de forme, le noyau ajusté sur l'observé perd contre celui ajusté sur le simulé aux stations retirées (0,591 contre 0,473) et le classement s'inverse selon le retrait ; un simple retard d'un jour est pire que les deux. Le profil commun observé transfère un peu mieux que le simulé (0,221 contre 0,240), mais le gain hors échantillon (8 %) est faible en regard de la dispersion entre orages, et il s'inverse au Saguenay (deux stations).

La dépendance à l'état, ensuite. Le retard de la forme croît avec la taille de la pluie dans l'observé (+0,32 de corrélation de rang ; à l'intérieur d'une station +0,30, pente +1,8 jour par décade de pluie) et plus encore dans le simulé (+0,48 et +2,6 jours). Sur les seuls orages à pluie d'un jour, où la durée de pluie ne confond plus, l'effet reste faible des deux côtés (+0,18 et +0,25). L'humidité antécédente, elle, ne déplace presque pas la forme simulée (−0,04 sur ces orages) mais avance la forme observée (−0,25) : sol plus humide, réponse plus rapide et plus concentrée. C'est la seule dépendance à l'état que le modèle ne porte pas, et elle est faible ; elle ne suffit pas à expliquer une dispersion de 0,61.

Verdict : candidate 5.6 arrêtée à l'épreuve 3.3. Il n'y a pas de noyau de versant commun à apprendre : la différence systématique de forme entre le modèle et l'observé (0,146) est plus petite que la variabilité propre à chaque station et quatre fois plus petite que celle d'un orage à l'autre. Le défaut est donc une absence de dépendance à l'état dans le TRANSFERT versant-rivière, et non dans la loi de partage : l'épreuve 3.4 de 5.3 a déjà montré que l'exposant des aires saturées ne porte pas cette dépendance, et un noyau commun ne la porte pas davantage. Trois formes sont écartées : un retard fixe d'un jour (erreur 0,632), un hydrogramme unitaire à deux paramètres commun (le noyau ajusté sur l'observé perd contre celui ajusté sur le simulé aux stations retirées, 0,591 contre 0,473) et une célérité apprise en fonction du débit (la propagation entre stations est déjà au bon jour, R395). La forme recommandée plus haut, une gamma à deux paramètres qui retarde d'un jour, doit être corrigée : il faut CONCENTRER sans retarder, puisque le modèle est déjà plus lent que l'observé (retard moyen 3,20 contre 2,95 jours). Ce qui reste ouvert est un transfert dont la forme dépend de l'état de l'orage, à écrire comme une fonction de l'intensité de la pluie, de l'humidité antécédente et du stock disponible.

Ce que seule une simulation dirait : si un transfert dépendant de l'état, une fois posé dans la colonne, reproduit la dispersion observée de la forme d'un orage à l'autre, et si l'écart résiduel de forme (0,146) compte encore pour le débit après routage. L'épreuve se fait en passe avant, à poids gelés, sur la Montérégie et l'Outaouais, avant tout entraînement.

## 5.4, épreuve 3.3 : forme physique de la porte de gel contre la réponse observée (2026-10-23, verdict consigné à 13 h 15)

`porte_gele_forme.py` (écrit par la session parallèle, relancé et relu ici), sur les 9 844 pluies de `sol_gele.py`. Formes communes ajustées aux médianes par cellule d'indice de gel et de manteau, écart moyen en logarithme.

| Forme | Ajustée sur tout | Stations retirées (médiane des 5 plis) | Années retirées (apprise jusqu'à 2014, éprouvée 2015-2024) | Territoires retirés où elle bat la rampe seule |
|---|---|---|---|---|
| rampe saturante (forme actuelle de la colonne) | 0,279 | 0,226 | 0,206 | référence |
| puissance | 0,305 | 0,263 | 0,265 | |
| rampe et facteur de manteau | 0,096 | 0,134 | 0,234 | 4 sur 6 |
| puissance et facteur de manteau | 0,105 | 0,122 | 0,222 | 4 sur 6 |

Paramètres de la rampe avec manteau, ajustés sur tout : amplitude 0,32, saturation à 41 °C·jour, facteur de manteau 1,21, demi-saturation du manteau à 9,2 mm. Aux territoires retirés, le facteur de manteau gagne en Gaspésie (0,44 à 0,33), Montérégie (0,38 à 0,33), Outaouais (0,33 à 0,23) et Saint-Laurent sud-ouest (0,28 à 0,08), et perd au Saguenay (0,51 à 0,54) et au Saint-Laurent nord-ouest (0,25 à 0,30).

Lecture. Ce qui porte la réponse, c'est la combinaison du gel et du manteau, pas la forme de la courbe de gel : sans le manteau, aucune forme ne descend sous 0,26 aux stations retirées ; avec lui, toutes passent à 0,12 ou 0,13. Mais le gain ne tient pas dans le TEMPS : aux années retirées, la rampe seule (0,206) fait mieux que les formes avec manteau (0,222 à 0,252). Et le « manteau » mesuré ici est un stock de degrés-jours calculé sur le forçage, pas le manteau du modèle ; les paramètres ajustés ne se transposent donc pas tels quels dans la colonne.

Verdict : candidate 5.4 ouverte, forme non retenue. Le défaut du modèle (saturation et erreur de territoire, épreuve 3.1) est établi ; la direction du remède (inclure le manteau dans la porte) est indiquée mais pas stable entre périodes. Prochaine épreuve, avant toute sonde ou passe avant : réajuster la même comparaison contre l'état SIMULÉ (fraction gelée de Rankinen et équivalent en eau de la neige du modèle aux stations), pour savoir si c'est la porte ou l'état qu'elle lit qui est en défaut.

## 5.4, état simulé contre porte : Montérégie (2026-10-23, 14 h 10)

`gel_etat_simule.py`, diagnostics journaliers du pilote aux tronçons des stations (`ETL_DUMP_STATIONS`, profondeur de gel de Rankinen et neige simulées), modèle provincial gelé, zéro époque, 2001-2024, Ubuntu. 3 173 pluies de novembre-décembre (≥ 5 mm, T > 0,5 °C), 27 stations. Limite : l'état est celui du tronçon exutoire, pas une moyenne du bassin.

| Gel simulé au tronçon la veille (cm) | Pluies | Réponse relative observée | Réponse relative simulée | Indice de gel atmosphérique médian (°C·jour) |
|---|---|---|---|---|
| 0 | 2 238 | 1,20 | 1,34 | 3 |
| 5 à 15 | 456 | 1,36 | 1,48 | 21 |
| 15 à 30 | 365 | 2,03 | 4,95 | 64 |
| 30 et plus | 102 | 2,26 | 7,94 | 149 |

Ce que la réponse observée suit le mieux (corrélation de rang) : l'indice de gel tiré de la température de l'air (+0,27) et la neige simulée (+0,26), mieux que le gel simulé (+0,16). Dans 38 % des pluies où l'indice dépasse 50 °C·jour, le tronçon simulé n'a aucun gel.

Lecture. La saturation de la porte vue en moyenne sur 2019-2024 cachait deux erreurs de sens contraire. D'une part l'ÉTAT : le gel simulé manque souvent quand l'air a gelé, et ordonne moins bien la réponse observée que l'indice atmosphérique. D'autre part la PORTE : quand le gel simulé dépasse 15 cm, la réponse simulée vaut 5 à 8 fois la réponse sur sol non gelé, contre 2,0 à 2,3 observée, soit une imperméabilisation trop forte. Une forme apprise de la porte ne corrigerait que la seconde. À confirmer sur l'Outaouais et le Saint-Laurent sud-ouest (en cours, diagnostics en mémoire centrale après un manque de mémoire graphique).

## 5.4, état simulé contre porte : trois territoires (2026-10-23, 13 h 20)

Même analyse sur la Montérégie, l'Outaouais et le Saint-Laurent sud-ouest : 7 072 pluies de novembre-décembre, 71 stations, 2001-2024 (Ubuntu ; l'Outaouais et le Saint-Laurent sud-ouest relancés avec les diagnostics en mémoire centrale après un manque de mémoire graphique).

| Gel simulé (cm) et neige simulée la veille | Pluies | Réponse relative observée | Simulée |
|---|---|---|---|
| aucun gel, moins de 5 mm de neige | 4 387 | 1,11 | 1,24 |
| aucun gel, 5 à 30 mm de neige | 148 | 2,57 | 3,80 |
| 15 à 30 cm, moins de 5 mm de neige | 394 | 1,44 | 3,88 |
| 30 cm et plus, moins de 5 mm de neige | 151 | 2,23 | 3,94 |
| 15 à 30 cm, plus de 30 mm de neige | 191 | 2,81 | 1,57 |

Par territoire, pluies à gel simulé de 5 cm et plus, réponse observée et simulée : Montérégie 1,75 et 2,88 ; Outaouais 1,85 et 2,00 ; Saint-Laurent sud-ouest 1,47 et 1,40.

Ce qui tient. La PORTE est en défaut dans deux situations précises, de sens contraire : sol gelé sans neige, le modèle ruisselle deux à trois fois trop (surtout en Montérégie) ; sol gelé sous un manteau épais, il ruisselle deux fois trop peu, ce qui est la règle codée (la porte ne s'applique que sous 10 mm de neige). Ailleurs, l'Outaouais et le Saint-Laurent sud-ouest sont justes en moyenne. La pluie sur neige sans gel est aussi trop forte (3,80 contre 2,57).

Ce qui ne tient pas, contrairement à ma lecture de la Montérégie seule. J'avais conclu que l'état de gel simulé était en cause (gel simulé nul dans 34 % des pluies à indice atmosphérique supérieur à 50 °C·jour, et réponse observée mieux ordonnée par cet indice que par le gel simulé). Mais l'indice cumule les degrés-jours négatifs sans redescendre après un redoux, quand le gel simulé dégèle ; la neige simulée est la même que le gel soit présent ou non (médiane 9,9 contre 11,6 mm) ; et la réponse, observée comme simulée, est plus forte quand le gel simulé manque (2,42 et 2,97 contre 1,90 et 2,27), ce qui ressemble à des dégels avec fonte. L'état n'est donc pas démontré faux.

Verdict pour 5.4 : le défaut est dans la porte, et il a une structure physique simple : trop forte sans manteau, coupée sous le manteau. Une forme apprise n'est pas nécessaire ; la forme à éprouver en passe avant est une porte continue en fonction de la neige (au lieu de la coupure à 10 mm) et d'amplitude plus faible, d'abord sur la Montérégie où l'excès est le plus net. Limite : état au tronçon exutoire seulement.

## 5.4, épreuve 3.4 : amplitude de la porte de gel en passe avant (2026-10-23, soir)

Réglage déclaré `column.frozen_gate_amplitude` (alias `MEANDRE_GEL_AMPLITUDE`, défaut 1 = porte d'origine) : capacité d'infiltration ks1 (1 − a · fraction gelée) et, depuis la seconde série, même facteur sur le frein du drainage de la couche 1. Modèle provincial gelé, zéro époque, Montérégie, 2001-2024, Ubuntu ; `porte-gel-amplitude.py`. Réponse relative aux pluies de novembre-décembre (rapportée à la réponse de la station sur sol non gelé), par classe de gel et de neige SIMULÉS au tronçon de la station.

| Variante | gel ≥ 15 cm, sans neige (observé 1,77) | gel ≥ 15 cm, neige > 30 mm (observé 3,06) | gel 5-15 cm, sans neige (observé 1,17) | perte | terme d'étiage | KGE médian |
|---|---|---|---|---|---|---|
| amplitude 1 (porte actuelle) | 6,63 | 2,66 | 1,38 | 1,4107 | 0,7227 | 0,5527 |
| 0,5 sur l'infiltration | 2,97 | 3,15 | 1,36 | 1,4090 | 0,7232 | 0,5531 |
| 0,25 sur l'infiltration | 2,95 | 3,15 | 1,36 | 1,4092 | 0,7233 | 0,5529 |
| 0,5 sur l'infiltration, sans coupure sous la neige | 3,96 | 4,26 | 1,44 | 1,4302 | 0,7539 | 0,5473 |
| 0,5 sur l'infiltration et le drainage | 1,83 | 2,76 | 1,50 | 1,4158 | 0,7182 | 0,5458 |
| 0,25 sur l'infiltration et le drainage | 1,70 | 2,75 | 1,54 | 1,4185 | 0,7153 | 0,5417 |

Lecture. L'excès de ruissellement sur sol gelé sans neige a deux causes qui s'additionnent : la porte d'infiltration ferme le sol quand le gel atteint l'épaisseur de la couche 1, et le gel freine en même temps le drainage de cette couche, qui se remplit puis ruisselle tout. Réduire la seule porte d'infiltration plafonne à 2,95 (les amplitudes 0,5 et 0,25 donnent la même chose) ; réduire aussi le frein du drainage ramène la réponse à l'observé (1,83 et 1,70 contre 1,77). Retirer la coupure sous la neige reste mauvais (perte 1,4302), conforme au verdict de septembre. À poids gelés, l'effet sur la perte est petit et de signe mixte : terme d'étiage −0,6 % (0,7227 à 0,7182), perte totale +0,4 % et KGE −0,007 ; les autres paramètres ont été appris avec l'ancienne porte.

Ce que seul un entraînement dirait : si l'optimiseur, la porte corrigée, retrouve une perte au moins égale et garde l'étiage meilleur. Avant cela, vérification en passe avant sur l'Outaouais (où la porte sous la neige avait détruit l'hiver en septembre), en cours.

Outaouais, même épreuve, amplitude 1 contre 0,5 sur l'infiltration et le drainage (2026-10-23, soir). Perte 2001-2024 0,9909 contre 0,9952 (+0,4 %), terme d'étiage 0,4604 contre 0,4593 (−0,2 %), KGE médian 0,630 contre 0,617. L'hiver n'est pas détruit (contrairement à la porte sous la neige de septembre) : les KGE par année bougent de moins de 0,01. Réponse relative aux pluies d'automne, gel ≥ 15 cm sans neige : 3,29 contre 1,33, pour 1,94 observé (la correction dépasse : trop faible désormais) ; gel ≥ 15 cm sous plus de 30 mm de neige : 1,58 contre 1,16, pour 2,17 observé (pire). Réserve : le nombre de pluies retenues diffère entre variantes (561 et 663), la référence sur sol non gelé de chaque station dépendant du simulé.

Verdict pour 5.4 : le mécanisme est établi (porte d'infiltration plus frein du drainage), et l'amplitude 0,5 rend la Montérégie juste ; mais elle sous-corrige l'Outaouais, et sur les deux territoires l'effet sur la perte à poids gelés est inférieur à 1 % et de signe mixte (étiage meilleur, KGE et perte un peu moins bons). Pas de changement de recette, pas de réentraînement : le gain attendu pour l'étiage est trop petit pour son coût. Le réglage reste déclaré, défaut 1, pour une épreuve future (amplitude intermédiaire ou porte dépendante de l'état du sol).

## Volume d'été du sud : forçage écarté (2026-10-24, nuit)

Mandat d'Essi : itérer 36 heures sur le volume d'été du sud agricole, principal défaut de meandre pour l'étiage.

Localisation (`ete-sud.py`, modèle provincial gelé, 2019-2024, juin-septembre, médiane des étés et des stations, simulé sur observé ; Hydrotel = médiane des six membres) :

| Territoire | Stations | Ligne de base meandre / Hydrotel | Événements meandre / Hydrotel | Moyenne meandre / Hydrotel |
|---|---|---|---|---|
| Saint-Laurent sud-ouest | 38 | 1,07 / 1,12 | 0,60 / 0,97 | 0,71 / 1,04 |
| Saint-Laurent nord-ouest | 32 | 0,98 / 0,73 | 0,90 / 0,90 | 0,94 / 0,81 |
| Montérégie | 28 | 0,96 / 1,17 | 0,82 / 0,95 | 0,85 / 0,97 |
| Saguenay | 22 | 1,07 / 0,72 | 0,79 / 0,81 | 0,92 / 0,79 |
| Outaouais | 16 | 1,00 / 0,84 | 0,93 / 1,03 | 0,99 / 1,02 |

La ligne de base de meandre est juste partout (mieux qu'Hydrotel, qui la sous-estime au nord) ; le défaut est le volume des ÉVÉNEMENTS, surtout au Saint-Laurent sud-ouest. La conductivité de surface apprise n'explique pas ce territoire : elle y est basse (médiane 0,22 m/j), alors qu'elle est collée à sa borne haute d'environ 1,09 m/j en Montérégie agricole, au Saguenay et en Gaspésie.

Forçage (`ete-pluie-goens.py`, `ete-pluie-intensite.py`) : sur le bassin amont de chaque station, avec les mêmes poids, le cumul de juin à septembre 2019-2024 de CaSR vaut 1,01 à 1,05 fois celui de GOENS, la météo de l'ensemble Hydrotel (1,03 au Saint-Laurent sud-ouest, 1,04 en Montérégie) ; la pluie journalière d'été est corrélée à 0,94-0,96 au même jour, avec le même nombre de jours de plus de 20 mm (5,5 contre 5,1 par été au Saint-Laurent sud-ouest) et la même part de pluie tombée ces jours-là (0,38 contre 0,36). Hors sujet mais noté : l'hiver, CaSR vaut 1,2 à 1,4 fois GOENS.

Verdict : à pluie d'été égale en volume et en intensité, Hydrotel rend 97 % du volume des événements au Saint-Laurent sud-ouest et meandre 60 % ; le défaut est dans la structure ou les paramètres de meandre, pas dans le forçage.

## Volume d'été du sud : sol, évaporation, saisons (2026-10-24, matin)

Sol de surface (`ete-parametres-sol.py`). Au Saint-Laurent sud-ouest, le champ de meandre a appris une porosité de 0,60 aux couches 1 et 2 sur tous les tronçons (borne haute ; calage d'Hydrotel 0,43), une capacité au champ de surface de 0,51 (Hydrotel 0,27), et des épaisseurs de couches 2 et 3 collées à leurs bornes basses (0,30 et 0,50 m ; Hydrotel 0,16 et 2,65 m). Même tableau en Montérégie et en Outaouais. Passes à poids gelés sur le Saint-Laurent sud-ouest, 2019-2024, multiplicateurs fixes sur le champ (`ETL_MULT_FIXE`) :

| Variante | Perte | Terme d'étiage | KGE médian | Été, base | Été, événements | Été, moyenne |
|---|---|---|---|---|---|---|
| témoin | 1,1202 | 0,4648 | 0,629 | 1,22 | 0,56 | 0,73 |
| capacité au champ de surface × 0,55 | 1,1600 | 0,4778 | 0,610 | 1,11 | 0,50 | 0,67 |
| idem, porosités 1 et 2 × 0,75 | 1,1183 | 0,4618 | 0,625 | 1,09 | 0,52 | 0,68 |

Le stockage du sol n'est pas le levier des événements à poids gelés ; une capacité au champ plus basse augmente l'évaporation et retire encore du débit d'été.

Bilan de la colonne en été (diagnostics aux tronçons des stations, 2011-2024, juin-septembre) : pluie 3,7 mm/j, ETR 3,06 (84 %), production 0,66 (21 %), toute transmise au tronçon. ETR simulée sur MOD16 au même tronçon, médiane des stations : Saint-Laurent sud-ouest mai 1,20, juin 1,11, juillet 0,97, août 0,94 ; Montérégie 1,45, 1,36, 0,91, 0,80 ; Outaouais 1,22, 1,06, 1,00, 0,94. L'évaporation d'été n'est pas trop forte ; celle de mai-juin l'est, surtout en Montérégie.

Saisons (débit simulé sur observé par mois, 2020-2024, médiane des stations) : au Saint-Laurent sud-ouest, meandre 1,08, 1,09, 1,20 de mars à mai puis 0,67, 0,64, 0,66, 0,75 de juin à septembre ; Hydrotel 0,85, 0,78, 0,88 puis 0,87, 1,00, 1,12, 1,10. Montérégie : meandre 0,98, 1,11, 0,93 puis 0,76, 0,48, 0,53, 0,75 ; Hydrotel 0,76, 0,88, 0,91 puis 0,79, 0,91, 1,03, 1,21. Les deux modèles se trompent de saison en sens contraires ; meandre a trop d'eau au printemps et pas assez en été. Pluie annuelle du territoire : CaSR 1 229 mm, GOENS 1 096 mm (−11 %, l'écart étant presque tout en hiver).

Hypothèse à éprouver : l'excès d'hiver de CaSR gonfle le printemps, et l'entraînement, pour garder le volume annuel juste, a appris à amortir l'été. Épreuve en cours : passe à poids gelés avec la pluie de GOENS (`forcage-pluie-goens.py`, forçage de diagnostic, non de production).

## Volume d'été du sud : la pluie d'hiver de CaSR fait l'excès de printemps (2026-10-24, après-midi)

Forçage de diagnostic (`forcage-pluie-goens.py`) : le forçage CaSR brut du Saint-Laurent sud-ouest dont la pluie est remplacée par celle de GOENS (météo de l'ensemble Hydrotel), cellule la plus proche de chaque tronçon ; pluie annuelle du territoire 1 096 mm contre 1 229 pour CaSR. Passes à poids gelés, 2019-2024, sur le poste (Ubuntu injoignable pendant une heure) et sur Ubuntu :

| Saint-Laurent sud-ouest | CaSR, poste | GOENS, poste | CaSR, Ubuntu | GOENS, Ubuntu |
|---|---|---|---|---|
| débit simulé / observé, mars, avril, mai | 1,26, 1,37, 1,39 | 1,11, 1,07, 1,04 | | |
| débit simulé / observé, juin, juillet, août | 0,78, 0,64, 0,69 | 0,70, 0,61, 0,66 | | |
| été : base / événements / moyenne | 1,07 / 0,52 / 0,66 | 0,94 / 0,49 / 0,61 | 1,22 / 0,56 / 0,73 | 1,09 / 0,55 / 0,69 |
| terme d'étiage | 0,4475 | 0,4190 | 0,4648 | 0,4170 |
| perte / KGE médian | 1,0837 / 0,652 | 1,1305 / 0,608 | 1,1202 / 0,629 | 1,1250 / 0,602 |

Lecture. L'excès de printemps de meandre vient de la pluie d'hiver de CaSR : avec la pluie de GOENS, mars à mai reviennent à 1,04-1,11. Le déficit d'été ne bouge pas à poids gelés (pluie d'été égale). Le terme d'étiage gagne 6 à 10 %, la perte totale et le KGE perdent : le modèle a été appris avec le volume de CaSR. Test croisé sans simulation (`hiver-printemps-ete.csv`) : d'une station à l'autre, l'excès de pluie d'hiver n'annonce que faiblement le déficit d'été (corrélation de rang −0,26), et au Saint-Laurent sud-ouest l'erreur de printemps et celle d'été varient dans le même sens (+0,55) ; une compensation, s'il y en a une, passe par le champ partagé, pas station par station.

Épreuve longue lancée, et ce que seul elle dira : si, une fois l'excès de pluie d'hiver retiré, l'entraînement rend aux orages d'été le volume qu'il leur retire avec CaSR. Deux affinages identiques du modèle provincial sur le Saint-Laurent sud-ouest, l'un sous CaSR, l'autre sous la pluie de GOENS, six époques au taux 5e-5, Ubuntu, un point de reprise par époque, jugés à la dernière époque sur 2019-2024. Issue attendue si l'hypothèse tient : sous GOENS, les événements d'été remontent nettement au-dessus de 0,55 et le terme d'étiage baisse ; si les deux bras finissent au même été, la pluie d'hiver n'est pas la cause du déficit d'été et la piste se ferme sur la structure de la réponse aux orages.

Montérégie, même passe gelée avec la pluie de GOENS (poste) : mars à mai 1,08, 1,12, 1,10 sous CaSR contre 0,76, 0,88, 0,92 sous GOENS ; juin à août 0,81, 0,50, 0,60 contre 0,71, 0,45, 0,55 ; terme d'étiage 0,790 contre 0,828, perte 1,502 contre 1,673. Ici la pluie de GOENS rend le printemps trop faible et l'été plus déficitaire encore : les pertes du modèle (évaporation, stockage) ont été apprises sur le volume de CaSR. L'excès d'hiver ne suffit pas à expliquer l'été de la Montérégie.

Chemins de l'écoulement d'été au tronçon des stations (diagnostics 2011-2024) : pour les orages de 15 mm et plus, la production sur trois jours vaut 0,114 de la pluie en surface en Outaouais, 0,085 au Saint-Laurent sud-ouest, 0,104 en Montérégie, l'hypodermique et la base moins de 0,03 partout. La colonne répond aux orages de la même façon partout ; c'est la réponse observée qui diffère d'un territoire à l'autre (plus forte au sud, et croissante avec l'humidité antécédente, mesure du 23 octobre).

## Volume d'été du sud : production du bassin contre débit à la station (2026-10-24, soir)

Le pilote écrit désormais aussi les MOYENNES DE BASSIN des diagnostics journaliers (`ETL_DUMP_STATIONS`, préfixe `basin_`, pondération par l'aire locale des tronçons amont). Passe gelée locale, Saint-Laurent sud-ouest, 2011-2024 ; `bassin-orages.py`.

Écart d'aire. Rapporté à l'aire du bassin DANS LE MODÈLE, le débit simulé égale l'apport latéral du bassin à 1 % près à toutes les stations, sur l'année comme en été : le routage ne perd pas d'eau. Mais plusieurs petites stations du Saint-Laurent sud-ouest ont un bassin modèle très différent de leur aire déclarée : 031501, 8,5 km² contre 21,9 ; 023007, 9,6 contre 19,9 ; 02E901, 256 contre 69 ; 023444, 35,5 contre 23,7 ; 023005, 49,7 contre 30,6 ; 024016, 50 contre 74 ; 024004, 169 contre 101. Sur les 144 stations du modèle provincial, 4 seulement s'écartent de plus de 25 % (`aires-stations.py`), mais le Saint-Laurent sud-ouest en porte plusieurs hors de cet ensemble. C'est un défaut de rattachement des stations au réseau PHYSITEL, à corriger hors de la physique, et qui fausse le volume de ces stations dans un sens ou dans l'autre.

Orages d'été, 27 stations à aire juste, 2 303 orages, coefficients rapportés à la pluie des cinq jours : production du bassin 0,049, débit simulé à la station en cinq jours 0,021, observé 0,062. Par taille de pluie : 10-20 mm 0,038 / 0,014 / 0,025 ; 20-35 mm 0,044 / 0,019 / 0,048 ; plus de 35 mm 0,066 / 0,034 / 0,128. Deux défauts se cumulent : (1) le routage retarde l'eau d'orage au-delà de cinq jours (46 % de la production arrive dans ce délai, alors que le bilan annuel est juste à 0,99), d'autant plus que le bassin a de tronçons (corrélation de rang −0,31) ; (2) la colonne produit trop peu pour les gros orages (0,066 contre 0,128 observé au-delà de 35 mm). Épreuve en cours : passe gelée avec le temps de transfert de Muskingum divisé par deux (`ETL_MULT_FIXE=K_musk_hours:0.5`).

## Volume d'été du sud : le temps de transfert et l'évaporation (2026-10-24, soir)

Temps de transfert (passe gelée, Saint-Laurent sud-ouest, 2011-2024, `ETL_MULT_FIXE=K_musk_hours:0.5`) : la part de la production d'orage arrivée en cinq jours passe de 0,46 à 0,65, le coefficient d'orage simulé de 0,021 à 0,029 (observé 0,064), la perte de 1,179 à 1,171, le KGE médian de 0,677 à 0,694, le terme d'étiage de 0,480 à 0,488. Les volumes mensuels ne bougent pas (juin-août 0,79, 0,72, 0,69 contre 0,76, 0,70, 0,71) : le routage retarde l'eau d'orage mais ne fait pas le déficit de volume d'été. Le temps de transfert appris (environ 22 h par tronçon, même court) étale les crues ; à reprendre avec la forme de la réponse rapide.

Évaporation, sensibilité (même passe, `K_c:0.85`) : moyenne d'été 0,73 à 1,00, événements 0,58 à 0,75, mais ligne de base 1,10 à 1,57, terme d'étiage 0,480 à 0,561, perte 1,179 à 1,299. Le volume d'été est très sensible à l'évaporation ; l'erreur n'est pas son niveau mais sa saison.

Évaporation de mai-juin et cultures (`et-printemps-agri.log`, `mod16-agri-foret.log`). Aux tronçons des stations de trois territoires, l'ETR simulée sur MOD16 vaut en mai 1,08, 1,17, 1,40 et en juin 1,03, 1,02, 1,36 pour moins de 10 %, 10 à 40 % et plus de 40 % de cultures ; corrélation de rang de la part agricole avec l'excès 0,56 en mai, 0,69 en juin, 0,16 en juillet. MOD16 lui-même donne aux tronçons agricoles 20 à 25 % de moins qu'aux tronçons forestiers de fin mai à juin et davantage en juillet-août, avec un maximum deux à trois semaines plus tard. Or la recette applique à la classe agricole le calendrier calé sur la forêt feuillue de l'Outaouais (`[phenology] params`, une seule forme pour les deux classes).

Réglage ajouté : `phenology.params_agri` (alias `MEANDRE_PHENOLOGIE_PARAMS_AGRI`, défaut absent = comportement d'origine), calendrier propre à la classe agricole dans le mode photo, sans paramètre appris nouveau (le point de reprise gelé se relit). Essai : levée à 190 degrés-jours base 10 (le seuil forestier, 56,8, est atteint vers le 12 mai en Montérégie et le 20 mai au Saint-Laurent sud-ouest ; 190 correspond à environ trois semaines plus tard), largeur 50, sénescence inchangée. Passes gelées 2011-2024 :

| | Saint-Laurent sud-ouest, témoin | calendrier agricole | Montérégie, témoin | calendrier agricole |
|---|---|---|---|---|
| ETR / MOD16 aux tronçons de station à plus de 40 % agricoles, mai, juin | 1,25, 1,25 | 1,04, 1,08 | 1,51, 1,51 | 1,12, 1,44 |
| débit simulé / observé, juin, juillet, août | 0,79, 0,72, 0,69 | 0,98, 0,81, 0,75 | 0,73, 0,64, 0,76 | 0,86, 0,73, 0,79 |
| été : moyenne / événements | 0,73 / 0,58 | 0,80 / 0,64 | 0,64 / 0,60 | 0,68 / 0,66 |
| perte / terme d'étiage | 1,179 / 0,480 | 1,195 / 0,490 | 1,450 / 0,752 | 1,434 / 0,745 |

Lecture. Le calendrier agricole corrige le défaut physique visé (évaporation de mai-juin des terres cultivées, jugée contre MOD16, observation indépendante du débit) et rend une partie de l'été, sans toucher la ligne de base. À poids gelés, la perte gagne 1,1 % en Montérégie et perd 1,3 % au Saint-Laurent sud-ouest, où le reste du champ a été appris avec l'ancienne phénologie. Un ajustement global du calendrier sur le rapport des MOD16 agricole et forestier ne converge pas (l'agricole évapore plus que la forêt en avril, sol nu, et moins en juin) ; le critère retenu est l'ETR sur MOD16 aux tronçons agricoles en mai-juin. Pas de valeur propre à un territoire. Suite : affinage avec ce calendrier contre témoin, quand Ubuntu aura fini les bras CaSR contre GOENS.

## Volume d'été du sud : affinages (2026-10-24, nuit)

Affinage du modèle provincial sur le Saint-Laurent sud-ouest, six époques à 5e-5, CaSR, Ubuntu (validation 2019-2021 : KGE 0,847 à 0,869, la dernière époque est la meilleure). Comparé au modèle gelé sur la même machine, 2019-2024 : perte 1,120 à 1,057, KGE médian 0,629 à 0,676, terme d'étiage 0,465 à 0,447 ; mais l'été se creuse, moyenne 0,73 à 0,68, juillet 0,74 à 0,67, événements 0,56 à 0,55, et mai reste trop fort (1,64 à 1,51). L'entraînement sous CaSR pousse l'été vers le bas. Bras en cours : même affinage sous la pluie de GOENS (Ubuntu), puis sous CaSR avec le calendrier agricole (Ubuntu, programmé) ; en Montérégie, témoin CaSR puis calendrier agricole (poste).

Bras GOENS terminé, jugé à la DERNIÈRE époque (points de reprise e6, réévalués à poids gelés ; le point « best » du pilote était celui de la troisième époque dans les deux bras). Saint-Laurent sud-ouest, 2019-2024, Ubuntu :

| | gelé, CaSR | gelé, pluie GOENS | affiné, CaSR | affiné, pluie GOENS |
|---|---|---|---|---|
| perte | 1,1202 | 1,1250 | 1,0498 | 1,0191 |
| terme d'étiage | 0,4648 | 0,4170 | 0,4480 | 0,4352 |
| KGE médian | 0,629 | 0,602 | 0,688 | 0,700 |
| été : base / événements / moyenne | 1,22 / 0,56 / 0,73 | 1,09 / 0,55 / 0,69 | 1,07 / 0,56 / 0,69 | 1,24 / 0,67 / 0,83 |
| avril, mai | 1,32, 1,64 | 1,03, 1,24 | 1,37, 1,55 | 1,08, 1,25 |
| juin, juillet, août | 0,83, 0,74, 0,77 | 0,76, 0,70, 0,72 | 0,78, 0,67, 0,74 | 0,83, 0,82, 0,88 |

VERDICT (une graine, un territoire). L'hypothèse tient dans ses deux moitiés : l'excès de pluie d'hiver de CaSR (pluie annuelle du territoire 1 229 mm contre 1 096 pour GOENS) gonfle le printemps, et l'entraînement sous CaSR compense en creusant l'été. À entraînement identique, la pluie de GOENS rend à l'été 20 % de volume (moyenne 0,69 à 0,83, événements 0,56 à 0,67), ramène avril-mai de 1,37-1,55 à 1,08-1,25, et améliore la perte de 2,9 % et le terme d'étiage de 2,9 %. Le déficit d'été du sud est donc d'abord un effet du forçage sur l'apprentissage, pas de la structure de la colonne. Conséquence pour la décision d'Essi (prétraitement de la météo hors de meandre) : la précipitation d'hiver de CaSR doit être corrigée, ou une autre source retenue pour l'hiver, avant tout calage ; le calendrier agricole et le temps de transfert restent des corrections physiques à éprouver ensuite. À confirmer en Montérégie (bras programmés sur Ubuntu) et sur une seconde graine.

Bras du calendrier agricole (CaSR, même affinage, même graine, `phenology.params_agri = 190,11.82,50,0.9`, dernière époque), Saint-Laurent sud-ouest, 2019-2024, Ubuntu : perte 1,0637 contre 1,0498 pour l'affinage CaSR sans calendrier (hausse de 1,3 %), terme d'étiage 0,4522 contre 0,4480, KGE médian 0,681 contre 0,688, été moyenne 0,67 contre 0,69, juillet-août 0,66-0,67 contre 0,67-0,74. À poids gelés le calendrier rendait du volume à l'été (moyenne 0,73 à 0,80) ; l'entraînement le reprend et finit un peu plus bas. VERDICT (une graine) : le calendrier agricole ne corrige pas le déficit d'été une fois l'apprentissage fait, et dégrade légèrement la perte ; il n'entre pas dans la recette. Cohérent avec R400 : tant que le printemps reçoit trop d'eau, l'optimiseur creuse l'été quelle que soit la demande évaporative de juin.

Le poste Windows s'est révélé inutilisable pour l'entraînement (une heure sans terminer la première époque, carte graphique à 0 %, contre 17 minutes par époque sur Ubuntu) ; les bras de la Montérégie et la seconde graine sont en file sur Ubuntu : Montérégie CaSR, pluie GOENS, calendrier agricole, pluie GOENS d'hiver seule (novembre à avril, `forcing-<reg>-goensPwinter.nc`) ; puis Saint-Laurent sud-ouest, graine 4321, pluie GOENS d'hiver seule, CaSR, pluie GOENS. Le bras « hiver seul » sépare l'effet de l'excès hivernal de celui de la pluie d'été : il retire 94 des 133 mm annuels d'écart au Saint-Laurent sud-ouest, 77 des 101 mm en Montérégie.

### Pluie d'hiver seule, à poids gelés (2026-10-25, nuit)

Modèle provincial gelé, poste Windows, 2019-2024 pour les trois forçages (une première série comparait un témoin 2011-2024 à des variantes 2022-2024 : jetée). Rapport simulé sur observé par mois, médiane des stations.

| | CaSR | pluie GOENS de novembre à avril | pluie GOENS toute l'année |
|---|---|---|---|
| Saint-Laurent sud-ouest : perte | 1,084 | 1,133 | 1,131 |
| terme d'étiage | 0,448 | 0,437 | 0,419 |
| janvier, février | 1,33, 1,10 | 1,28, 1,12 | 1,21, 1,10 |
| avril, mai | 1,37, 1,39 | 1,08, 1,06 | 1,07, 1,04 |
| juillet, août ; été moyenne | 0,64, 0,69 ; 0,66 | 0,63, 0,68 ; 0,63 | 0,61, 0,66 ; 0,61 |
| Montérégie : perte | 1,502 | 1,655 | 1,673 |
| terme d'étiage | 0,790 | 0,820 | 0,828 |
| janvier, février | 1,46, 1,48 | 1,29, 1,21 | 1,24, 1,19 |
| avril, mai | 1,12, 1,10 | 0,88, 0,98 | 0,88, 0,92 |
| juillet, août ; été moyenne | 0,50, 0,60 ; 0,69 | 0,50, 0,59 ; 0,62 | 0,45, 0,55 ; 0,54 |

Au Saint-Laurent sud-ouest, la pluie d'hiver seule retire tout l'excès de printemps que retirait la pluie GOENS entière : l'excès vient de l'hiver, et l'été d'un modèle gelé n'en dépend presque pas (le report d'hiver sur l'été est faible à poids gelés ; c'est l'apprentissage qui le crée, R400). En Montérégie, le tableau est différent : sous CaSR le printemps n'est pas en excès (1,10 à 1,12), seul l'hiver l'est (1,46 à 1,48), et l'été est à moitié vide sous les trois forçages (juillet 0,45 à 0,50). Le déficit d'été de la Montérégie ne s'explique donc pas par la compensation d'un excès de printemps ; c'est un défaut propre à l'été (évaporation, drainage agricole, ou stockage), que le forçage ne corrige pas. Les bras entraînés diront si l'apprentissage sous pluie corrigée y rend du volume (Ubuntu, en file).

### L'été 2023 de la Montérégie n'est pas la pluie (2026-10-25, nuit)

Volume de juin à septembre simulé sur observé, médiane des stations ayant plus de 100 jours mesurés dans l'été, mêmes stations et mêmes jours pour les trois séries. meandre gelé (poste), sous CaSR et sous la pluie de GOENS ; Hydrotel = moyenne des six membres de l'ensemble RQH 2026-10, qui tourne sur GOENS.

| | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| Montérégie, meandre CaSR | 1,10 | 1,09 | 0,60 | 0,29 | 0,50 |
| meandre pluie GOENS | 0,93 | 0,91 | 0,43 | 0,30 | 0,48 |
| Hydrotel (GOENS) | 2,00 | 0,93 | 1,23 | 0,85 | 1,10 |
| Saint-Laurent sud-ouest, meandre CaSR | 0,68 | 0,67 | 0,85 | 0,56 | 0,56 |
| meandre pluie GOENS | 0,60 | 0,61 | 0,65 | 0,58 | 0,54 |
| Hydrotel (GOENS) | 1,19 | 1,13 | 1,18 | 0,92 | 1,06 |

Avec la même pluie, Hydrotel rend l'été 2023 de la Montérégie à 0,85 et meandre à 0,30. Le déficit des étés d'orages record n'est donc pas un défaut du forçage, contrairement à la lecture retenue le 9 octobre (CaSR sous les stations canadiennes en 2023-2024) : il est dans la réponse du modèle aux gros orages. Les étés ordinaires (2020-2021) sont justes en Montérégie avec CaSR. Au Saint-Laurent sud-ouest, le déficit est présent toutes les années. Le post-traitement RQH (`postprocess_*.zarr`) est la simulation Hydrotel brute qui sert de fond à l'interpolation avec les stations (`interpolation/`, produit distinct) : pas d'assimilation des débits dans ce qu'on compare (config_RQH_2026-10.toml) ; une étape de concaténation de débits observés existe pour certaines stations, non vérifiée tronçon par tronçon. Épreuve en cours : surface moins perméable en passe avant (`K_sat_1` × 0,33 et × 0,1), lue année par année.

### Montérégie, affinages CaSR contre pluie GOENS (2026-10-25, nuit)

Même protocole que le Saint-Laurent sud-ouest (six époques à 5e-5, graine 1234, Ubuntu, dernière époque, 2019-2024). Perte 1,3672 sous CaSR, 1,3814 sous GOENS (hausse de 1,0 %) ; terme d'étiage 0,714 contre 0,682 (baisse de 4,5 %) ; KGE médian 0,550 contre 0,542 ; été moyenne 0,77 contre 0,71, événements 0,72 contre 0,64. Été par année, 2019 à 2024 : CaSR 1,07, 1,18, 1,23, 0,72, 0,31, 0,54 ; GOENS 1,02, 1,11, 1,07, 0,65, 0,46, 0,63. En Montérégie, la pluie d'hiver corrigée ne fait pas le gain net du Saint-Laurent sud-ouest : l'étiage s'améliore, la perte totale non. Cohérent avec la passe gelée (pas d'excès de printemps à compenser en Montérégie) et avec R402 : le défaut de la Montérégie est la réponse des étés humides, que l'affinage ne corrige sous aucun des deux forçages (2023 à 0,31 et 0,46, Hydrotel 0,85).

Conductivité de surface réduite, passe gelée, Montérégie, CaSR, été par année 2019 à 2024 : témoin 1,02, 1,10, 1,09, 0,60, 0,29, 0,50 ; `K_sat_1` × 0,33 : 1,15, 1,32, 1,12, 0,68, 0,36, 0,57 (perte 1,502 à 1,520) ; × 0,1 : 1,74, 1,85, 2,07, 0,83, 0,48, 0,81 (perte 1,772). La perméabilité de surface élève tous les étés à peu près dans la même proportion ; elle ne crée pas l'amplification des étés humides. Bilan de la colonne de juin à septembre (médiane des bassins jaugés) : 2021, apport au sol 282 mm, ETR 372, écoulement latéral 17, observé environ 16 ; 2023, 483, 418, 41, observé environ 140. La couche de surface reste vers 0,24 de teneur en eau, contre une porosité apprise de 0,60 (borne) et une capacité au champ de 0,48 (Hydrotel 0,43 et 0,27) : le sol n'approche jamais la saturation, l'aire contributive ne s'ouvre pas, l'excédent de pluie part en évaporation et en stock. Épreuve en cours : porosités des couches 1 et 2 × 0,72, avec et sans capacité au champ de surface × 0,56.

Porosité, passe gelée, Montérégie, CaSR, 2019-2024. Les multiplicateurs s'appliquent APRÈS le calcul de la capacité au champ (fraction de la porosité) : porosités 1 et 2 × 0,72 seules laissent la capacité au champ de surface à 0,48 au-dessus d'une porosité de 0,43, sol incohérent qui met l'évapotranspiration en stress permanent. Résultat de cette passe : perte 1,502 à 1,369, étiage 0,790 à 0,727, été 2023 0,29 à 0,46, mais ETR de juin-septembre réduite d'environ 50 mm chaque année (2021 : 372 à 323 mm ; 2023 : 418 à 369). Version cohérente (porosités × 0,72 et capacité au champ de surface × 0,56, soit 0,43 et 0,27 comme Hydrotel) : perte 1,535, étiage 0,830, KGE 0,569, été par année 0,98, 1,32, 1,00, 0,62, 0,30, 0,62 ; ETR rétablie (2023 : 431 mm). VERDICT : le stockage du sol n'est pas la cause du déficit des étés humides ; le gain apparent de la première passe venait de l'évaporation. À poids gelés, retirer 50 mm d'ETR d'été améliore la perte de 9 %, ce qui contredit l'évaporation d'été jugée juste le 9 octobre ; vérification en cours contre MOD16, année par année.

### Pourquoi les étés humides manquent : réponse trop peu dépendante de l'humidité (2026-10-25, matin)

Volume d'été décomposé par année (Montérégie, gelé, CaSR, médiane des stations ; ligne de base = quantile 20 % de l'été, événements = volume au-dessus) : en 2021, été sec, événements simulés sur observés 1,72, base 0,88 ; en 2023, 0,19 et 0,60 ; en 2022 et 2024, 0,55 et 0,45. Le défaut des étés humides est surtout un défaut de crues d'orage ; les étés secs en ont trop. C'est, à l'échelle de l'année, la même forme que celle mesurée le 23 octobre sur les orages pris un à un (R394).

ETR de juin-septembre au tronçon des stations contre MOD16 (`et-ete-annee.py`), Montérégie, 2011-2024 : la simulée suit la pluie (2,81 mm/j en 2021, 3,59 en 2022), MOD16 reste entre 2,96 et 3,40 sans lien avec elle ; rapport 0,85 à 0,88 les étés secs (où le débit est juste), 1,01 à 1,03 les étés humides (où il manque). Saint-Laurent sud-ouest : rapport 1,00 à 1,14, sans tendance avec l'année.

Exposant de l'aire contributive (`column.saturated_area_exponent`, 8 dans la recette ; part (theta1/porosité1)^b). La porosité de surface est apprise à 0,60, sa borne haute, et la teneur en eau d'été reste vers 0,18 à 0,24 : à b = 8 la part ruisselée vaut 10⁻⁴ à 10⁻³, l'aire contributive est éteinte. Le modèle provincial a été entraîné de zéro avec b = 8 (R387), et le champ a porté la porosité à la borne, ce qui est la même échappatoire que celle mesurée en Outaouais le 1er octobre (R249 : le champ annule la réponse aux orages qu'on lui impose). Passes gelées, été par année 2019 à 2024 :

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | perte | étiage |
|---|---|---|---|---|---|---|---|---|
| témoin (b = 8) | 1,02 | 1,10 | 1,09 | 0,60 | 0,29 | 0,50 | 1,502 | 0,790 |
| b = 4 | 1,26 | 1,37 | 1,42 | 0,74 | 0,45 | 0,64 | 1,485 | 0,851 |
| b = 2 | 2,20 | 2,57 | 2,94 | 1,25 | 0,80 | 1,00 | 1,799 | 1,174 |
| `K_sat_1` × 0,1 | 1,74 | 1,85 | 2,07 | 0,83 | 0,48 | 0,81 | 1,772 | 1,065 |

Aucun levier de surface ne crée l'amplification des étés humides : tous élèvent les étés dans une proportion voisine. Bilan de la colonne (b = 4, juin-septembre, médiane des bassins) : 2021, apport 282 mm, ETR 367, écoulement 22, variation de stock −107 ; 2023, apport 483, ETR 413, écoulement 68, stock +2. Sur 200 mm de pluie en plus, le modèle en évapore 46 de plus et en garde 109 dans le sol, parce qu'il a puisé 107 mm dans le stock pendant l'été sec ; le débit observé, lui, passe d'environ 16 à 140 mm. VERDICT : le défaut est un stock de sol trop grand et trop disponible à l'évaporation, qui amortit l'écart entre étés secs et humides ; ni la perméabilité de surface ni l'exposant de l'aire contributive ne le corrigent, et la porosité collée à sa borne en est une partie. La pluie d'été des deux forçages est la même (CaSR sur GOENS 1,04 en Montérégie, juin-septembre) ; sur la part du forçage, voir la correction ci-dessous.

Calendrier agricole affiné en Montérégie (mêmes réglages, dernière époque, 2019-2024, Ubuntu) : perte 1,3719 contre 1,3672 sans calendrier (+0,3 %), étiage 0,724 contre 0,714, KGE médian 0,533 contre 0,550, été moyenne 0,78 contre 0,77. Confirme R401 sur un second territoire : pas de gain après apprentissage.

Été par année, Saint-Laurent sud-ouest, Ubuntu, 2019 à 2024 : gelé CaSR 0,85, 0,76, 0,76, 0,95, 0,62, 0,66 ; affiné CaSR 0,80, 0,71, 0,73, 0,89, 0,59, 0,59 ; affiné pluie GOENS 0,81, 0,87, 0,80, 0,91, 0,77, 0,73. Le gain de la pluie corrigée y porte sur toutes les années, et surtout sur les étés humides (2023 : 0,59 à 0,77).

### Correction : la comparaison à Hydrotel n'innocente pas la pluie (2026-10-25, matin)

La section « L'été 2023 de la Montérégie n'est pas la pluie » conclut trop vite. Hydrotel obtient l'été de la Montérégie par une demande d'évaporation calée à 0,40 fois Linacre, environ 35 % sous MOD16 (R337) ; son 0,85 de 2023 ne dit donc rien de la suffisance de la pluie. Bilan de juin-septembre fermé par station (médiane des bassins jaugés, mm ; débit observé rapporté à l'aire déclarée) :

| | apport au sol (forçage) | débit observé | ETR simulée du bassin | MOD16 au tronçon | apport − débit − MOD16 |
|---|---|---|---|---|---|
| 2021 | 284 | 13 | 360 | 353 | −93 |
| 2022 | 531 | 123 | 464 | 382 | +9 |
| 2023 | 482 | 157 | 415 | 362 | −59 |
| 2024 | 506 | 154 | 462 | 376 | −31 |

Un déstockage de 59 mm pendant l'été le plus humide n'est pas vraisemblable : soit la pluie des étés d'orages est sous-estimée (CaSR à 0,79 des stations canadiennes en 2023, R355 ; GOENS donne la même pluie d'été), soit MOD16 est trop haut. Et l'ETR simulée du bassin dépasse MOD16 de 50 à 90 mm les étés humides. Le déficit des étés humides de la Montérégie relève donc des deux : une pluie probablement trop basse, et une évaporation simulée qui suit la pluie au lieu de rester bornée par l'énergie. Le verdict R403 (amortissement par le stock et l'évaporation) tient ; la part du forçage reste ouverte et se tranche hors de meandre, par la pluie des stations.

Montérégie, affinage sous la pluie GOENS d'hiver seule (novembre à avril, CaSR le reste de l'année ; même protocole, dernière époque, 2019-2024, Ubuntu) : perte 1,4191 contre 1,3672 sous CaSR (+3,8 %) et 1,3814 sous GOENS toute l'année ; étiage 0,705 contre 0,714 et 0,682 ; KGE médian 0,515 ; été par année 1,07, 1,14, 1,04, 0,81, 0,37, 0,62. En Montérégie, corriger la pluie d'hiver ne paie pas sur la perte : le mécanisme établi au Saint-Laurent sud-ouest (R400, compensation d'un printemps en excès) ne s'y applique pas, faute d'excès de printemps sous CaSR. Le Saint-Laurent sud-ouest, graine 4321, trois forçages, tourne sur Ubuntu ; il dira si R400 tient à la seconde graine et si la pluie d'hiver seule y suffit.

### Champs collés à leurs bornes (2026-10-25, matin)

`bornes-collees.py`, modèle provincial gelé, 27 834 tronçons, part des tronçons à moins de 2 % de l'étendue d'une borne : porosité de la couche 1 à la borne haute 0,60 sur 81 % des tronçons, couches 2 et 3 sur 66 % ; capacité au champ de la couche 3 à la borne haute de sa fraction de la porosité (0,85) sur 82 % ; épaisseur de la couche 2 à la borne basse 0,30 m sur 84 %, couche 3 à 0,50 m sur 89 % (Hydrotel : 0,16 et 2,65 m en Montérégie) ; k_gw à la borne haute sur 19 %. L'optimiseur veut des couches 2 et 3 plus minces que ne le permettent les bornes, et des porosités plus fortes.

Passe gelée, Montérégie, CaSR, 2019-2024, bornes basses relâchées (`experiments/plan_ml/modele-provincial-minces.toml` : couche 2 à 0,10 m, couche 3 à 0,20 m ; les sorties saturées vont à la nouvelle borne) : perte 1,502 à 1,466 (−2,4 %), KGE médian 0,533 à 0,557, étiage 0,790 à 0,788, été inchangé (0,69 à 0,65), ETR d'été de l'année sèche 367 à 331 mm. Affinage de six époques avec ces bornes programmé sur Ubuntu après la seconde graine du Saint-Laurent sud-ouest ; il dira jusqu'où le champ va et si le gain grandit, contre l'affinage CaSR de la Montérégie (perte 1,3672).

Même passe au Saint-Laurent sud-ouest : perte 1,084 à 1,037 (−4,4 %), KGE médian 0,652 à 0,665, étiage 0,448 à 0,444, été 0,66 à 0,61. Deux territoires, même sens, sans entraînement.

Bornes plus basses encore (couche 2 à 0,05 m, couche 3 à 0,10 m), Saint-Laurent sud-ouest, gelé : perte 1,029 (contre 1,037 à 0,10/0,20 et 1,084 aux bornes actuelles), KGE médian 0,676, mais étiage 0,456 (contre 0,444 et 0,448). Le gain s'épuise et l'étiage se dégrade ; 0,10 et 0,20 m restent le candidat (couche 2 d'Hydrotel : 0,16 m).

Borne haute de la porosité abaissée de 0,60 à 0,50 (`[field.bounds] porosity`, `modele-provincial-poro050.toml` ; la capacité au champ se calcule après la borne, comme fraction de la porosité, donc le sol reste cohérent, à la différence du multiplicateur), passes gelées 2019-2024, CaSR : Saint-Laurent sud-ouest perte 1,084 à 1,062, étiage 0,448 à 0,442, KGE 0,652 à 0,654 ; Montérégie perte 1,502 à 1,474, étiage 0,790 à 0,788, KGE 0,533 à 0,556, événements d'été 0,67 à 0,77, été 2024 0,50 à 0,58. L'entraînement a porté la porosité à la borne, et l'en redescendre améliore la perte : le signe d'une sortie saturée dont la sigmoïde n'a plus de gradient, et non d'un optimum. Combinaison des deux relâchements en cours.

Les deux relâchements ensemble (`modele-provincial-minces-poro050.toml`), gelé, 2019-2024 : Saint-Laurent sud-ouest perte 1,084 à 1,030 (−5,0 %), KGE médian 0,652 à 0,665, étiage 0,448 à 0,447 ; Montérégie perte 1,502 à 1,435 (−4,4 %), KGE 0,533 à 0,580, étiage 0,790 à 0,784, été par année 0,98, 1,07, 1,14, 0,61, 0,29, 0,56. Meilleure variante sur les deux territoires, sans entraînement ; les étés humides de la Montérégie ne bougent pas (R403). Affinages de six époques avec ces bornes programmés sur Ubuntu après la seconde graine (Montérégie, puis Saint-Laurent sud-ouest), contre les affinages CaSR aux bornes actuelles (perte 1,3672 et 1,0498, graine 1234).

Contrôle au nord, Outaouais, gelé, 2019-2024 : bornes actuelles perte 0,884, étiage 0,404, KGE 0,710, base d'été 0,87 ; bornes relâchées (couches minces et porosité 0,50) perte 1,009 (+14 %), étiage 0,487, KGE 0,702, base d'été 0,66. La combinaison aide le sud et dégrade nettement le nord : à poids gelés, elle ne peut pas entrer dans la recette provinciale. Séparation des deux effets en Outaouais en cours. Lecture provisoire : les bornes collées sont un compromis du champ partagé entre territoires, et non un simple défaut de gradient ; un relâchement ne se juge qu'après réentraînement de la province.

Séparation en Outaouais, gelé : porosité 0,50 seule perte 0,904 (+2,3 %), étiage 0,418, base d'été 0,80 ; couches minces seules 0,968 (+9,5 %), 0,466, 0,69. Le nord perd surtout par les couches minces : son débit de base d'été tient à l'épaisseur de la couche 3. Or celle-ci est aussi collée à 0,50 m sur la plupart des tronçons : la sortie est saturée sans que la perte le demande, et un relâchement à poids gelés la déplace partout à la fois. Seul l'entraînement dira si le champ libéré fait des couches minces au sud et épaisses au nord. Affinages Outaouais CaSR, bornes actuelles contre bornes relâchées, programmés sur Ubuntu après les bras du sud.

### Pluie d'hiver seule, entraînée ; la graine n'agit pas sur un affinage (2026-10-25, matin)

Saint-Laurent sud-ouest, affinage de six époques, dernière époque, 2019-2024, Ubuntu. La graine 4321 sous CaSR reproduit la graine 1234 à la quatrième décimale (perte 1,0498, étiage 0,4480, KGE 0,6884) : un affinage depuis le même point de reprise, sans tirage aléatoire, est déterministe, et `ETL_SEED` n'y change rien. Les « secondes graines » ne mesurent donc pas de dispersion ; il faudrait perturber autre chose (point de départ, ordre des blocs) pour l'estimer.

Pluie GOENS d'hiver seule (novembre à avril) : perte 1,0490 (CaSR 1,0498, GOENS toute l'année 1,0191), étiage 0,444 (0,448 ; 0,435), KGE 0,704 (0,688 ; 0,700), été 0,77 (0,69 ; 0,83), avril-mai 1,08 et 1,28 (1,37 et 1,55 ; 1,08 et 1,25), été par année 0,90, 0,78, 0,78, 0,99, 0,68, 0,70. Corriger l'hiver suffit à retirer l'excès de printemps et rend la moitié du volume d'été perdu, mais pas le gain de perte : celui de R400 tient aussi à la pluie de mai à octobre (CaSR 2 à 9 % au-dessus de GOENS selon le mois). R400 se précise : l'excès d'hiver fait le printemps et une part de l'été ; la perte demande la pluie de GOENS toute l'année.

Troisième contrôle, Saint-Laurent nord-ouest (épaisseurs collées sur 100 % des tronçons), gelé, 2019-2024 : bornes actuelles perte 1,027, étiage 0,492, KGE 0,665, base d'été 0,75 ; bornes relâchées 1,284 (+25 %), 0,680, 0,660, 0,53. Part des tronçons collés par territoire (couche 3 sous 0,57 m) : Saint-Laurent sud-ouest et nord-ouest, Abitibi, Côte-Nord 93 à 100 % ; Gaspésie 93 % ; Outaouais, Outaouais-Montréal, Saguenay, Montérégie 71 à 75 % ; médiane de la couche 3 de 0,50 à 0,53 m partout. À poids gelés, relâcher les bornes ne gagne qu'aux deux territoires agricoles du sud et coûte 14 à 25 % de perte ailleurs, surtout par le débit de base d'été. Verdict provisoire : le relâchement n'est pas un remède à poids gelés ; les affinages en cours sur Ubuntu (Montérégie, Saint-Laurent sud-ouest, Outaouais) diront si le champ, entraîné, sépare les territoires ; une adoption exigerait un réentraînement de la province entière, à proposer à Essi.

### Bornes relâchées, entraînées : Montérégie (2026-10-25, fin de matinée)

Affinage de six époques à 5e-5, graine 1234, Ubuntu, dernière époque, 2019-2024, `modele-provincial-minces-poro050.toml` (couches 2 et 3 à partir de 0,10 et 0,20 m, porosité bornée à 0,50) contre le même affinage aux bornes actuelles : perte 1,3672 à 1,2673 (−7,3 %), étiage 0,714 à 0,676 (−5,3 %), KGE médian 0,550 à 0,592, été moyenne 0,77 à 0,78, événements 0,72 à 0,78 ; été par année 1,04, 1,17, 1,19, 0,78, 0,37, 0,62 (contre 1,07, 1,18, 1,23, 0,72, 0,31, 0,54). Le gain grandit à l'entraînement (−4,4 % à poids gelés) : la borne empêchait l'optimiseur d'aller où la perte le demande. Restent à voir le Saint-Laurent sud-ouest et surtout l'Outaouais, où le relâchement coûtait 14 % à poids gelés.

Paramètres appris par l'affinage de la Montérégie (`ETL_DUMP_REACH`, médiane des tronçons, entre parenthèses p10 et p90), bornes actuelles contre relâchées ; calage Hydrotel LN24HA en regard : porosité de la couche 1 0,49 (0,40-0,57) contre 0,43 (0,37-0,48), Hydrotel 0,43 ; capacité au champ de surface 0,33 contre 0,29, Hydrotel 0,27 ; couche 2 0,44 m (0,34-0,65) contre 0,26 m (0,15-0,49), Hydrotel 0,16 ; couche 3 0,67 m (0,55-0,98) contre 0,42 m (0,27-0,75), Hydrotel 2,65 ; K_sat_1 et K_c inchangés. Libéré, le champ rejoint le sol d'Hydrotel en surface sans qu'on le lui impose (porosité et capacité au champ), et amincit la couche 2 vers sa valeur ; la couche 3 s'amincit au lieu de s'épaissir. Remarque : l'affinage territorial seul, aux bornes actuelles, avait déjà décollé la Montérégie de ses bornes (porosité 0,60 à 0,49, couche 2 0,32 à 0,44 m) ; le collage du modèle provincial est donc en partie un compromis entre territoires.

Saint-Laurent sud-ouest, même affinage aux bornes relâchées : perte 1,0498 à 0,9682 (−7,8 %), étiage 0,448 à 0,442, KGE médian 0,688 à 0,712, été 0,69 à 0,73, événements 0,56 à 0,67 ; été par année 0,88, 0,78, 0,67, 0,95, 0,65, 0,67. Plus que la pluie de GOENS sous les bornes actuelles (perte 1,0191). Mai reste en excès (1,64) : l'excès de pluie d'hiver de CaSR (R400) est un défaut distinct, que ce changement ne traite pas. Deux territoires du sud, même sens, gain plus fort après entraînement qu'à poids gelés. Le contrôle décisif est l'Outaouais (en cours sur Ubuntu), où le relâchement coûtait 14 % à poids gelés.

Outaouais, même affinage : bornes actuelles perte 0,8573, étiage 0,398, KGE 0,713, base d'été 0,95 ; bornes relâchées 0,9063 (+5,7 %), 0,423, 0,708, 0,87 ; été par année proche (2023 : 0,63 contre 0,70). L'entraînement réduit le coût du relâchement au nord (+14 % à poids gelés) sans l'annuler. Le changement n'est donc pas adoptable tel quel pour la province ; séparation de ses deux pièces en Outaouais, entraînées, lancée sur Ubuntu (porosité seule, puis couches minces seules).

Outaouais, porosité bornée à 0,50 seule, affinée : perte 0,8593 contre 0,8573 (+0,2 %, dans le bruit), étiage 0,396 contre 0,398, KGE 0,708 contre 0,713, base d'été 0,95. La porosité bornée est neutre au nord une fois le champ adapté ; le coût de la combinaison au nord vient des couches minces. Affinages de la porosité seule au sud programmés (Montérégie, Saint-Laurent sud-ouest), après le bras « couches minces seules » de l'Outaouais.

Outaouais, couches minces seules, affinées : perte 0,8822 (+2,9 %), étiage 0,412 (+3,6 %), KGE 0,710, base d'été 0,94, octobre-novembre 1,27 (contre 1,17 et 1,14) : l'automne déborde, faute du stock de la couche 3. Synthèse de l'Outaouais affiné : porosité seule +0,2 %, couches minces seules +2,9 %, les deux +5,7 %. Au sud, la combinaison gagne 7,3 et 7,8 %. Une borne d'épaisseur commune ne peut pas servir les deux ; la porosité bornée, neutre au nord, est le seul changement candidat pour la province si elle gagne seule au sud (affinages en cours).

Montérégie, porosité bornée seule, affinée : perte 1,3430 (−1,8 % contre 1,3672), étiage 0,712 (0,714), KGE 0,547 (0,550), événements d'été 0,75 (0,72). Le gain du sud vient surtout des couches minces (combinaison −7,3 %).

Saint-Laurent sud-ouest, porosité bornée seule, affinée : perte 1,0223 (−2,6 % contre 1,0498), étiage 0,439 (−2,0 %), KGE 0,700. Bilan de la porosité bornée à 0,50 seule, affinée : Montérégie −1,8 %, Saint-Laurent sud-ouest −2,6 %, Outaouais +0,2 % ; étiage −0,3, −2,0 et −0,4 %. Changement sûr pour la province, petit gain. Le gros du gain du sud (−7 à −8 %) tient aux couches minces, qui coûtent au nord avec une borne commune : d'où la recherche d'une borne d'épaisseur par tronçon tirée des dépôts.

### Épaisseur des dépôts et épaisseur apprise (2026-10-25, soir)

`epaisseur-depots.py` : épaisseur des dépôts quaternaires par tronçon d'après SIGEOM (zones morphosédimentologiques, épaisseur inscrite VAL_EPAIS_MINIM/MAXIM sur 46 % de la surface cartographiée, sinon médiane du code : roc 0,3 m, till mince 1 m, till de couverture 15 m, argile marine 10 m, fluvioglaciaire 30 m). Surface couverte : Saint-Laurent sud-ouest 75 %, Outaouais 55 %, Montérégie 43 %, Saguenay 39 %, Saint-Laurent nord-ouest 26 %, Abitibi 12 %, Côte-Nord et Labrador 0 à 5 %. `depth_to_bedrock_m` de la table du champ vaut zéro partout (attribut mort).

`bornes-covariables.py`, couche 3 apprise par l'affinage aux bornes relâchées : corrélation de rang par tronçon avec le logarithme de l'épaisseur SIGEOM +0,37 (Montérégie, 899 tronçons) et +0,42 (Saint-Laurent sud-ouest, 2 238) ; avec la part des dépôts minces −0,46 et −0,52 ; avec la profondeur du roc aux puits du SIH +0,37 et +0,23 ; avec la part de roc −0,28 et −0,16. Prédiction de chaque territoire par l'autre (régression sur cinq covariables SIGEOM), erreur sur le logarithme de la couche 3 : −25 % et −35 % contre le témoin. Le champ, qui n'a jamais reçu l'épaisseur des dépôts, la retrouve en partie de lui-même, dans le même sens sur les deux territoires. Contrôle de l'Outaouais en cours.

### Correction : relâcher une borne remappe toutes les valeurs (2026-10-25, soir)

Les sorties du champ passent par une sigmoïde, valeur = borne basse + étendue × sigmoïde(sortie brute). Changer une borne déplace donc TOUS les tronçons, pas seulement ceux qui étaient collés : une couche 3 de 0,75 m passe à 0,47 m sans que l'optimiseur ait rien choisi. Médianes, bornes actuelles affinées → remappage pur de ces valeurs aux nouvelles bornes → affinage aux nouvelles bornes : couche 3 Montérégie 0,67 → 0,39 → 0,42 m, Saint-Laurent sud-ouest 0,58 → 0,28 → 0,31, Outaouais 0,66 → 0,37 → 0,45 ; couche 2 0,44 → 0,26 → 0,26, 0,33 → 0,14 → 0,14, 0,44 → 0,26 → 0,27 ; porosité de surface Montérégie 0,49 → 0,42 → 0,43.

Ce qui est retiré : « le champ libéré rejoint de lui-même la porosité et la capacité au champ d'Hydrotel » (c'est le remappage) ; « le champ retrouve de lui-même l'épaisseur des dépôts » (la corrélation de rang avec l'épaisseur SIGEOM existait déjà aux bornes actuelles : +0,36, +0,42, +0,24, contre +0,37, +0,42, +0,22 après relâchement). Ce qui reste : (1) au sud, des couches 2 et 3 plus minces de 0,2 à 0,3 m font baisser la perte affinée de 7 à 8 % ; (2) l'entraînement, à partir du remappage, épaissit la couche 3 là où les dépôts sont épais au sud (corrélation de rang du rapport appris sur remappé avec l'épaisseur +0,23 et +0,20) et l'épaissit partout en Outaouais (0,37 à 0,45 m), où la perte de fin d'entraînement est plus haute qu'aux bornes actuelles (1,4436 contre 1,4208, et validation 0,8245 contre 0,8339) : ce n'est pas un surapprentissage, l'Outaouais part de trop mince et six époques ne suffisent pas à remonter. Une épreuve propre du relâchement doit conserver les valeurs de départ (sorties brutes recalées sur les nouvelles bornes) pour que seul l'optimiseur déplace le sol.

### Bornes prolongées sans remappage : Montérégie (2026-10-26, midi)

Épreuve propre du relâchement (réglages `spatial.gradient_floor = 0,01` sur Z2, Z3 et la porosité, `spatial.extend_below = Z2:0,10, Z3:0,20`) : les valeurs de départ sont conservées exactement, une sortie collée reçoit un gradient, et la valeur ne descend sous l'ancienne borne que si l'optimiseur l'y pousse. Six époques à 5e-5 depuis le modèle provincial, dernière époque, 2019-2024, Ubuntu (instantané de code e84dda5) : perte 1,2467 contre 1,3672 aux bornes actuelles (−8,8 %) et 1,2673 au remappage ; terme d'étiage 0,681 contre 0,714 et 0,676 ; KGE médian 0,647 contre 0,550 et 0,592 ; été moyenne 0,91, événements 0,85 (0,72 au témoin) ; été par année 1,17, 1,33, 1,30, 0,88, 0,39, 0,61. Le gain est donc le fait de l'optimiseur et non du remappage, et il dépasse celui du remappage. Le printemps monte (avril 1,34, mai 1,19) : à voir sur les autres territoires et avec le taux 2e-4 (bras en file). Point de reprise rapatrié pour lire les paramètres choisis.

Paramètres appris par le bras prolongé (export sur Ubuntu, la machine de l'entraînement ; médiane, p10 à p90 des 1 916 tronçons) : couche 2 0,30 m (0,12 à 0,30), 29 % des tronçons sous l'ancienne borne de 0,30 ; couche 3 0,50 m (0,32 à 0,51), 19 % sous 0,50 ; porosité de surface 0,599 (0,59 à 0,60), à la borne haute presque partout ; capacité au champ de surface 0,49 ; coefficient cultural 0,92 (0,71 à 1,06), en baisse. Lecture : libéré, le champ n'amincit les couches que sur un cinquième à un tiers des tronçons, et pousse la porosité à sa borne haute sur tous ; une partie du gain vient d'une évaporation réduite par K_c. Une porosité de 0,60 partout n'est pas un sol, c'est une capacité de stockage que le débit ne borne pas : c'est le cas d'école du prior de l'étape 3. Réserve : les exports des bras aux bornes actuelles ont été faits sur le poste, dont la table d'attributs diffère ; export sur Ubuntu en cours pour une comparaison loyale.

CORRECTION (2026-10-26, 13 h). Les exports de paramètres faits sur le poste le 25 octobre (`reach-mont-ft-casr.npz`, `reach-mont-ft-bornes.npz`) ne décrivent pas les modèles entraînés : la table d'attributs du poste diffère de celle d'Ubuntu (distance à l'exutoire), et le champ, qui la lit, y produit d'autres sorties. Export du même bras aux bornes actuelles sur Ubuntu : couche 2 0,300 m sur TOUS les tronçons (p10 à p90 0,300 à 0,301), couche 3 0,501 (0,500 à 0,505), porosités 0,600, capacité au champ de surface 0,486. Le modèle affiné reste donc exactement collé à ses bornes, comme le modèle gelé ; les médianes de 0,44 et 0,67 m rapportées hier pour ce bras sont un artefact de machine. Comparaison loyale, Ubuntu contre Ubuntu : le bras prolongé descend la couche 2 sous 0,30 m sur 29 % des tronçons (p10 0,12 m) et la couche 3 sous 0,50 m sur 19 % (p10 0,32 m), laisse les porosités à 0,60, et abaisse le coefficient cultural de 1,02 à 0,92 (p10 0,74 à 0,71). Toute lecture de paramètres se fait désormais sur la machine de l'entraînement, avec la table qui a servi.
