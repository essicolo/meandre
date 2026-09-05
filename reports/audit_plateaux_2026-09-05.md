# Audit de la flotte du 4 septembre 2026 : plateaux, divergences et défauts de la boucle d'apprentissage

Document d'état au 5 septembre 2026, compilé à partir de cinq relectures indépendantes (anatomie des débits simulés, routage et lacs, colonne de sol, perte et trainer, journaux) et d'un audit externe de crédibilité mené sur les mêmes fichiers. Chaque affirmation chiffrée renvoie à une mesure sur les sorties de la flotte ou à une ligne du code. Le registre des hypothèses (R79 à R83) porte le détail.

## 1. Ce que l'on peut conclure

La flotte du 4 septembre a entraîné quatorze régions du Québec en deux bras, A avec la boucle d'optimisation corrigée (un pas par bloc, historique amorcé) et B avec l'ancienne boucle (un pas par époque), sous la même recette du socle et le même forçage CaSR corrigé. Sur la période d'évaluation indépendante 2022-2024, le KGE médian vaut 0,712 pour A et 0,687 pour B, six régions pour B, cinq pour A, trois à égalité : la boucle corrigée n'apporte rien de mesurable au KGE.

Le KGE ne suffit toutefois pas à qualifier ces modèles. Les hydrogrammes simulés présentent des plateaux, c'est-à-dire des suites de jours où le débit varie de moins de 1 % d'un jour à l'autre, sur 20 à 45 % des jours contre 3 à 14 % dans l'observé, avec des suites de 30 à 143 jours. Un auditeur externe appliquant des règles de forme explicites refuse les quatorze régions. La Gaspésie, meilleur KGE de la flotte (0,819), tire son score du point de reprise de l'époque 0, l'entraînement s'étant effondré dès l'époque suivante ; ce point produit 115 jours plats d'affilée en hiver sur une station et quarante suites de plus de trente jours sur quinze stations.

Deux régimes de plateaux coexistent et n'ont pas la même cause. Les plateaux d'hiver, de décembre à mars, sont communs aux deux bras, présents aux mêmes stations et aux mêmes dates, et l'observé y est lui aussi plat sous la glace (5 à 31 % des jours) ; ils relèvent pour partie du régime physique et pour partie d'observations reconstruites, et ne sont pas tranchés ici. Les plateaux d'été, de mai à novembre, sont propres à certaines solutions : SLSO-B est plat 31,5 % des jours d'été contre 2,0 % pour l'observé, SAGU-B 34,9 % contre 5,2 %, GASP-A 24,6 % contre 7,9 %, tandis que SLSO-A et SAGU-A restent près de l'observé (6,8 % et 16,7 %). Le modèle non entraîné n'a pas ce défaut sur le sous-bassin gaspésien testé (6,3 % de jours plats contre 9,3 % observés). Les plateaux d'été sont donc créés par l'apprentissage, et deux entraînements de la même recette y tombent ou non.

Le mécanisme physique est identifié dans le code : sous la recette du socle, le débit d'été peut être porté par la seule vidange linéaire de la troisième couche du sol, dont la constante de temps vaut 1/krec, soit 417 jours à la borne haute de krec et 2083 jours à sa valeur de référence. Le ruissellement de surface est nul pour toute pluie réelle dès que la conductivité de surface dépasse 0,03 m par jour, et l'écoulement hypodermique, seul chemin rapide restant, s'éteint dès que la deuxième couche sèche ou que sa conductivité est apprise basse. Une pluie d'été percole alors en un jour jusqu'à la troisième couche, y comble le déficit creusé par l'évapotranspiration, et le débit ne bouge pas. La mesure sur les dumps concorde : les plateaux dérivent avec une constante de temps de 300 à 950 jours, montent un jour sur trois, et une crue observée de 45 à 88 % en un jour fait bouger le simulé de 0,2 à 1,6 %.

Les journaux des 28 tâches désignent la boucle corrigée elle-même comme premier facteur des effondrements. Huit journaux du bras A sur quatorze présentent une chute de validation de plus de 0,2 en une époque, contre un seul du bras B ; six de ces neuf chutes surviennent dans des régions sans GRACE ni MOD16. Les termes GRACE amplifient le phénomène là où ils sont actifs : leur part réelle dans la perte, une fois corrigée d'un défaut d'affichage qui la gonfle d'un facteur 146, passe de 12 à 50 % à l'époque 0 à 60 à 90 % pendant les effondrements du bras A, et reste sous 30 % dans les bras B qui ne s'effondrent pas. Les gels de validation, quatre à sept époques identiques à la quatrième décimale, n'existent que dans le bras A, suivent chaque fois un épisode de blocs jetés pour gradient non fini, et le prior constant à trois chiffres prouve que le champ spatial ne bouge plus. Dès la première époque à pas par bloc, le champ s'éloigne des cibles de littérature trois à huit fois plus vite qu'en une époque du bras B : 146 pas d'Adam par époque à un taux réglé pour un seul, dix fois plus sur la couche de sortie et cinquante fois plus sur les lacs.

La perte d'entraînement ne s'oppose pas à cette solution. Le seul terme qui punit directement la platitude, la composante gamma du KGE, avait un gradient dilué d'un facteur cent par l'historique détaché introduit le 3 septembre : il était de fait inerte. Les autres termes, MSE, log-MSE, terme de pics et biais par bloc, ont pour optimum un hydrogramme lissé dès qu'un décalage d'un jour sépare le simulé de l'observé, ou ne voient pas l'été. Dans les régions où GRACE est active, les deux termes de stockage pesaient environ 84 % de la perte, pour un écart quadratique moyen de 160 mm entre stockage simulé et satellitaire, une cible que la colonne ne peut pas atteindre ; c'est dans ces régions que la validation s'est effondrée, tous les termes montant ensemble, ce qui est une divergence et non un compromis.

## 2. Comment c'est mesuré

Sorties : pour chaque région et chaque bras, le débit simulé et observé aux stations, au pas journalier, du 1er janvier 2022 au 31 décembre 2024 (1096 jours), issu du meilleur point de reprise de chaque entraînement ; 184 couples station-bras, 92 stations. Journaux d'entraînement des 28 tâches, avec la validation par époque (2019-2021) et la composition de la perte.

Platitude : part des jours où la variation relative du débit d'un jour à l'autre est inférieure à 1 %, calculée sur les jours où simulé et observé sont tous deux disponibles, puis médiane des stations d'une région. Plus longue suite : nombre maximal de jours plats consécutifs. Niveau du plateau : médiane du débit simulé pendant les jours plats rapportée au débit médian de la station. Rabotage : rapport des maxima annuels simulé sur observé, médiane des années puis des stations. Réponse aux pluies : parmi les jours où l'observé monte de plus de 30 %, part de ceux où le simulé monte aussi. KGE : efficacité de Kling-Gupta avec gamma défini comme rapport des coefficients de variation.

Forme des hydrogrammes par région, médiane des stations, période 2022-2024, A boucle corrigée, B ancienne boucle :

```
région    n   KGE A  KGE B  pointes A  B     plat A   B     obs
abit      3   0,628  0,666   0,83  0,79   15,8  15,3  17,8
cndb      2   0,729  0,788   0,67  0,91   38,3  33,5  12,4
cndc      2   0,705  0,633   0,65  0,51   23,3  18,9  17,9
gasp     15   0,819  0,817   0,84  0,94   32,1  25,8  12,1
mont     23   0,570  0,652   0,81  1,02   10,6   8,2   2,5
outv     16   0,719  0,704   0,85  1,16   19,3  14,9   8,7
sagu     19   0,752  0,732   0,73  0,62   24,6  44,3  14,1
slno     26   0,644  0,666   0,86  1,18   11,0  18,9   5,8
slso     29   0,414  0,665   0,97  0,88   14,8  24,4   3,3
médiane       0,712  0,686   0,84  0,90   17,5  18,9  11,7
```

Plateaux d'été, mai à novembre, part de jours plats et niveau :

```
run      plat sim  plat obs  niveau / médiane annuelle  plus longue suite  suites de 30 j et plus
slso-B    31,5 %     2,0 %          0,90                  102 j            67 sur 29 stations
sagu-B    34,9 %     5,2 %          0,94                  143 j            75 sur 19
gasp-A    24,6 %     7,9 %          0,90                  115 j            40 sur 15
slso-A     6,8 %     2,0 %          0,97                   83 j            19 sur 29
```

Croisement avec les lacs : corrélation entre la part de jours plats et le nombre de lacs en amont 0,04, avec la fraction de lacs 0,08, avec l'aire drainée 0,02 (184 couples). Les 88 couples sans aucun lac en amont ont une part plate de 0,256 contre 0,263 pour l'ensemble. La suite la plus longue de la flotte, 143 jours, est sur un bassin de 121 km² sans lac. L'observé, lui, s'aplatit avec les lacs (0,067 sans lac, 0,135 au-delà de dix lacs), ce que le simulé ne reproduit pas.

Bilan de la perte reconstitué sur la Gaspésie, bras B, pendant l'effondrement de la validation (0,78 à 0,15 des époques 1 à 8) : total de 1,27 à 3,08, prior de littérature de 0,70 à 2,79, biais de 0,55 à 1,43, KGE de 0,29 à 0,68, biais saisonnier GRACE multiplié par 28. La composante GRACE étant accumulée sans le poids de bloc dans le journal, sa valeur affichée (155) doit être divisée par le nombre de blocs (146) : sa contribution réelle vaut environ 1,06 sur 1,27, soit 84 % du total.

Gradient de chaque terme pondéré sur la région gaspésienne, modèle non entraîné, trois premiers blocs, GRACE non atteinte par la trace : norme 37 pour le KGE et 40 pour le terme de pics au premier bloc, 6 pour le biais, 1,5 pour le log-MSE, 0,6 pour la MSE et l'ET, 0,04 pour le prior. Le prior ne pousse pas ; sa montée pendant l'effondrement mesure la dérive des paramètres, elle ne la cause pas.

## 3. Défauts de l'appareil d'apprentissage établis et corrigés le 5 septembre

Chaque défaut est consigné au registre avec sa preuve. Tous sont corrigés dans le dépôt, aucun n'a encore été validé par un entraînement.

Le banc de sous-bassin prenait la branche groupée de la perte, qui calcule le KGE sur le bloc courant seul sans historique, alors que la production passe par la branche par station (R79). Tous les verdicts d'entraînement du banc antérieurs au 4 septembre 14 h 35 sont caducs ; la version 1.0 n'est pas concernée.

Avec des blocs de quinze jours, douze jours vivants restaient après rodage, sous le minimum de trente observations : la perte de débit valait zéro sur 23 blocs sur 24 et le trainer, qui ajoute le prior à chaque bloc, faisait 23 pas par époque sur le prior seul (R80). Le minimum compte désormais l'historique, et un bloc sans donnée ne fait plus de pas.

Le KGE à historique détaché avait un gradient en 1/N par jour vivant contre 1/n pour les termes locaux, puis recevait le poids de bloc n/N : un facteur n/N de trop, environ 1 % (R82). La sensibilité par jour vivant est remise à l'échelle des termes locaux, la valeur imprimée restant celle de la série complète.

Le pilote lisait la profondeur des lacs comme leur surface en km² : 3,5 km² en médiane au lieu de 0,63, jusqu'à 17,8 au 90e centile, donc des coefficients de tarage cinq à trente fois trop petits et des constantes de temps de lac de 12 à 17 jours en médiane au lieu de 0,5 à 1,3 jour, jusqu'à plusieurs années (R81). Corrigé dans le pilote ; sept scripts d'évaluation portent encore la même lecture. Ce défaut aplatit les régions lacustres mais n'explique pas les plateaux d'été, indépendants des lacs.

Sous une boucle juste, sur un sous-bassin et un an d'entraînement, l'optimiseur améliore bien son objectif (KGE de l'année d'entraînement de 0,751 à 0,788 en 144 pas) sans raboter ni aplatir ; l'écart de validation vient d'un conflit de volume de 24 % entre l'année d'entraînement et l'année de validation (R80 bis).

Le pilote imprime désormais, à côté du KGE tenu de côté, la platitude simulée contre observée, la part d'été, la plus longue suite plate, les pointes et le quantile 99, avec un verdict FORME REFUSÉE si la platitude dépasse le double de l'observée ou si une suite dépasse trente jours.

Deux anomalies d'ancrage relevées dans les journaux, communes aux deux bras. L'épaisseur de la deuxième couche du sol est bornée à 0,30 m par la recette alors que le calage Hydrotel de la Gaspésie, de la Montérégie et du Saint-Laurent sud la fixe à 0,157 m : l'ajustement ne peut pas l'atteindre et reste à sa borne. La surface médiane des lacs de la Côte-Nord est (CNDE) vaut 15,2 km² contre 0,24 à 1,57 km² ailleurs, conséquence de la lecture erronée décrite plus haut.

Sur un sous-bassin du Saint-Laurent sud (station 030101, 2021-2024), le modèle non entraîné est déjà déficient : KGE 0,243, gamma 0,478, pointes à 0,41 de l'observé, 13,8 % de jours plats contre 4,1 % observés. Le point de reprise entraîné du bras B y fait mieux, KGE 0,584, gamma 0,874, pointes à 0,76, 10,8 % de jours plats, mais avec une suite plate de 23 jours contre 8. L'apprentissage y a donc corrigé le rabotage et allongé les plateaux, ce qui confirme que les deux défauts ne se corrigent pas ensemble et que le socle non entraîné n'est pas une référence sûre sur toutes les régions. Cette comparaison locale porte une réserve : l'environnement de simulation local diffère de celui de la grappe sur 27 réglages, signalés au chargement.

## 4. Ce que ces résultats ne disent pas

Les dumps ne contiennent que les débits : le réservoir qui porte le stock lent (troisième couche, aquifère, milieu humide) et les paramètres qui diffèrent entre SLSO-A et SLSO-B ne se lisent que dans les points de reprise et les diagnostics internes du modèle. La part des plateaux d'hiver imputable au modèle plutôt qu'aux observations reconstruites sous glace n'est pas mesurée. La contribution respective de GRACE et du taux d'apprentissage à la divergence ne sera connue qu'avec la paire de contrôle sans GRACE, lancée sur la grappe le 5 septembre. Le forçage de Vaudreuil manque du colis de la grappe ; la région n'a pas été entraînée.

## 5. Tests à faire, par ordre de coût

Colonne isolée, trois minutes chacun : balayage de krec et de K_sat_2 sur un été réel, puis de K_c et du seuil de stress, en mesurant la part de la production hypodermique, la fraction d'une pluie de 20 mm restituée en cinq jours et le nombre de jours plats. Sous-bassin, trois minutes : simulation avec le point de reprise SLSO-B puis SLSO-A, comparaison des 37 champs de paramètres et des diagnostics de production pendant le plateau du 15 mai au 22 août 2024 à la station 022704. Grappe, deux heures : paire de contrôle gaspésienne sans GRACE. Scripts : aligner les sept lectures de troncon.trl sur celle du chargeur. Garde-fous : test de tendance sur la perte et repli sur la forme à chaque validation, indépendants des époques de grâce de l'autopilote.
