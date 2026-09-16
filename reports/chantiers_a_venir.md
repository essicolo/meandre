# Chantiers à venir de méandre

Ouvert le 2026-09-08, remis en ordre le 2026-09-16. Ce document liste les chantiers identifiés mais non entrepris, avec leur raison d'être, leur coût et le test qui dira s'ils tiennent. Le registre des hypothèses dit ce qui est vrai aujourd'hui ; le journal des expériences raconte ce qui a été fait. Un chantier n'entre ici qu'avec un critère de réussite mesurable.

## Ordre de priorité au 2026-09-16

Prochain livrable le 2026-09-30, deux semaines après la présentation. Aucune simulation globale avant : les deux premiers chantiers préparent des données, et la prochaine ronde de modélisation les intègre en une seule fois.

| rang | chantier | raison du rang |
|---|---|---|
| 1 | Couches pédologiques de l'IRDA et gestion agricole de l'eau | Sol trop peu décrit pour les petits bassins ; prépare la ronde |
| 2 | Neige mesurée au sol | Seule cible hivernale fiable ; prépare la ronde |
| 3 | Enveloppe probabiliste après 2022 | Couverture de 0,58 au Saguenay et 0,66 en Abitibi |
| 4 | Coût de la simulation provinciale | Rend la ronde et les scénarios moins chers |
| 5 | Prochaine ronde de modélisation | Un pas par bloc, couches pédologiques, neige, poids du terme des variations |
| 6 | Forçage climatique MRCC6 et scénarios | Livrable du plan de travail, dépend de la ronde |
| 7 | Voie positionnelle du champ spatial | Priorité basse, décidée par Essi |
| 8 | Enveloppe du scénario naturalisé | Jugée non nécessaire pour l'instant |

## 1. Couches pédologiques de l'IRDA et gestion agricole de l'eau

Pourquoi. Le champ spatial ne reçoit que trois pourcentages granulométriques pour décrire le sol, et deux petits bassins voisins reçoivent presque sûrement les mêmes. Le KGE médian vaut 0,526 sous 100 km² contre 0,763 entre mille et trois mille. Sur 69 stations couvertes par l'IRDA, le biais de volume suit le matériau parental, que le modèle ne reçoit pas. Sa corrélation de rang avec la part de till vaut -0,52 à aire contrôlée.

Ce que l'IRDA donne. La couverture pédologique au 1:20 000 porte la classe de drainage naturel, le matériau parental et l'affleurement rocheux. D'autres couches de l'IRDA sont à inventorier et à ingérer par la même voie.

La couverture est partielle et inégale selon la région. En moyenne par tronçon, elle vaut 86 % au Saint-Laurent sud-ouest, 47 % en Montérégie et 32 % en Outaouais aval. Elle tombe à 21 % en Gaspésie, 11 % au Saint-Laurent nord-ouest et 7 % au Saguenay. Elle est nulle ou presque sur la Côte-Nord, en Abitibi et en Outaouais moyen. Chaque couche entre donc comme une composition par tronçon : parts de chaque classe, plus une part non cartographiée. Une variable mise à zéro là où la donnée manque confondrait sol inconnu et classe absente. L'absence n'est pas aléatoire, l'IRDA ayant surtout cartographié les terres agricoles.

Épreuves du 2026-09-16, sans simulation. Le croisement comptait mal l'aire des unités hydrologiques en plusieurs morceaux ; corrigé. Depuis le géopaquet 2026_01, texture en ilr, drainage, perméabilité, contact lithique et sols organiques sont calculés par tronçon. Les attributs actuels ne les prédisent pas, R² hors bloc spatial de 0,09 à 0,37, et 40 % des voisins que les attributs actuels confondent sont distingués. Structure et groupe hydrologique sont dérivés de la texture et de la perméabilité, donc retirés. Ajouter les colonnes avec des poids nuls laisse le modèle intact, et leurs poids reçoivent un gradient.

Gestion agricole de l'eau. La classe de drainage est celle du sol naturel, pas la présence de drains souterrains, posés surtout dans les sols mal drainés. En Montérégie, le drainage artificiel, les bassins de rétention et l'irrigation modifient la réponse des petits bassins. Quatre des six stations les moins bien reproduites y sont, entre 46 et 246 km². Ces pratiques demandent une autre source, par exemple la superficie drainée du Recensement de l'agriculture, à vérifier.

Critère de réussite. La carte des classes se lit dans le champ appris, et le KGE médian des bassins de moins de 100 km² progresse sans dégrader les autres. Contrôle du masque : masquer des zones cartographiées ne change pas la simulation plus que l'écart entre deux graines.

## 2. Neige mesurée au sol

Pourquoi. Le débit d'hiver observé est reconstruit sous glace sur 49 à 95 % des jours et ne peut pas servir de cible. La masse de neige mesurée au sol le peut. La cible CanSWE est construite, rattachée au réseau et vérifiée depuis août 2026, mais son poids vaut zéro.

La cible est la masse, pas le calendrier. Sur quatre sites du Saguenay et six hivers, le champ non entraîné rend 0,75 de la neige observée en janvier, 0,66 en mars et 1,00 en avril. La date de disparition est correcte ; c'est l'accumulation qui manque, d'un tiers au maximum hivernal.

Couverture. Sites avec au moins cinq relevés en période de fonte : Outaouais 76, Saint-Laurent nord-ouest 30, Saguenay 22, Gaspésie 22, Côte-Nord B 21. Les autres régions en ont peu : Abitibi 7, Côte-Nord C 5, Montérégie 1, Saint-Laurent sud-ouest 0. Un indicateur territorial étend la cible aux régions sans station. Un modèle à gradient boosté prédit la masse au maximum avec 0,77 de variance expliquée, validé en retirant des stations entières.

Représentativité. Entre stations voisines à moins de 50 km, la masse au maximum diffère de 21 % en médiane. Le poids ne peut pas être uniforme : la densité de sites varie d'un facteur cent entre régions.

Ce qu'il faut mesurer avant de l'activer. Une paire appariée sur l'Outaouais, avec et sans la contrainte, jugée sur la masse, sur le calendrier de la crue printanière et sur l'évapotranspiration, jamais sur le KGE seul.

Critère de réussite. La masse simulée se rapproche des relevés sans dégradation de l'évapotranspiration satellitaire ni du stockage gravimétrique.

## 3. Enveloppe probabiliste après 2022

Pourquoi. Sur 2022-2024, la tête de quantiles couvre 0,85 de l'intervalle annoncé à 0,90 et 0,42 de celui annoncé à 0,50. Le diagramme de Talagrand est plat sauf sa dernière classe, à 9,7 % contre 5 attendus. Deux régions décrochent des deux côtés. Au Saguenay, 18 % des observations tombent sous le quantile à 5 % et 24 % au-dessus de celui à 95 %. En Abitibi, ces parts valent 18 % et 16 %. Une enveloppe ajustée avant 2022 ne couvrait déjà que 0,61 au Saguenay : le changement de régime est la cause probable, non établie.

Ce que le chantier demande. Séparer l'effet du régime de celui de la tête. Ajuster la tête sur 2019-2021 puis la juger sur 2022-2024, contre un ajustement sur 2001-2018. Si le régime domine, conditionner la tête sur une variable du climat récent plutôt que sur les seuls paramètres du champ.

Critère de réussite. Couverture de l'intervalle à 90 % entre 0,85 et 0,95 dans chaque région, et dernière classe du diagramme de Talagrand sous 7 %.

## 4. Coût de la simulation provinciale

Pourquoi. Sur la RTX 2000 du poste, l'évaluation d'une région prend 18 minutes, dont 54 secondes de mise en place. Chronométré sur LABI le 16 septembre, ce temps est le même avec les données sur le disque Windows ou sur celui de WSL. Les lectures ne sont donc pas en cause. Une mesure du 15 septembre inscrite dans la colonne attribue 8,5 minutes à la compilation du sol et 5,5 minutes à la simulation. La compilation est refaite pour chaque taille de région.

Ce que le chantier demande. Mesurer LABI deux fois de suite avec `MEANDRE_COMPILE_DYNAMIQUE=1` et un même cache de compilation, pour séparer compilation et simulation. Puis mesurer si deux régions tiennent ensemble sur le GPU, dont l'utilisation plafonnait à 33 %.

Critère de réussite. Le Québec entier, scénario géré, en moins de deux heures sur le poste, à KGE identique au millième.

## 5. Prochaine ronde de modélisation

Pas avant le livrable du 2026-09-30 : une simulation globale ne se relance qu'une fois les couches pédologiques et la neige mesurée prêtes, pour n'en payer qu'une.

Protocole d'optimisation.

Pourquoi. Les modèles du 15 septembre ont été entraînés sans `MEANDRE_PAS_PAR_BLOC=1`. Le pilote fait alors un seul pas d'Adam par époque, sur les gradients accumulés de tous les blocs. Trente époques valent trente pas, dont cinq à taux réduit par la montée progressive.

La tête de quantiles entraînée ainsi le 16 septembre n'avait bougé que de 0,004 depuis son initialisation. La couverture de l'intervalle à 90 % valait 0,37. Avec un pas par bloc et un taux de 1e-2, elle vaut 0,85 sur 2022-2024.

Le modèle déterministe a subi le même protocole. Ses paramètres restent donc proches de leur départ : calage d'Hydrotel pour le sol, valeurs de littérature ailleurs. Les verdicts du 15 septembre sur le terme des variations restent des comparaisons appariées valides, mais ils portent sur des modèles presque non entraînés.

Ce que la ronde demande. Refaire les deux versions, avec et sans le terme des variations, avec un pas par bloc sur les quatorze régions, et y intégrer les chantiers 1 et 2. Entraîner ensuite la tête de quantiles sur chacune. Le script `reference_variations.sbatch` ne demande que l'ajout de la variable. Le registre avertit que ce réglage change les paramètres physiques obtenus : l'épreuve doit donc garder les modèles actuels comme témoins.

Coût. Une nuit sur Narval pour le modèle déterministe, une heure pour la tête.

Critère de réussite. Le KGE médian par région ne recule pas, la plus longue suite plate reste sous 35 jours en médiane, et le rapport des pointes reste entre 0,8 et 1,2. Les paramètres appris s'écartent de leur départ, mesuré sur la norme des lignes de sortie du champ.

Poids du terme des variations d'un jour.

Pourquoi. À un poids de 1,0, le terme raccourcit les plateaux d'hiver et relève le KGE dans sept régions sur neuf. Il abaisse toutefois les pointes là où elles étaient correctes : de 1,04 à 0,81 en Outaouais aval et de 0,86 à 0,72 au Saguenay. Un poids plus faible conserverait vraisemblablement le gain sur les plateaux.

Ce que la ronde demande. Ces mesures portent sur des modèles à trente pas : le poids se rejuge dans la ronde, d'abord sur l'Outaouais aval et le Saguenay.

Critère de réussite. Dans ces deux régions, les suites plates ne rallongent pas et les pointes simulées valent au moins 0,8 de l'observé.

## 6. Forçage climatique MRCC6 et scénarios

Pourquoi. Le plan de travail prévoit une modélisation exploratoire selon des scénarios de prélèvements et de rejets. Le Modèle régional canadien du climat de sixième génération, développé à l'UQAM avec Ouranos, est la source régionale naturelle pour le Québec.

Ce qu'il faut comprendre avant de commencer. Un modèle climatique reproduit la statistique d'un climat, pas la météorologie d'une journée donnée. Il ne sert donc ni à l'entraînement ni à un calcul d'efficacité contre des débits observés. L'usage correct est d'entraîner sur la réanalyse CaSR corrigée, puis de simuler avec les sorties climatiques.

Ce que le chantier demande. Obtenir les six variables de la colonne au pas journalier, les interpoler aux tronçons, puis corriger leur biais contre CaSR sur une période commune. Cette correction est l'étape déterminante, le forçage fixant le plafond de performance.

Critère de réussite. Les signatures hydrologiques simulées sous le climat historique du modèle régional, après correction, sont compatibles avec celles obtenues sous la réanalyse.

Ce que la différentiabilité apporte. La dérivée du débit par rapport à chaque paramètre et à chaque variable de forçage est exacte. Une projection peut donc attribuer son changement à ses causes.

## 7. Voie positionnelle du champ spatial

Pourquoi. La voie des attributs interpole, mais celle de la position extrapole, et l'encodage de Fourier est périodique. Ses deux bandes les plus fines, de 150 et 75 km, sont plus courtes qu'une région.

Mesuré le 2026-09-14. Au transfert du Saguenay vers la Gaspésie, déplacer les positions de 277 km en médiane change les paramètres de 0,5 % en médiane. Les trois conductivités à saturation font exception, de 3,7 à 6,6 %. Le gradient attribue pourtant 45,7 % de la sensibilité à la position : le réseau s'appuie sur elle, mais la surface apprise est plate à grande échelle.

Priorité basse, décidée par Essi le 2026-09-14. À reprendre si un transfert lointain échoue sans explication. Le script est `audit_position.py`.

## 8. Enveloppe du scénario naturalisé

La passe naturalisée du pilote rejoue la simulation avec les prélèvements à zéro et les mêmes poids. Elle n'écrit pas les quantiles, et le second étage ne la déclenche pas. Une naturalisation probabiliste demanderait d'ajouter les quantiles à cette passe puis de reprendre le second étage : environ une heure de travail et une de calcul. Essi l'a jugée non nécessaire le 2026-09-15.

## Chantiers fermés le 2026-09-16

Fonction objectif non paramétrique. Mesurée sans entraînement le 2026-09-12, la version non paramétrique du KGE tolère davantage le lissage que la version usuelle. Le rabotage des pointes n'existe par ailleurs que dans les basses-terres, et le terme des variations y relève les pointes de la Montérégie de 0,74 à 0,98. La justification première du chantier est tombée.

Plateaux d'hiver. La perte rejette déjà un hiver figé sur toutes les stations. Ajouter le terme des variations fait passer la plus longue suite plate de 60 à 35 jours en médiane. L'absence des paramètres de gel dans les anciens points de reprise n'expliquait rien.

Cause du gradient non fini. Close le 2026-09-08 : la perte posait des NaN dans le débit simulé pour ignorer les observations manquantes, et leur rétropropagation contaminait le champ. Corrigé, test de non-régression écrit.

Modèle trop lissé et temps de transfert du routage. Un temps de transfert fixé par la longueur et la pente de chaque tronçon, éprouvé sur Narval le 2026-09-15, corrige les pointes. Il dégrade en revanche le KGE, la platitude et les suites plates. Le lissage ne concerne que les basses-terres ; ailleurs, le modèle est trop nerveux.

Erreur d'amplitude du forçage. La précipitation cumulée avant chaque pointe n'a pas de biais mesurable alors que le pic simulé vaut la moitié de l'observé. Le déficit est dans la transformation pluie-débit, et le rabotage qui motivait le chantier est localisé aux basses-terres.

Écoulement rapide d'été. Le modèle déployé produit déjà 53 % de son écoulement estival en ruissellement de surface, et l'excès d'infiltration sous-journalier n'ajoute que 0,6 point. Le canal de durée d'averse reste disponible, désactivé par défaut.
