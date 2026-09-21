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
| 5 bis | Drainage de la couche profonde et nappe libre | Verrou levé le 2026-09-19 ; reste à recaler le débit |
| 5 | Prochaine ronde de modélisation | Un pas par bloc, couches pédologiques, neige, temps de séjour de l'aquifère, poids du terme des variations |
| 6 | Forçage climatique MRCC6 et scénarios | Livrable du plan de travail, dépend de la ronde |
| 7 | Voie positionnelle du champ spatial | Priorité basse, décidée par Essi |
| 8 | Enveloppe du scénario naturalisé | Jugée non nécessaire pour l'instant |
| 9 | Plafond de sous-pas de la colonne | Ouvert le 2026-09-18 ; conditionne la nappe, les pics et le coût |
| 10 | Module de recharge et de nappe | Ouvert le 2026-09-18 ; porte l'identifiabilité et l'étiage sous prélèvement |

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

## 5 bis. Drainage de la couche profonde, et nappe libre — LE VERROU EST LEVÉ, RESTE LE DÉBIT

Ouvert le 2026-09-18 sous le titre « temps de séjour de l'aquifère ». Ce titre était faux et le chantier a changé de nature le 2026-09-19.

Ce qui est écarté. Le temps de séjour de l'aquifère, son coefficient de récession et la forme de sa loi stock-débit ne sont PAS en cause : en les gardant tels quels et en corrigeant seulement le drainage du sol, le maximum de nappe tombe sur le mois observé, l'amplitude de stock atteint 60 mm dans la plage visée de 13 à 130, et la corrélation des anomalies passe de -0,34 à +0,48 avec un puits sur trois au-dessus de 0,5. Le réservoir fonctionnait ; sa nourriture ne suivait pas.

La cause racine. La couche 3 retenait 289 à 465 mm d'eau gravitaire en permanence selon le territoire, cinq à six fois la recharge annuelle, faute d'un exutoire à la bonne constante de temps. D'où aucune capacité disponible à la fonte, une teneur en eau qui ne variait pas de plus de trois pour cent, donc une recharge plate culminant en août, et un sol saturé neuf jours sur dix qui resserrait la condition de Courant au point de laisser 62 % des journées d'avril non calculées. Trois défauts, une cause.

Ce qui reste à faire, dans l'ordre.

Fixer le plafond de percolation du substratum, qui règle le volume sans toucher à la saison : le maximum reste en avril pour tous les plafonds essayés, seul le volume annuel bouge, de 48 mm par an à 0,25 mm par jour de plafond jusqu'à 371 à 4 mm par jour. La plage utile de 0,5 à 2 mm par jour vise les 100 à 250 mm par an plausibles et correspond à un till silteux. Critère : retrouver le KGE du témoin, 0,654 en Outaouais et 0,677 au Saint-Laurent nord-ouest, sans perdre la phase ni l'amplitude de nappe.

Rendre ce plafond SPATIAL, avec un test d'identifiabilité posé sur une grandeur OBSERVÉE. L'indice d'écoulement de base mesuré sur les hydrogrammes par le filtre de Lyne et Hollick vaut 0,58 en médiane sur les 16 stations de l'Outaouais et 0,54 sur les 25 du Saint-Laurent nord-ouest, contre 0,10 et 0,13 produits par le modèle retenu. Sa dispersion est surtout INTRA-régionale, de 0,39 à 0,76 en Outaouais contre 0,04 d'écart entre les deux médianes régionales. Un plafond uniforme ne peut donc pas la reproduire. Le test : la géologie du socle et les dépôts de surface, ingérés le 2026-09-18, expliquent-ils une part de ce facteur deux entre stations voisines ? Si oui, le plafond se prédit par le champ et l'identifiabilité est démontrée sur une observation et non sur un paramètre ajusté. Si non, ce ne sont pas les bonnes covariables. Préalable technique : l'appariement station-tronçon existe à l'exécution du pilote mais n'est enregistré nulle part, il faut le sortir.

Rendre ce plafond SPATIAL et prédit par le champ, à partir des dépôts de surface et de la géologie du socle ingérés le 2026-09-18. C'est le gain d'identifiabilité du chantier : un paramètre qui a un sens physique, une plage connue, et des covariables non redondantes.

Activer le terme de perte sur les niveaux mesurés, livré et branché, et vérifier qu'il tient la phase pendant que le débit se recale.

Reprendre l'audit de fermeture du bilan avec la pile complète, deux termes nouveaux y entrant : le drainage gravitaire et l'évapotranspiration prélevée sur la zone saturée.

Critères d'acceptation du réglage, en cinq points, le dernier étant éliminatoire. Amplitude et phase de la nappe ensemble : corrélation médiane des anomalies mensuelles au-dessus de 0,5, maximum simulé en avril ou mai, amplitude de stock entre 13 et 130 mm. Partition de la production plausible : écoulement de base entre 0,4 et 0,6 du débit, hypodermique entre 0,3 et 0,4. Réduction de l'écart de volume d'avril, qui vaut -7,0 pour cent du total annuel au témoin et se rend en mai, juin et juillet ; c'est la mesure la plus directe puisqu'elle porte sur les débits observés. Ni le KGE ni les cibles satellitaires ne reculent. Et la PART DE JOURNÉE NON TRAITÉE reste proche de zéro : une configuration qui laisse la couche 3 se resaturer retrouve la troncature du schéma, et son KGE remonte alors en revenant au régime d'artefact plutôt que par une meilleure physique. Le plafond de substratum seul récupère 0,646 de KGE avec une part non traitée de 0,236, contre 0,001 pour le drainage gravitaire seul ; les seules configurations saines à ce jour portent la sortie latérale.

Quatre boutons, quatre grandeurs, chacun avec sa prise. La constante de drainage vide la couche. Le plafond de substratum fixe la part d'écoulement de base. Le multiplicateur latéral fixe la part hypodermique. Et le temps de séjour de l'aquifère fixe l'amplitude de la nappe : il avait été écarté à juste titre le 2026-09-18, puisqu'avec une recharge plate il ne crée aucune saison, mais avec une recharge qui culmine en mai il devient le bouton d'amplitude. Calcul du 2026-09-19 : à 89 mm de recharge annuelle, l'amplitude de stock sature vers 37 mm et cent jours de temps de séjour suffisent à placer le maximum en mai avec vingt jours de retard.

Ronde appariée, à faire sur le SAINT-LAURENT NORD-OUEST et non sur l'Outaouais. Mesure du 2026-09-19 : le réseau de l'Outaouais ne transmet que 23 à 27 pour cent de la nervosité produite par la colonne, contre 46 à 55 au Saint-Laurent nord-ouest, 43 pour cent de ses tronçons ayant leur temps de parcours collé à la borne de 48 heures. Juger une correction de la génération sur ce territoire revient à la juger à travers un filtre qui en absorbe les trois quarts.

Prédiction posée d'avance, vérifiable sans passe supplémentaire puisque le paramètre est déjà enregistré. Si la borne de Muskingum est saturée parce que le routage compensait une génération trop brutale, alors avec la colonne corrigée, qui ramène le ruissellement de surface de 0,86 à 0,11, la part de tronçons collés à la borne doit diminuer nettement dans le bras expérimental. Si elle reste saturée, le besoin de retard vient d'ailleurs, vraisemblablement des pseudo-lacs importés comme réservoirs actifs, et les deux chantiers restent indépendants.

Ce qui reste ouvert. Le couplage par étranglement livré le 2026-09-19 n'a aucun effet tant que le drainage est lent, et n'a de sens qu'une fois le drainage gravitaire actif, où il empêchera la couche de se vider dans une nappe déjà haute. La corrélation des anomalies plafonne à 0,48 sans calage. Et l'identifiabilité spatiale de la porosité de drainage n'est toujours pas démontrée.

## 5 ter. Spatialiser le plafond de percolation — REFERMÉ LE 2026-09-21, pour une raison physique

Ouvert le 2026-09-20 après que la mesure eut coupé le chantier en deux.

Ce qui est décidé par la mesure. Sur 95 stations de cinq territoires, chacun prédit par un modèle ajusté sur les AUTRES, la texture du sol explique la part souterraine du débit à +27 % contre le témoin qui prédit la moyenne, la géologie du socle à +3 % donc rien, le relief LiDAR à +11 %. Les six profondeurs de SIIGSOL sont redondantes : trois coordonnées d'une seule suffisent, et les empiler toutes ramène le gain à +17 %. Mais AUCUNE covariable ne prédit la dynamique : l'exposant de récession sort à −8 %, les temps de vidange à +3 % et −19 %. Les attributs de terrain disent combien d'eau passe par le souterrain, pas à quelle vitesse elle revient.

Ce qui est livré. Le plafond de percolation du substratum est la quarante-troisième sortie du champ spatial, construite pour rendre exactement sa référence de 1 mm/jour à sortie brute nulle, donc sans effet sur un ancien point de reprise. Un profil de sol déclaré peut la nommer au lieu de donner un nombre, et le plafond devient une valeur par tronçon. Profil `pile-spatiale` dans `.runs/quebec/config/soil-profiles.toml`.

Ce qui reste. Vérifier en entraînement que le champ apprend effectivement un plafond structuré par la texture, et non un plafond uniforme déguisé ; l'épreuve est le transfert entre territoires, pas le KGE. Décider ensuite du sort de l'exposant et des constantes de temps, qui restent uniformes faute de covariable : soit les laisser ainsi, soit les rendre libres par nœud et les contraindre par les niveaux de puits, qui identifient la présence et la phase du mécanisme sans en identifier les paramètres.

Borne du gain, à dire d'avance : une prédiction PARFAITE de la part souterraine ferait passer l'erreur de transfert de 0,070 à 0,051, sur une grandeur dont l'écart-type vaut 0,080. Le chantier ne peut donc pas rapporter beaucoup, et c'est une raison de le garder petit.

POURQUOI IL SE REFERME, le jour même. Remarque d'Essi : la granulométrie ne gouverne pas le drainage, deux tills de même texture pouvant différer de plusieurs ordres de grandeur en conductivité selon leur COMPACTION. Trois mesures ont suivi et se recoupent.

Le +27 % de la texture était du transfert ENTRE territoires. Repris en validation croisée interne, il tombe à +5 % en moyenne et à −19 % en Montérégie, alors que la dispersion de l'indice d'écoulement de base est à 63 % intra-territoriale. La texture explique donc ce qui distingue les territoires, pas ce qu'un champ doit reproduire.

Le champ spatial, mis devant ce descripteur, a appris un motif POSITIONNEL : la variation de son plafond est expliquée à 0,67 par les seules coordonnées et à 0,30 par la texture. Il ne s'est pas trompé, on lui offrait un signal faible.

Le rang de drainage de l'IRDA, qui décrit la structure et non la granulométrie, est meilleur là où il existe, +4 % contre −19 % pour la texture en Montérégie. Mais sa couverture s'arrête au sud agricole, deux territoires sur quinze au-dessus de la moitié, médiane provinciale NULLE.

CE QUI REMPLACE CE CHANTIER. Quand le facteur de contrôle n'est pas cartographié, un descripteur faible vaut moins qu'un paramètre LIBRE par nœud contraint par des observations. La machinerie existe déjà, l'effet aléatoire additif par nœud, et deux observations la contraignent : les niveaux du réseau de puits et l'exposant de récession mesuré par territoire. Le code livré reste utile, le plafond étant désormais une sortie du champ que l'on peut laisser libre au lieu de la faire prédire.

---

## 5 quater. Fonction de perte : retirer la redondance — OUVERT, l'outil est livré

Ouvert le 2026-09-20 sur une remarque d'Essi : la perte porte le KGE, l'écart quadratique et le biais, soit plusieurs fois la même information.

Mesuré, sans aucune simulation, sur 32 stations et 9 déformations (`.runs/quebec/redondance_perte.py`) : huit termes de débit, mais trois directions indépendantes portent 90 % de leur variation. Le KGE est expliqué à 96,9 % par les autres, l'écart quadratique à 99,1 %, les pics à 98,5 %. Écart quadratique et pics corrèlent à 0,99 : ce sont pratiquement le même terme, pesant 0,1 et 0,5. Seuls le biais de volume et le soutien d'étiage sont distincts, et le soutien d'étiage, plus grand angle mort de la recette à 48,2 %, est à poids NUL. C'est la grandeur que le projet vise.

Livré : `w_r`, `w_beta` et `w_gamma` portent séparément les trois facteurs du KGE, à poids nuls par défaut, exposés par le pilote. À nombre de termes égal, cinq contre cinq, la recette en vigueur laisse 14,2 % de manque moyen et la version décomposée 2,8 %.

Reste : choisir la recette et la mesurer en entraînement. Candidat à égalité de termes, calendrier, volume, amplitude, soutien d'étiage, variations journalières. L'issue à énoncer d'avance : si la version décomposée ne change pas le tenu de côté mais réduit la dispersion entre tirages, elle est retenue, la reproductibilité étant le critère qui a servi pour les contraintes auxiliaires.

---

## 2 bis. NEISIM, neige et apport au sol sur grille — OUVERT le 2026-09-21

Fourni à l'interne par le gouvernement du Québec. Quatre fichiers, 8,4 Go, couvrant tout le Québec méridional de 1980 à 2025, soit quarante-cinq ans.

| Produit | Grandeur | Grille | Pas |
| --- | --- | --- | --- |
| EENEIG, deux versions | équivalent en eau de la neige, mm | 245 × 105, un dixième de degré | 3 heures |
| APPVER, deux versions | apport vertical au sol, mm | 733 × 313, environ trois kilomètres | 1 jour |

Ce que cela change par rapport à CanSWE. La couverture du réseau au sol est très inégale, 76 sites en Outaouais, 30 au Saint-Laurent nord-ouest, 1 en Montérégie et aucun au Saint-Laurent sud-ouest, si bien que la contrainte de neige n'existe aujourd'hui que par endroits. NEISIM couvre tout le domaine.

L'APPORT VERTICAL est le produit le plus intéressant, et ce n'était pas attendu. C'est exactement la grandeur que la colonne calcule en interne et que `SimDiagnostics` expose sous le nom trompeur de `snowmelt`, puisqu'elle contient l'apport total au sol et non la fonte. Une cible gridée sur quarante-cinq ans pour cette variable contraindrait le CALENDRIER de la fonte, qui commande la crue printanière, et pas seulement la masse accumulée. Aucune observation disponible ne le fait aujourd'hui.

RÉSERVE DÉCISIVE. NEISIM est un MODÈLE, pas une mesure, et l'employer comme cible imposerait ses biais aux nôtres. La question préalable, en cours de mesure, est de savoir s'il s'accorde aux relevés au sol du réseau CanSWE là où les deux existent. S'il s'en écarte autant que notre propre modèle, il n'apporte qu'une seconde opinion et n'a pas sa place comme cible. Banc : `.runs/quebec/neisim_contre_canswe.py`.

Le repère de comparaison est connu : le maximum hivernal simulé vaut 121 mm sur l'Outaouais contre 238 mm mesurés au sol, soit la moitié.

---

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
