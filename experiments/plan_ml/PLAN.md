# Plan d'expériences : apprentissage automatique à l'intérieur de la physique de meandre

EN COURS, NE PAS SUPPRIMER. Rédigé le 2026-10-22 à partir de trois rapports (annexes 1 à 3 : inventaire du code, littérature, observations) et de l'épreuve du correcteur aval (experiments/gru_residual, registre R389).

## 1. Ce qui cadre le plan

Décision d'Essi : entre empirique et physique, la physique d'abord ; l'empirique comble les trous de la physique, et des relations empiriques déjà présentes dans la physique peuvent gagner à devenir non linéaires par des modèles simples.

Ce que la littérature établit (annexe 2). Une seule place du ML dans la physique a une preuve de généralisation à des régions jamais vues : des paramètres STATIQUES spatialisés par réseau depuis les attributs, avec un modèle conservatif en aval (Feng et al. 2023 : +0,07 de NSE et +0,12 de KGE sur le LSTM quand des régions entières sont retirées, alors qu'en interpolation spatiale les deux font jeu égal). Deux autres places ont des preuves plus faibles mais réelles : une fraction ou un stress borné entre 0 et 1 qui multiplie une demande physique (évaporation, GLEAM4), et un partage de l'eau liquide par fractions à somme un (Kraft et al. 2022). À l'inverse, des paramètres rendus VARIABLES DANS LE TEMPS par un réseau récurrent compensent une structure fausse autant qu'ils améliorent la vraie (Acuña Espinoza et al. 2024 : un modèle volontairement absurde gagne 0,29 de NSE), ce qui retire à la physique son rôle de garde-fou. Le partage pluie-neige par réseau ne gagne rien sur un seuil à deux variables (Jennings et al. 2025). Les LSTM régionaux (Arsenault et al. 2023) prouvent l'interpolation sur les mêmes années, pas l'extrapolation. Aucune publication n'a évalué un réseau de coordonnées (Fourier, NeRF) pour spatialiser des paramètres hydrologiques ; le risque documenté est la bonne interpolation en domaine et la mauvaise généralisation spatiale.

Ce que meandre a mesuré (annexes 1 et 3, registre). Un correcteur récurrent en aval gagne au plus +0,03 de KGE médian aux stations retirées, toujours au prix de l'étiage ou du volume, et les attributs de terrain y dégradent le transfert (R389). Dix champs du réseau spatial ont une sensibilité nulle et sept sorties n'ont aucun lecteur dans le code (annexe 1). Le gradient ne voit pas au-delà d'un bloc de 90 jours (R365) : les lois des réservoirs lents ne s'apprennent pas par le débit journalier. Les observations non circulaires qui peuvent juger une pièce sont peu nombreuses : CanSWE pour la date de fonte et l'accumulation, les récessions observées pour l'exposant de la nappe, les puits pour la présence et la phase des mécanismes souterrains, MOD16 en forme hors neige pour K_c, le bilan P − Q pluriannuel pour le niveau de l'ET. Les aires saturées, la porte de gel, le tarage des lacs et la demande évaporative n'ont pas de juge direct, ou un juge non ingéré (SWOT, niveaux des jauges, tour de Chibougamau).

## 2. Règles d'admission d'une pièce apprise

1. Elle remplace une loi existante ou un coefficient calé, à l'endroit exact où la physique lit la valeur ; jamais le débit final, jamais une source d'eau.
2. Elle est bornée et monotone là où la physique l'exige, et s'écrit comme la loi actuelle plus un écart initialisé à zéro : la passe avant de départ est identique au modèle actuel, et le test de fidélité au binaire Hydrotel reste au bit près à l'initialisation.
3. Elle est statique dans le temps par défaut. Une dépendance au temps n'entre que par des variables d'état ou de forçage explicites (température, stock, saison), jamais par une mémoire récurrente libre, sauf si l'épreuve 3.6 montre que les cumuls ne suffisent pas.
4. Elle a un juge indépendant du débit, ou à défaut une statistique du débit observé calculée sans simulation.
5. Elle passe la sonde de gradient et le banc de sensibilité avant toute épreuve au banc : une loi que rien n'observe n'est pas candidate.
6. Elle entre au catalogue déclaré (TOML, settings.py), à côté de la loi d'origine.

## 3. Épreuves communes, par ordre croissant de coût

Chaque candidate traverse ces épreuves dans l'ordre ; un échec arrête la candidate.

3.1 Statistique sur l'observé, sans simulation (une minute). La loi laisse-t-elle une signature dans les observations ? Exemples déjà écrits : réponse du débit aux pluies sur sol gelé (88 stations), volume d'événement par orage (6 042 orages, 132 stations), récessions de Brutsaert-Nieber par station, décalage pluie-débit par station.

3.2 Sonde de gradient et banc de sensibilité (une à cinq minutes, ETL_EPOCHS=0). La dérivée de chaque observable par rapport à la loi est-elle non nulle, et dans quel sens pousse chaque terme de la perte ? MEANDRE_SONDE_GRADIENT=1 au banc de sous-bassin ; MEANDRE_SENSIBILITE=1 au pilote.

3.3 Ajustement hors ligne de la fonction contre son juge (minutes). La forme non linéaire est apprise directement sur l'observation qui la juge, avec retrait de sites ou de stations et retrait d'années, exactement comme la phénologie a été calée sur MODIS sans simuler (R206). Critère : gain hors échantillon sur un témoin paramétrique (la loi actuelle avec coefficient réajusté). Le gain se mesure en erreur de prédiction, pas en part de variance.

3.4 Passe avant à poids gelés (trois à cinq minutes par territoire). La fonction ajustée est posée dans la colonne ; effet sur la perte complète, l'étiage, le volume et le bilan de masse (ETL_BILAN=1), avec et sans diagnostics.

3.5 Test de structure fausse (une époque courte au banc, environ une heure). La pièce est posée sur une variante volontairement fausse de la loi (par exemple exposant 2 au lieu de 8, ou réservoir linéaire). Si elle améliore autant la variante fausse que la vraie, elle compense au lieu de combler ; refusée.

3.6 Cumuls contre mémoire (hors ligne, minutes). Quand une dépendance au temps est envisagée, comparer un réseau sans état nourri de cumuls (1 à 90 jours) à un GRU, sur le même juge. Deux fois déjà, le GRU n'a rien ajouté (évapotranspiration en juillet ; correcteur aval en octobre).

3.7 Conditionnement spatial (hors ligne, minutes). Trois variantes de la pièce : poids partagés sans conditionnement ; conditionnée par les attributs de terrain ; conditionnée par le champ spatial (coordonnées de Fourier et latents). Jugée aux stations ou sites retirés, ET par retrait de territoires entiers. Ce qui gagne aux stations retirées mais perd aux territoires retirés est une interpolation, pas une loi.

3.8 Banc de sous-bassin (une à deux heures par variante). Trois graines, dernière époque, toute la période 2020-2024, perte complète ; dérive des paramètres physiques comparée à la version sans la pièce.

3.9 Territoire entier, puis province (demi-journée à une journée). Seulement si 3.1 à 3.8 sont passés et si la phrase « ce que seule cette épreuve dira » est écrite.

## 4. Phase 0, préalables sans entraînement (une journée de machine, aucune pièce apprise)

| # | Épreuve | Pourquoi d'abord | Coût | Ce que chaque issue change |
|---|---|---|---|---|
| 0.1 | Routage operator contre operator-lagged, passe avant à poids gelés, Outaouais puis Abitibi | Le retard de 0,75 à 1,1 jour aux stations à trois lacs en série ou plus (R389) est un artefact probable du mode lagged ; toute pièce apprise sur le routage apprendrait ce retard | 5 min par territoire, après stabilisation d'etl_run.py | Si le retard disparaît : changer le mode dans la recette, relancer le plan de correction du tarage sans ce biais. Sinon : le retard est dans le tarage des lacs, candidate 5.5 remonte |
| 0.2 | Relancer sensibilite_des_observables.py sur la physique d'octobre (Outaouais, Montérégie) | Le dernier résultat (dix champs muets, trois directions) date du 22 septembre, avant le sol semi-implicite et la couche 3 déclarée | 10 min par territoire | Donne la liste à jour des lois que quelque chose observe ; toute candidate sur un champ muet est écartée |
| 0.3 | Audit des sorties du champ sans lecteur (frost_alpha, interception_capacity, T_snow, rain_hours, vg_n, alpha_T, vsa_b) | Une pièce greffée sur une sortie morte n'a pas de gradient ; en outre le réseau dépense de la capacité à les calculer | 1 h de lecture | Liste des sorties à retirer ou à brancher ; vsa_b à brancher à bv3c2.py:167 si la candidate 5.3 passe |
| 0.4 | Ingestion des niveaux aux jauges du CEHQ (fichiers _N) et de la géométrie des tronçons pour SWOT dans le banc de sensibilité | Les seuls juges du tarage des lacs et du routage ; le code du banc est prêt | demi-journée | Sans eux, les candidates 5.5 et 5.6 n'ont pas de juge et ne sont pas lancées |

## 5. Candidates, par ordre d'épreuve

Pour chacune : forme, juge, épreuves, critère d'arrêt, et ce que l'issue change. Les coûts sont ceux des épreuves 3.1 à 3.4 ; le banc (3.8) ne vient qu'après.

### 5.1 Fonte de la neige

Loi actuelle : fonte = C_f · degrés-jours · indice de rayonnement · (1 − albédo) · (1 + 0,5 sin saisonnier), C_f par tronçon. Défauts : l'amplitude saisonnière déplace l'excès d'hiver sans le régler (R294, R356) ; le manteau simulé dépasse NEISIM de 16 à 36 % au sud et de 100 % en Gaspésie (R319, R324).
Forme proposée : fonte = C_f · g(T, rayonnement, albédo, âge du manteau, classe de couvert), g positive, croissante en T, nulle sous T_melt, initialisée pour reproduire la loi actuelle ; sans amplitude saisonnière calée, que g doit retrouver si elle existe.
Juge : CanSWE apparié site et jour (baisse de masse un jour sans précipitation = ablation mesurée ; 20 998 jours-sites utilisables au Saguenay) ; NEISIM par nœud là où le réseau se tait, en arbitre seulement.
Épreuves : 3.3 hors ligne sur le module neige seul (il se simule sans sol ni routage, cibles déjà construites par build_neisim_targets.py), retrait de sites et retrait d'années ; 3.6 cumuls contre GRU ; 3.7 trois conditionnements ; 3.4 passe avant.
Critère d'arrêt : le registre dit qu'il reste environ 0,11 de part de variance expliquée à gagner sur l'intensité de fonte (la forêt aléatoire capte les deux tiers du captable). Si g ne dépasse pas la loi actuelle réajustée de 15 % d'erreur de prédiction aux sites retirés, la candidate s'arrête.
Ce que l'issue change : succès, g entre au catalogue de la neige et la Gaspésie est le premier territoire relancé ; échec, l'excès de neige est un problème de forçage ou de représentativité, pas de loi de fonte, et le partage pluie-neige n'est pas candidat (Jennings 2025).
Coût : une demi-journée.

### 5.2 Loi stock-débit de la nappe

Loi actuelle : Q_b = k_b (h/h_ref)^2, exposant 2 uniforme. Mesuré : exposant de récession b = 1,60 en médiane sur 76 stations (n = 2,27), de 1,36 à 4,65 selon le territoire, variable avec la saison (R147, R213) ; aucune covariable ne le prédit ; les trois scalaires rendus apprenables ne bougent pas sous le débit (R237) parce que le gradient ne voit pas au-delà d'un bloc.
Forme proposée : Q_b = k_b · g(h ; k_gw, saison), g croissante et convexe, g(h_ref) = 1, initialisée sur le carré ; k_gw est le champ déjà ancré sur les récessions observées.
Juge : courbes de récession observées sans recharge, par station (Brutsaert-Nieber), en forme et courbure ; puits RSESQ en forme pour la phase.
Épreuves : 3.1 déjà faite ; 3.3 ajustement de g sur les récessions, stations retirées et années retirées, témoin = puissance réajustée ; 3.4 passe avant sur l'étiage (Q7min, jours sous Q90) ; 3.5 structure fausse avec réservoir linéaire.
Critère d'arrêt : si g ajustée hors échantillon ne bat pas la puissance réajustée par station, il n'y a pas de forme commune à apprendre, et l'exposant reste une ancre mesurée par station (ETL_NAPPE_EXP=mesure).
Ce que l'issue change : succès, g figée dans la recette (apprise hors ligne, jamais par le débit journalier) ; échec, on garde l'ancre mesurée et la question de l'étiage se reporte sur la conductance spatiale (R380).
Coût : une demi-journée.

### 5.3 Part de ruissellement des aires saturées

Loi actuelle : part = (theta1/thetas1)^8, exposant uniforme lu des réglages ; le champ vsa_b n'a aucun lecteur. Mesuré : volume d'événement simulé sur observé 0,64 (0,32 en Montérégie) ; au banc, b = 6 rend la platitude et relève la corrélation à volume égal, non confirmé au pilote (R386) ; hydrogramme trop amorti partout (R389).
Forme proposée : part = g(theta1, theta2, pente, fraction de milieux humides, texture) dans [0, 1], croissante en theta, initialisée sur la puissance 8, branchée au point exact où bv3c2 lit la part.
Juge : aucun juge direct n'existe (annexe 3) ; à défaut, volume et temps de montée par orage d'été sur 132 stations, statistique observée déjà écrite, et variations journalières du débit MESURÉ par temps sec ; en appoint statique, l'humidité LiDAR et les milieux humides détaillés, jamais confrontés à une sortie du modèle. C'est la candidate la plus exposée au risque d'ajustement, d'où le test de structure fausse obligatoire.
Épreuves : 3.2 sonde ; 3.3 ajustement hors ligne de g sur les volumes d'événement, la colonne à poids gelés fournissant theta ; 3.4 passe avant avec bilan ET (la part retire de l'eau au sol, donc à l'ETR) ; 3.5 structure fausse avec exposant 2 ; 3.7 conditionnement.
Critère d'arrêt : si le rapport des volumes d'événement aux stations retirées ne se rapproche pas de 1 d'au moins la moitié de l'écart, ou si l'ET annuelle dérive de plus de 5 %, arrêt.
Ce que l'issue change : succès, brancher vsa_b ou g au catalogue et relancer la Montérégie entière (le défaut y est le plus fort) ; échec, l'amortissement vient du routage (candidate 5.6) ou du forçage d'orage.
Coût : une journée.

### 5.4 Porte de gel sur l'infiltration

Loi actuelle : trois seuils durs (10 mm de neige, facteur 0,5 sur la percolation, rampe linéaire en profondeur de gel). Mesuré : la pluie sur sol gelé fait monter le débit 1,8 fois plus, la règle tout ou rien est fausse en amplitude (R224) ; aucune des trois portes ne convient à deux bassins à la fois (R212).
Forme proposée : facteur f(profondeur de gel, équivalent en eau de la neige, theta1) dans [0, 1], décroissant en profondeur de gel, initialisé sur la rampe actuelle, posé sur l'infiltration et sur les échanges, remplaçant le seuil des 10 mm.
Juge : statistique de réponse aux pluies de novembre sur sol gelé par station (sans simulation) ; puits en hiver ; débits d'hiver aux seuls jours mesurés, parce que 60 à 87 % des valeurs de décembre à mars sont reconstituées (R19, R385).
Épreuves : 3.1 déjà écrite ; 3.2 ; 3.3 sur la statistique de réponse ; 3.4 ; 3.5.
Critère d'arrêt : la profondeur de gel n'est mesurée nulle part dans nos sources ; si la réponse observée ne se distingue pas par classes de gel simulé (3.1), la loi n'a pas de juge et s'arrête.
Coût : une demi-journée.

### 5.5 Tarage des lacs et retard des lacs en série

Loi actuelle : Q = k (S/A)^β A, β centré sur 1,5, sans saison ; mode lagged, un jour de retard par lac traversé.
Préalable : 0.1 et 0.4. Si le mode operator supprime le retard, la seule question restante est β saisonnier (glace, régulation) ; sinon la loi elle-même.
Forme proposée : Q = k_anc · g(h, saison), g croissante en h, initialisée sur h^1,5.
Juge : niveaux aux jauges du CEHQ (fichiers _N), altimétrie SWOT (1 008 tronçons, 19 directions indépendantes contre 8 pour toutes les sources employées, R176), retard pluie-débit aux stations aval.
Épreuves : 3.1 décalage par station (fait) ; 3.2 sensibilité SWOT ; 3.3 ajustement sur les niveaux ; 3.4.
Critère d'arrêt : sans niveaux ingérés, pas d'épreuve.
Coût : une journée après 0.4.

### 5.6 Temps de transfert et forme de la réponse rapide

Loi actuelle : Muskingum à K constant par tronçon, borné [4, 48] h. Mesuré : K vaut 15 à 26 fois le temps physique (R62) ; libre, le routage s'effondre (R245) ; une tête sur attributs gagne 0,03 à 0,04 mais dégrade l'étiage et déstabilise (R297-R300) ; variation journalière simulée 0,16 contre 0,22 observée.
Forme proposée : K_eff = K · g(Q/Q_ref, saison), g bornée, initialisée à 1 (la célérité en débit, désactivée, en est la forme paramétrique) ; alternative, un hydrogramme de versant à deux paramètres par tronçon (loi gamma, comme Feng 2022), à somme un donc conservatif.
Juge : décalage pluie-débit et propagation entre stations en série, jamais le KGE (retard contre atténuation non identifiables par une station, R205).
Préalable : 0.1 ; la candidate n'est lancée que si l'amortissement subsiste après correction du retard des lacs.
Épreuve 3.3 faite (2026-10-23, après-midi). La forme observée est plus rapide et plus concentrée que la simulée, mais elle varie quatre fois plus d'un orage à l'autre qu'entre le modèle et l'observé, et aucun noyau linéaire commun ne transfère (voir RESULTATS.md). Candidate arrêtée : le défaut est une absence de dépendance à l'état dans le transfert versant-rivière, où ni l'exposant des aires saturées (épreuve déjà faite) ni un noyau commun ne la portent. Le remède à chercher n'est donc pas un retard mais une concentration dépendante de l'état de l'orage.
Coût : une journée.

### 5.7 Demande évaporative et stress (plus tard)

Loi actuelle : Penman FAO-56 à résistance de surface fixe, fois K_c par tronçon, fois phénologie calée sur MODIS ; stress racinaire jamais actif (ETR/ETP 0,91 à 0,96, R341). Défauts les plus documentés du registre ; forme naturelle : une fraction de stress g_s(déficit de vapeur, rayonnement, T, theta, LAI) dans [0, 1] multipliant la demande, comme GLEAM4.
Mais le juge est le problème : MOD16 est lui-même un Penman-Monteith, trop haut de 8 à 24 % aux conifères boréaux canadiens (Wang et al. 2015 ; +24 % à Chibougamau), fiable en forme d'été seulement ; le bilan P − Q ne donne qu'un niveau annuel. Décision sur MOD16 (annexe 3) : le garder comme contrainte de forme hors neige sur K_c, où il divise la dispersion entre graines par quatre à sept et où aucune autre source ne regarde K_c ; ne jamais l'employer en niveau ; ancrer le niveau annuel sur l'ET implicite des bassins jaugés (pluie CaSR moins débit observé) ; vérifier à Chibougamau (CA-Qfo) et contre EALCO de RNCan ; un second produit en forme (GLEAM4 ou PML-V2.2) comme témoin pour savoir si le choix du produit déplace K_c.
Un défaut du terme centré subsiste : sensible à l'amplitude que K_c multiplie, il pousse K_c vers le bas et a annulé l'ancrage annuel sur P − Q essayé le 9 octobre. Avant tout ancrage de niveau, le rendre insensible à l'amplitude en le normalisant par l'écart-type, comme GRACE en forme.
Préalables : comparaison sans simulation de MOD16, EALCO et GLEAM4 à Chibougamau (CA-Qfo, 2004-2010 ; Wang et al. 2015 y mesurent MOD16 à +24 %) et contre P − Q par territoire, quelques minutes ; ingérer la tour et EALCO ; mesurer la sensibilité SPATIALE de MOD16 (le banc ne perturbe qu'un facteur uniforme, or son contraste spatial est la seule information qui ne soit pas une courbe saisonnière).
Coût : deux jours de préparation avant toute épreuve.

## 6. Ce qui n'est pas lancé, et pourquoi

- Correcteur du débit en aval (GRU, transformeur, noyau) : refusé le 2026-10-22 (R389).
- Partage pluie-neige par réseau : gain au plus 0,6 % sur un seuil à deux variables (Jennings et al. 2025) ; le seuil au bulbe humide dérivé de CanSWE reste.
- Paramètres de la colonne rendus variables dans le temps par un GRU libre : preuve contraire (Acuña Espinoza et al. 2024) ; n'entre que par l'épreuve 3.6 et le test de structure fausse.
- LSTM ou GRU sans physique comme modèle de production : la fuite temporelle des études régionales est documentée ; une mesure locale (GRU pur, 80 % des stations, 2022-2024 aux stations retirées) ne vaut qu'en témoin de l'écart interpolation contre extrapolation, et seulement si un entraînement provincial avec les mêmes stations retirées est fait pour la comparaison juste (plusieurs heures).
- Toute loi sur un champ à sensibilité nulle ou sans lecteur, tant que 0.2 et 0.3 ne l'ont pas rebranchée.

## 7. Calendrier et budget

| Étape | Durée | Machine |
|---|---|---|
| Phase 0 (0.1 à 0.4) | 1 à 2 jours | passes avant de 5 à 10 min, ingestion |
| 5.1 fonte et 5.2 nappe, hors ligne jusqu'à la passe avant | 1 jour | CPU et quelques minutes de carte |
| 5.3 aires saturées et 5.4 porte de gel, hors ligne | 1,5 jour | idem |
| 5.5 et 5.6 routage, après 0.1 et 0.4 | 2 jours | idem |
| Banc de sous-bassin pour les candidates ayant passé 3.1 à 3.7 | 1 à 2 h par variante, 3 graines | carte |
| Territoire entier pour une candidate retenue | demi-journée | carte |
| 5.7 demande évaporative | après ingestion de Chibougamau et EALCO | |

Règle de clôture : chaque candidate reçoit un verdict au registre (établi, réfuté, sans juge) et la recette TOML du territoire est mise à jour le jour même d'un succès au banc, avant de changer de territoire.

## 7 bis. État au 2026-10-23, fin de journée (voir RESULTATS.md)

- 0.1 fait : le mode lagged porte 0,4 jour du retard, le changement seul ne gagne rien à poids gelés ; réentraînement reporté après 5.6.
- 0.2 fait, déporté sur Ubuntu, deux territoires (Montérégie et Outaouais) : quatre directions de paramètres sur quarante-trois, contre dix-neuf pour un témoin de bruit, et vingt-cinq champs qu'aucun des six observables ne touche, les mêmes sur les deux territoires. 0.3 fait : quatorze sorties du champ sans lecteur. 0.4 : l'adresse directe des fichiers de niveaux du CEHQ répond 404, à retrouver ; SWOT non branché.
- 5.1 fonte : épreuve 3.3 passée (gain 26 % sur la loi réajustée, 25 % au territoire retiré) ; la moitié du gain tient dans une forme physique à quatre paramètres ; suite : la déclarer au catalogue de la neige, sonde, passe avant.
- 5.2 nappe : fermée, la puissance à exposant commun (carré de Boussinesq) est la meilleure loi hors échantillon.
- 5.3 aires saturées : épreuve 3.1 passée (défaut de forme net), épreuve 3.4 faite et RÉFUTÉE comme levier (l'exposant à 2, 8 ou 16 ne rend jamais la dépendance observée, et à 2 la perte se dégrade par l'étiage). La décomposition du chemin rapide au pilote (Montérégie, diagnostics journaliers aux stations) montre que la production de surface répond un peu à l'état du sol mais que le DÉBIT ne suit pas, même sans les trois nœuds qui rendent toute leur pluie. Le défaut est entre la production et la rivière, pas dans la loi de partage.
- 5.4 porte de gel : épreuve 3.1 passée, réponse continue et commune ; le modèle sature et se trompe de territoire ; suite : forme physique sans saturation avec le manteau, sonde, passe avant.
- 5.6 routage et réponse rapide : candidate ARRÊTÉE à l'épreuve 3.3 (après-midi). La réponse observée est plus rapide et plus concentrée que la simulée, mais la dispersion entre orages (0,61) domine quatre fois la différence entre les formes communes (0,15), et ni le noyau ajusté sur l'observé (0,591) ni le noyau simulé (0,473) ni un retard d'un jour (0,632) ne transfèrent aux stations retirées. La lecture du matin (part du jour de l'orage 0,008 contre 0,106) est corrigée : elle vient d'une médiane par orage tirée vers le bas par les petits orages à pluie étalée ; le profil moyen donne 0,064 des deux côtés et le modèle est plus LENT que l'observé (retard 3,20 contre 2,95 jours, 27 stations sur 36 dans ce sens). Ce qui reste ouvert est un transfert versant-rivière dont la forme dépend de l'état de l'orage, à écrire comme une fonction de l'intensité de la pluie, de l'humidité antécédente et du stock disponible, puis à éprouver en passe avant à poids gelés sur la Montérégie et l'Outaouais.
- Constat transversal, quatre fois en deux jours : conditionner une loi par les attributs ou les coordonnées améliore les lieux vus et dégrade les lieux retirés. Les lois communes transfèrent ; la spatialisation d'une loi, non.
- Fait de méthode : le poste Ubuntu et le poste Windows ne donnent pas les mêmes nombres pour le même modèle (KGE médian 0,6874 contre 0,7096 sur l'Outaouais, mêmes entrées au bit près) ; toutes les comparaisons de variantes doivent se faire sur une même machine.

## 8. Décisions à prendre par Essi

1. Confirmer l'ordre des candidates, ou en retirer.
2. Autoriser l'ingestion des niveaux _N du CEHQ, de la géométrie SWOT, de la tour de Chibougamau et d'EALCO (préalables 0.4 et 5.7).
3. Trancher l'usage de MOD16 : forme hors neige sur K_c conservée, niveau ancré sur P − Q, second produit en témoin.
4. Décider si le témoin « GRU pur contre meandre, stations retirées » mérite l'entraînement provincial qu'il exige pour être juste.
