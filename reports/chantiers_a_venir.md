# Chantiers à venir de méandre

Ouvert le 2026-09-08. Ce document liste les chantiers identifiés mais non entrepris, avec leur raison d'être, ce qu'ils coûtent et le test qui dira s'ils tiennent. Il ne remplace ni le registre des hypothèses, qui dit ce qui est vrai aujourd'hui, ni le journal des expériences, qui raconte ce qui a été fait. Un chantier n'entre ici qu'avec un critère de réussite mesurable.

## Revue du 2026-09-14, à la lumière des mesures des 12 et 13 septembre

Trois mesures faites sans entraînement ordonnent les chantiers autrement que leur numéro.

La première est l'erreur d'amplitude par événement. Sur 2 876 événements au-dessus du quantile 0,90 des 127 stations de la période d'évaluation, le rapport du pic simulé au pic observé a pour moyenne géométrique 0,59 et pour écart-type logarithmique 0,57 en médiane, 0,38 sur la crue d'avril-mai et 0,61 le reste de l'année ; 84 pour cent des stations dépassent 0,40. Or le banc de perte établit qu'à 40 pour cent d'erreur d'amplitude tout terme ponctuel préfère une série rétrécie, et qu'à 20 pour cent aucun ne la préfère. Le rabotage des pointes est donc la réponse d'un objectif ponctuel à l'incertitude de la précipitation d'entrée et des processus d'été, non un défaut de pondération. Aucun chantier de fonction objectif ne rendra les pointes ; seuls le forçage et la physique estivale le peuvent, et la tête probabiliste porte entretemps la dispersion que la médiane ne peut pas porter.

La deuxième est l'absence de retard. Le décalage optimal entre simulation et observation est nul en médiane et le recalage gagne 0,001 de KGE. Tout chantier fondé sur un retard de la simulation, dont la borne de la constante de temps du routage, perd sa justification première.

La troisième est l'origine des plateaux hivernaux : la fonction objectif les rejette, le champ non entraîné ne les produit pas, le banc de sous-bassin ne les reproduit pas, et l'observation hivernale est reconstruite sous glace sur 49 à 95 pour cent des jours. Le débit d'hiver n'est pas une cible sur laquelle juger le modèle.

**Ordre révisé une seconde fois le 2026-09-14 au soir, après la mesure décisive.** Le rapport du pic simulé au pic observé a été mesuré sur quatre sous-bassins jaugés de Gaspésie avec le champ NON entraîné et le sol imposé de la recette, puis comparé au modèle régional entraîné sur les mêmes stations :

| station | champ non entraîné | modèle régional entraîné |
|---|---|---|
| 011003 | 1,11 | 0,39 |
| 021702 | 1,07 | 0,59 |
| 020404 | 0,79 | 0,45 |
| 022301 | 0,79 | 0,49 |

La physique produit donc des amplitudes proches de l'observé, de 0,79 à 1,11, et l'entraînement lui en retire environ la moitié. Le contrôle qui écarte l'explication par le changement d'échelle est l'épreuve à une époque du Saguenay, à échelle constante, où le rapport tombe de 1,05 à 0,70.

Le rabotage des pointes n'est donc pas un défaut de représentation des processus mais une propriété de l'objectif : le banc de perte établit qu'au-delà de 40 pour cent d'erreur d'amplitude par événement, tout terme ponctuel préfère une série rétrécie, et l'erreur mesurée vaut 0,57. C'est la régression vers la moyenne, la prédiction qui minimise une erreur quadratique sous incertitude étant la moyenne conditionnelle, donc amortie.

Ordre final : la fonction objectif (chantier 2) passe premier, avec la contrainte que la solution doit comparer des distributions et non des points ; la neige mesurée au sol comme seule cible hivernale honnête (chantier 5) ; l'erreur d'amplitude du forçage (chantier 6), qui reste utile pour abaisser l'incertitude sous le seuil de 0,40 ; les projections climatiques (chantier 1). Les chantiers 3 et 7 se ferment, le 4 reste opportuniste.

Deux réserves. L'erreur d'amplitude de 0,57 est mesurée sur des modèles déjà entraînés, donc déjà rabotés ; le rabotage gonfle le biais et pourrait gonfler un peu la dispersion. Et le champ non entraîné n'est pas parfait : 0,79 sur deux stations des quatre.

## 1. Forçage climatique MRCC6 et projections futures

**Pourquoi.** Le modèle sert aujourd'hui à reconstituer le passé et à naturaliser des débits observés. Le troisième volet de la stratégie de publication, les scénarios, demande de le faire tourner sous un climat qui n'a pas eu lieu. Le Modèle régional canadien du climat de sixième génération, développé au centre ESCER de l'UQAM avec Ouranos et fondé sur le coeur atmosphérique GEM, est la source régionale naturelle pour le Québec.

**Ce qu'il faut comprendre avant de commencer.** Un modèle climatique ne reproduit pas la météorologie d'une journée donnée, il reproduit la statistique d'un climat. Il ne peut donc servir ni à l'entraînement, ni à un calcul d'efficacité contre des débits observés. L'usage correct est en deux temps : entraîner sur une réanalyse, ici les données CaSR corrigées, puis simuler avec les sorties climatiques. Confondre les deux produirait des scores dénués de sens.

**Ce que le chantier demande.** Obtenir les six variables consommées par la colonne verticale au pas journalier sur le domaine québécois, soit la précipitation, les températures minimale et maximale, le rayonnement net, le vent et l'humidité. Les interpoler aux tronçons, ce que la chaîne de préparation du forçage sait déjà faire. Construire une correction de biais contre CaSR sur une période commune : c'est l'étape déterminante, puisque le plafond de performance mesuré sur ce modèle est fixé par le forçage et non par la structure. Vérifier enfin que le régime simulé sur la période historique du modèle climatique, après correction, reproduit les signatures hydrologiques observées.

**Critère de réussite.** Les signatures hydrologiques simulées sous le climat historique du modèle régional, après correction de biais, doivent être compatibles avec celles obtenues sous la réanalyse. Sans cela, aucune projection n'est défendable.

**Ce que la différentiabilité apporte ici.** La dérivée du débit par rapport à chaque paramètre et à chaque variable de forçage est calculable exactement. Une projection peut donc être accompagnée de l'attribution de son changement aux différentes causes, ce qu'un modèle calé de façon classique ne permet pas.

## 2. Fonction objectif non paramétrique

**Pourquoi.** Le diagnostic de septembre 2026 a établi que la composante de variabilité de l'efficacité de Kling-Gupta, seul terme de la perte qui pénalise un hydrogramme figé, est satisfaisable en abaissant la moyenne autant qu'en augmentant la variabilité, puisqu'elle compare des coefficients de variation. Wagener et ses collaborateurs, dans Hydrology and Earth System Sciences en 2026, calent quarante-sept structures sur dix bassins avec huit fonctions objectif et concluent que le choix de la fonction objectif pèse souvent plus lourd que le choix de la structure, et que la version non paramétrique de l'efficacité de Kling-Gupta donne la plus faible erreur globale sur les signatures sensibles. Cette version remplace précisément le rapport des variances par un terme fondé sur la courbe des débits classés, et la corrélation de Pearson par celle de Spearman.

**Ce que le chantier demande.** Implémenter cette version comme terme différentiable, l'apparier à un terme de basses eaux selon la recommandation des auteurs, et juger le résultat sur des signatures hydrologiques plutôt que sur une efficacité globale.

**Un banc de perte remplace désormais l'entraînement pour la plus grande part de ce chantier (2026-09-13).** Régler une fonction objectif est une question sur la fonction objectif, pas sur le modèle : on prend un hydrogramme observé, on le déforme comme le modèle se trompe, et on demande à chaque terme s'il préfère l'original. `banc_perte.py` le fait sur les 32 stations complètes des neuf régions en une minute, `banc_perte_dosage.py` en déduit le poids qui renverse une préférence. L'entraînement ne reste nécessaire que pour la dernière question, celle de savoir si l'optimiseur atteint la solution que la perte préfère.

Ce que ces deux bancs ont établi, en partant d'une simulation en retard d'une journée, ce qui est la situation ordinaire dès qu'une averse est mal localisée :

| déformation | effet sur la perte de la recette | stations où la perte la préfère |
|---|---|---|
| lissage sur 7 jours | -8,0 % | 64 % |
| lissage sur 15 jours | +47,6 % | 10 % |
| rabotage des pics de 30 % | +35,3 % | 23 % |
| hiver figé, volume conservé | +28,3 % | 1 % |

Trois conséquences pour la conception de la perte.

Les basses eaux d'hiver n'ont besoin d'aucun terme supplémentaire : la perte rejette déjà leur nivellement, et c'est l'écart quadratique sur le logarithme qui le rejette, à 127 pour cent. Les plateaux hivernaux observés dans les sorties ne sont donc pas récompensés par l'objectif ; ils sont un échec d'optimisation, non un défaut de définition.

Le terme de pics ne peut pas empêcher le rabotage des pics, à aucun poids. Sur au moins une station sur dix, raboter les pointes améliore ce terme, parce que la simulation est décalée et que raboter réduit l'écart aux dates où les pointes ne coïncident pas. Le terme écrit pour protéger les crues les dessert donc dès qu'il y a du retard, ce qui est une conception à revoir et non un poids à régler.

Un entraînement de vingt époques sur un sous-bassin jaugé va dans le même sens sans permettre d'attribuer : la composition complète de la recette porte le rapport des pointes annuelles à 0,58 contre 0,77 sans ses deux termes supplémentaires, et double la platitude estivale, mais elle ajoute le terme de pics et l'écart quadratique logarithmique ensemble. Le bras qui aurait séparé les deux a été arrêté. L'attribution au terme de pics repose donc sur le banc de perte seul.

Les résultats ci-dessus supposent une simulation en retard d'une journée. Cette prémisse a été vérifiée le 2026-09-14 et ne tient pas : sur les 129 stations de la période d'évaluation, le décalage optimal est nul en médiane et le recalage gagne 0,001 de KGE. Sous cette prémisse, le terme sur les variations d'un jour à l'autre à 0,78 renversait la préférence pour le lissage ; sous la prémisse correcte, il ne tient plus.

La prémisse correcte est une erreur d'amplitude par événement, la simulation étant bien datée mais se trompant sur la taille de chaque averse. Le banc de perte, sur une série observée multipliée par un bruit lognormal corrélé sur cinq jours puis rétrécie de 25 pour cent vers sa moyenne, donne :

| erreur d'amplitude par événement | perte totale : préfère la série rétrécie à | écart quadratique | terme de pics | variations d'un jour | KGE | écart quadratique log |
|---|---|---|---|---|---|---|
| 20 % | 0 % des stations | 4 % | 8 % | 27 % | 0 % | 0 % |
| 40 % | 49 % des stations | 70 % | 70 % | 66 % | 34 % | 29 % |

Sous une erreur d'amplitude de 40 pour cent, tout terme ponctuel préfère la série rétrécie sur deux stations sur trois. C'est la régression vers la moyenne : sous incertitude sur la taille des événements, un objectif ponctuel préfère une amplitude réduite. Seuls les termes qui comparent des variabilités ou des distributions résistent, l'efficacité de Kling-Gupta par son rapport des variabilités et l'écart quadratique logarithmique, et ils ne résistent qu'en partie. Le rabotage des pics n'est donc pas un défaut de pondération : c'est la réponse correcte d'un objectif ponctuel à l'incertitude de la précipitation d'entrée. La mesure qui décide est l'erreur d'amplitude réelle par événement des modèles : autour de 20 pour cent, un terme de distribution à fort poids se justifie ; autour de 40, le levier est le forçage.

**Ce chantier est passé premier le 2026-09-14 au soir**, la mesure des pointes à zéro époque ayant montré que le rabotage vient de l'objectif et non de la physique. Deux points étroits. Le terme de pics préfère la série rétrécie sur 70 pour cent des stations sous une erreur d'amplitude de 40 pour cent : il dessert ce qu'il doit protéger et doit être retiré ou refondu sur une grandeur de distribution, un rapport de quantiles hauts par exemple. Et un terme de distribution à fort poids, l'efficacité de Kling-Gupta à 12,9 ou un terme de courbe des débits classés, ne se justifie que comme atténuation partielle, à juger d'abord sur le banc de perte sous l'erreur d'amplitude mesurée de 0,57, jamais par entraînement en premier.

**Critère de réussite.** Sur le banc de perte, sous une erreur d'amplitude par événement de 0,57, la perte doit préférer la série à la bonne variance à la série rétrécie sur au moins 80 pour cent des stations ; puis, à l'entraînement, le rapport des pointes annuelles doit rester dans l'intervalle de 0,8 à 1,2 à efficacité médiane égale.

**Mesuré le 2026-09-12, sans entraînement : la prémisse ne tient pas pour ce défaut.** La version non paramétrique a été calculée sur les 135 stations de la période d'évaluation, à partir des débits déjà simulés, puis sur des séries fabriquées en lissant une observation réelle par moyenne mobile. Elle est plus tolérante au lissage que la version usuelle, et non moins. Sur un lissage à 90 jours, qui porte la part de jours plats de 11,8 à 53,4 pour cent, l'efficacité usuelle tombe de 1,000 à 0,383 tandis que la version non paramétrique tient encore 0,682 ; à 30 jours, 0,708 contre 0,887. Sur les stations réelles, la corrélation de rang entre la part de jours plats simulée et l'efficacité reste positive dans les deux cas, 0,19 pour la version usuelle et 0,11 pour la non paramétrique : ni l'une ni l'autre ne pénalise un hydrogramme figé. La cause est le terme de courbe des débits classés, qui est une distance bornée entre deux distributions et punit donc moins qu'un rapport de coefficients de variation, lequel s'effondre quand la variance disparaît.

Cela ne contredit pas Wagener et ses collaborateurs, dont le résultat porte sur l'erreur globale sur les signatures et non sur les hydrogrammes figés. Cela retire en revanche à ce chantier sa justification première. S'il est repris, ce doit être pour les signatures, avec un terme de forme distinct chargé des plateaux.

## 3. Plateaux d'hiver

**Pourquoi.** Huit régions sur quatorze échouent au verdict de forme, et leurs suites plates les plus longues sont hivernales, de cinquante-neuf à cent dix-neuf jours. La part imputable au modèle et celle imputable aux observations n'ont pas été séparées : le Centre d'expertise hydrique reconstruit la majorité des débits de janvier et février sous couvert de glace, et l'observé est lui-même plus plat en hiver.

**Ce que le chantier demande.** Reprendre le verdict de forme en distinguant les jours réellement mesurés des jours reconstruits, à partir des remarques accompagnant les débits publiés, puis rejuger la platitude hivernale sur les seuls jours mesurés.

**Critère de réussite.** Savoir si la règle actuelle est trop sévère en hiver, et si oui la corriger, ou constater que le modèle fige réellement son hiver et ouvrir alors un diagnostic de physique.

**Mesuré les 2026-09-12 et 2026-09-13 : deux défauts distincts, et le principal vient de l'entraînement.**

Le premier concerne la variation d'un jour à l'autre. Recalculée par saison sur les seules paires de jours calendaires consécutifs dont les deux observations sont mesurées, la part de jours plats simulée dépasse l'observée d'un facteur médian de 1,10 de décembre à mars, avec une seule région sur huit au-dessus de deux, contre 2,63 de juin à septembre, avec six régions sur neuf au-dessus de deux. Sur ce critère, la règle actuelle est trop sévère en hiver : elle compare une simulation disponible tous les jours à une observation dont l'hiver est largement reconstruit sous couvert de glace, donc lissée par construction.

Le second concerne les longues suites à débit figé, que la part de jours plats ne peut pas voir puisqu'elle est une moyenne sur des paires de jours. Ces suites sont hivernales et massivement plus nombreuses que dans les observations : sur les neuf régions de la période d'évaluation, 395 suites simulées d'au moins vingt jours contre 29 observées, dont 278 commencent en décembre, janvier ou février. Elles ne sont pas des débits constants mais des récessions perdant 17 pour cent en trois mois sans jamais varier de un pour cent par jour. Pendant ces plages, l'observation varie de 1,76 pour cent par jour contre 0,27 pour la simulation et double au moins une fois dans 58 pour cent des cas.

Leur origine a été isolée sur un sous-bassin jaugé du Saguenay, à graine fixée et sous le forçage de la recette. Le champ non entraîné rend une plus longue suite de 8 jours et 8,1 pour cent de jours plats en hiver ; le champ entraîné trente époques sur la région rend 38 jours et 22,1 pour cent, pour un gain de KGE de un millième. Les plateaux ne sont donc pas produits par la physique, ils sont produits par l'optimisation, ce qui rejoint le constat qu'aucun terme de la fonction de perte ne pénalise un hydrogramme figé.

**Fermé le 2026-09-14.** L'épreuve appariée d'un terme de forme a été faite : après vingt époques sur ce sous-bassin, la plus longue suite plate vaut 8 jours sans le terme et 5 avec, pour 0,022 de KGE et 7 points d'évapotranspiration en moins, et le témoin ne figeait pas. Le banc de perte a ensuite montré que la fonction objectif rejette déjà le figeage hivernal à volume conservé, préféré à une station sur cent. Les plateaux de 38 jours du point de reprise régional ne se reproduisent ni avec quatre termes ni avec six ; leur étude demande une région et ses vingt-deux stations. Comme l'observation hivernale est reconstruite sous glace, le débit d'hiver n'est pas une cible : la neige mesurée au sol l'est, et c'est le chantier 5.

**Pistes de physique écartées, chacune sur une mesure appariée.** La modulation saisonnière du facteur de fonte et l'aquifère ne déplacent ni la plus longue suite ni la platitude hivernale au-delà de la dispersion due au tirage du champ initial. Le degré-jour intégrant le cycle diurne, écrit et testé à cette occasion, ne les déplace pas non plus et coûte 0,014 de KGE de façon reproductible sur trois tirages ; il reste désactivé.

## 4. Cause du gradient non fini

**Pourquoi.** Des blocs sont écartés à chaque entraînement parce que leur gradient devient non fini. Le filet de sécurité les jette sans perdre l'époque, mais la cause reste inconnue, et les gels de validation observés dans le bras à un pas par bloc suivent chaque fois un épisode de blocs jetés.

**Ce que le chantier demande.** L'instrumentation existe et imprime, avant restauration, les paramètres touchés et la norme du gradient de chaque terme de la perte. Elle n'a pas encore attrapé de bloc fautif sur le sous-bassin d'essai. Il faut la faire tourner sur une région entière, où le phénomène se produit.

**Critère de réussite.** Nommer l'opération et le terme responsables, et proposer un correctif qui supprime le rejet plutôt que de le compenser.

## 5. Activer la contrainte de neige mesurée au sol

**Pourquoi.** La cible CanSWE est construite, rattachée au réseau et vérifiée depuis août 2026, mais son poids dans la fonction de perte vaut zéro : aucun modèle n'a jamais été contraint par la neige mesurée au sol. Or le manteau nival gouverne la crue printanière, qui est l'événement dominant de l'année hydrologique québécoise, et deux défauts persistants du modèle sont hivernaux, les longues suites plates de décembre à mars et le calendrier de la fonte.

**Ce que le chantier demande.** Le rattachement est déjà fait : chaque station est liée au tronçon le plus proche, plafond de vingt-cinq kilomètres, écart d'altitude conservé comme diagnostic. Ce qui manque est le réglage du poids, et il ne peut pas être uniforme : la densité du réseau varie d'un facteur cent entre les régions, de cent dix sites en Outaouais à un seul en Montérégie, et les distances médianes de rattachement vont de 2,1 à 24,9 kilomètres. Un poids qui a du sens en Outaouais n'en a aucun en Montérégie.

**Ce qu'il faut mesurer avant de l'activer.** Une paire appariée sur une région bien couverte, avec et sans la contrainte, jugée sur la perte et ses composantes, sur le calendrier de la crue printanière et sur les signatures hivernales, jamais sur le KGE seul. Une contrainte auxiliaire qui améliore le débit en dégradant les autres cibles ne contraint rien, elle déplace la compensation.

**Critère de réussite.** La masse simulée du couvert et sa date de disparition se rapprochent des relevés, sans dégradation de l'évapotranspiration satellitaire ni du stockage gravimétrique, et la platitude hivernale diminue.

## 5 bis. Neige mesurée au sol : ce que les mesures du 2026-09-14 fixent

**La cible est la MASSE, pas le calendrier.** Comparée aux relevés de quatre sites du Saguenay sur six hivers, aux dates de relevé et sans ajustement de courbe, la neige simulée par le champ non entraîné vaut 0,75 de l'observée en janvier, 0,71 en février, 0,66 en mars et 1,00 en avril. Le modèle n'a jamais de neige quand l'observation n'en a plus, et l'inverse ne se produit qu'une fois sur dix-huit en février. Le calendrier de la fonte est donc correct et n'a pas besoin d'être contraint ; c'est l'accumulation qui manque, d'un tiers au maximum hivernal. Le terme de masse existe déjà dans l'entraîneur, avec son échelle de cent millimètres et son filtre de représentativité à quinze kilomètres et cent cinquante mètres.

**Un indicateur territorial étend la cible aux régions sans station.** Un modèle à gradient boosté prédisant la masse au maximum à partir du forçage CaSR du nœud et des attributs de PHYSITEL, validé en retirant des STATIONS entières et non des hivers, rend 0,77 de variance expliquée sur 1 488 couples station-hiver de 173 stations, pour une erreur absolue de 31 millimètres sur un manteau médian de 158. La durée de la fonte, elle, ne se prédit pas : 0,09 de variance, et une erreur de 11,6 jours contre 13,5 pour la simple moyenne. Les deux résultats convergent avec le précédent : la grandeur défaillante est aussi la grandeur régionalisable.

**Ce que la représentativité impose.** Entre stations voisines à moins de 50 km, la masse au maximum diffère de 21 pour cent en médiane et de plus de 30 pour cent pour 37 pour cent des couples. Le taux de fonte normalisé diffère de 36 pour cent à deux relevés par station, 25 à trois et 15 à cinq : l'essentiel est du bruit d'estimation, non de la variabilité spatiale. La fonte dure 36 jours en médiane et quatre relevés y tombent malgré un calendrier bimensuel.

**Couverture.** Sites avec au moins cinq relevés dans la fenêtre de fonte : Outaouais 76, Saint-Laurent nord-ouest 30, Saguenay 22, Gaspésie 22, Centre-du-Québec B 21, Abitibi 7, Centre-du-Québec C 5, Montérégie 1, Saint-Laurent sud-ouest 0. Les deux régions les plus faibles, Montérégie à 0,611 et Saint-Laurent sud-ouest à 0,652, n'ont rien : seul l'indicateur territorial peut les atteindre.

**Réserve.** La comparaison aux relevés porte sur quatre sites d'un seul sous-bassin et six hivers, avec un champ non entraîné. La refaire sur l'Outaouais et ses 110 sites ne coûte qu'une simulation.

## 8. Voie positionnelle du champ spatial, à reprendre plus tard

**Pourquoi.** Au départ à chaud, la voie des attributs interpole mais la voie de la position extrapole, et l'encodage de Fourier est périodique : les six bandes ont des périodes de 2 400, 1 200, 600, 300, 150 et 75 km sur un rayon de référence de 1 200 km, donc les deux dernières sont plus courtes qu'une région. Rien à l'entraînement ne force le réseau à préférer les attributs.

**Mesuré le 2026-09-14, transfert Saguenay vers Gaspésie, sans entraînement.** En remplaçant les positions d'accueil par celles des tronçons du donneur les plus proches en attributs, ce qui impose un déplacement de 277 km en médiane, l'écart relatif des 42 paramètres vaut 0,5 pour cent en médiane et 1,2 au neuvième décile. Annuler les bandes de 150 et 75 km le ramène à 0,3 pour cent. Les trois conductivités à saturation font exception, de 3,7 à 6,6 pour cent en médiane et jusqu'à 25 au neuvième décile ; ce sont les seuls paramètres de sol laissés au champ par la loi des ancrages, donc les seuls réellement appris.

Le gradient raconte autre chose : la position porte 45,7 pour cent de la sensibilité en médiane et plus de la moitié pour dix-huit paramètres sur quarante-deux, en tête la deuxième fraction racinaire à 73 pour cent. Le réseau s'appuie donc largement sur la position, mais la surface apprise est plate à grande échelle, si bien que l'effet au transfert reste petit.

**Priorité basse**, décidée par Essi le 2026-09-14. À reprendre si un transfert lointain échoue sans explication. Le script est `audit_position.py`.

**Au passage.** Le point de reprise ne porte que 37 sorties de champ sur 42 : krec, la diffusivité de gel, la fraction de neige et les deux décalages de canopée sont figés au milieu de leurs bornes et identiques sur tous les tronçons. Ils ne sont pas appris.

## 9. Le modèle est trop lissé, et la cause n'est pas identifiée

**Mesuré le 2026-09-14 sur 129 stations.** La variabilité des variations de débit d'un jour à l'autre vaut 0,54 de l'observée en médiane, et 79 pour cent des stations sont sous 0,8. Lisser l'observation améliore la corrélation sur 95 pour cent des stations, de 0,791 à 0,831, le lissage optimal médian étant de cinq jours : le modèle ressemble davantage à une moyenne mobile de l'observation qu'à l'observation.

**Aucun facteur structurel ne l'explique.** Corrélations de rang de l'indice de sur-lissage avec le nombre de tronçons en amont -0,274, l'aire drainée -0,215, l'accord entre la pluie d'entrée et la montée observée -0,121, la variabilité de la pluie d'entrée +0,101, la part de tronçons-lacs +0,025. Aucun ne dépasse huit pour cent de variance expliquée, et la part de lacs ne donne pas de tendance monotone. La borne inférieure de la constante de temps du Muskingum a déjà été abaissée sous quatre heures sans succès en août 2026.

**Ce qui compte.** La corrélation porte l'essentiel du déficit de score : elle vaut 0,791 en médiane et la rendre parfaite vaudrait 0,110 de KGE, contre 0,051 pour le rapport des moyennes et 0,043 pour le rapport des variabilités. La crue d'avril-mai porte 59 pour cent de la variance annuelle du débit observé pour une corrélation de 0,817, l'été 22 pour cent à 0,730, l'hiver 13 pour cent à 0,738, l'automne 4 pour cent à 0,674.

**Cause trouvée le 2026-09-14 au soir : le temps de transfert du canal n'a jamais été appris.** Quatre mesures enchaînées, toutes sans entraînement.

La colonne n'est pas en cause. Sur un sous-bassin du Saguenay choisi parce qu'il porte le défaut, la variabilité des variations d'un jour vaut 1,350 pour la production de la colonne avant routage, 0,174 pour le débit simulé après routage et 0,437 pour l'observé. La colonne produit donc une eau trois fois plus nerveuse que la rivière, et le réseau en retire 87 pour cent. L'atténuation du réseau vaut 0,17 à 0,19 en crue et en été mais 0,55 en hiver, donc elle dépend du régime.

Le canal porte tout. En passe avant seule, quatre configurations sur le même sous-bassin : la recette actuelle donne une nervosité de 0,40 et des pointes à 0,60 ; le canal purement advectif donne 2,12 et 1,34 ; l'hydrogramme de versant, qui est désactivé et absent de la recette, ne déplace rien, 0,38 et 0,61. Le KGE ne bouge que de 0,006 entre la recette et le canal advectif, parce que le gain sur la variabilité et les pointes compense exactement une perte de 0,139 sur la corrélation. La bonne réponse est entre les deux.

Les paramètres du routage sont constants. Sur 2 212 tronçons du Saguenay, le temps de transfert vaut 23,96 h avec un écart-type de 0,3 h et le coefficient de pondération 0,202 à 0,006 près ; sur 3 917 tronçons de Gaspésie, 19,4 h et 0,278. Aucun tronçon n'est à une borne. Un ruisseau de trois kilomètres et un fleuve reçoivent le même temps de parcours.

La cause est dans les poids. La ligne de `fc_out` qui produit le temps de transfert a une norme de 0,058 et son entrée brute un écart-type de 0,021, contre 0,878 et 1,68 pour la conductivité à saturation, alors que le tronc du réseau différencie bien les tronçons, ses unités cachées ayant un écart-type de 0,34. Le routage est resté à son initialisation. La fraction verticale montre le même profil, norme de 0,055 et brut de 0,020 : ce n'est pas propre au routage mais commun à tout ce que la loi des ancrages n'a pas laissé au champ.

Le tronçon de rivière médian mesure 5,7 km au Saguenay et 3,6 km en Gaspésie, une fois écartés les nœuds de lac dont la colonne de longueur porte un périmètre de rive, 461 km en médiane. Son temps de parcours physique vaut environ une heure à un mètre par seconde. L'initialisation constante à 24 h est donc vingt-cinq fois trop longue, et la borne basse de 4 h reste quatre fois trop longue pour le tronçon médian. Le commentaire du code le mesurait déjà le 2026-08-09, bornes rendues configurables à cette date, et la recette est restée à 4, 48, 24.

**Correctif préparé.** `set_routing_anchor` pose un temps de transfert par tronçon égal à la longueur divisée par la célérité cinématique, la vitesse suivant la racine de la pente selon Manning, et le réseau module autour, sur le patron déjà employé pour les lacs. Sept vérifications le couvrent, dont sa neutralité exacte quand il est absent et le passage du gradient. L'épreuve appariée est `routage_ancre.sbatch`, six tâches, deux régions et trois bras : témoin, bornes physiques seules, ancrage. Le bras des bornes seules est indispensable pour distinguer l'effet de l'ancre de celui de la simple permission de descendre sous quatre heures.

## 6. Erreur d'amplitude du forçage

**Pourquoi.** Le pic simulé vaut en moyenne géométrique 0,59 du pic observé, et l'erreur d'amplitude par événement, écart-type du logarithme de leur rapport, vaut 0,57 en médiane sur la période d'évaluation. Le banc de perte montre qu'à ce niveau d'incertitude tout objectif ponctuel rétrécit les pointes. Le plafond de 0,76 de KGE attribué au forçage en juillet 2026 et ce rabotage sont la même limite vue de deux côtés. Le passage de CaSR à sa version corrigée a relevé l'efficacité mais n'a pas été jugé sur cette grandeur.

**Mesuré le 2026-09-14 : l'erreur ne vient pas de la lame d'eau reçue.** La décomposition a été faite par `decompose_amplitude.py`, qui apparie chaque pointe observée au cumul de précipitation des stations d'Environnement Canada situées dans le bassin amont, sur les quatre jours qui la précèdent.

| région | jauges par bassin | biais de la pluie d'entrée | biais du pic simulé | part de l'erreur de débit expliquée |
|---|---|---|---|---|
| Montérégie, toutes saisons | 5 | -0,07 | -0,60 | 3 % |
| Montérégie, hors crue de printemps | 5 | -0,12 | -0,73 | 1 % |
| Gaspésie, toutes saisons | 1 | +0,01 | -0,56 | 0 % |

La précipitation cumulée avant l'événement n'a pas de biais mesurable alors que le pic simulé vaut la moitié de l'observé, et l'erreur de l'un ne suit pas celle de l'autre. Le déficit est donc dans la transformation pluie-débit. Un krigeage sur stations ne le corrigerait pas, et ce chantier passe derrière le chantier 7.

Deux réserves. La part expliquée est une borne inférieure : une jauge ponctuelle porte sa propre erreur de représentativité pour une pluie de surface, et cette erreur dilue la corrélation vers zéro. Et la mesure ne dit rien de la répartition spatiale de la pluie à l'intérieur du bassin, que le cumul moyen efface.

**Ce que le chantier demande désormais.** Reprendre la décomposition avec une pluie de surface krigée plutôt qu'une moyenne de jauges, ce qui lève la dilution, et l'étendre aux régions à réseau dense. Le krigeage reproductible construit en juillet 2026 est l'outil ; la fusion point-surface de juin l'avait aggravée et ne doit pas être reprise telle quelle.

**Critère de réussite.** Mesuré par `erreur_amplitude.py` sur une région, l'écart-type du log-rapport des pics passe sous 0,40 hors crue printanière, seuil en dessous duquel le banc de perte ne récompense plus le rétrécissement.

## 7. Écoulement rapide d'été

**Pourquoi.** Sur les seules paires de jours mesurés, la part de jours plats simulée dépasse l'observée d'un facteur 2,6 de juin à septembre, six régions sur neuf au-dessus de deux, contre 1,1 en hiver. L'erreur d'amplitude par événement vaut 0,61 hors crue printanière contre 0,38 pendant celle-ci, et la simulation est en avance d'un jour en été. Le déficit d'écoulement rapide sous les averses estivales, établi en juin 2026 avec un pas de temps journalier trop long pour le ruissellement hortonien, est la cause candidate ; elle n'a jamais été traitée.

**Fermé le 2026-09-14 au soir : la prémisse ne tient pas pour le modèle déployé.** Le banc ci-dessous a été conduit avec la conductivité de littérature du limon, 0,108 mètre par jour, alors que le champ apprend 0,021 sur le bassin d'essai, cinq fois moins. Le modèle réel produit déjà 66 pour cent de sa lame en ruissellement de surface sur l'année et 53 pour cent en été, par excès de saturation, et l'activation de l'excès d'infiltration sous-journalier n'ajoute que 0,6 point en été sans déplacer ni les pointes ni l'erreur d'amplitude. Surtout, le champ non entraîné rend des pointes de 0,79 à 1,11 : il n'a pas de déficit de production à corriger. Le canal de durée d'averse et le mécanisme restent disponibles et testés, désactivés par défaut.

**Ce que le banc avait mesuré, sur une configuration non représentative.** Le banc `banc_ruissellement.py` fait recevoir à une colonne BV3C2 isolée une averse d'un jour sur un sol à humidité donnée, sans réseau ni routage ni entraînement, et lit la part qui sort en écoulement rapide le jour même.

| averse, sol à 70 % d'humidité relative | clone au pas journalier | durée d'orage de 6 h | durée d'orage de 2 h |
|---|---|---|---|
| 20 mm | 0,5 % | 0,5 % | 32 % |
| 40 mm | 0,4 % | 0,4 % | 66 % |
| 80 mm | 0,4 % | 49 % | 83 % |

La référence en forêt boréale est de 5 à 30 pour cent selon l'humidité antérieure. La colonne est un à deux ordres de grandeur en dessous, ce qui explique que le pic simulé vaille 0,48 à 0,55 de l'observé hors crue printanière. Le résultat est stable de 16 à 1 024 sous-pas de Courant : ce n'est pas un artefact numérique. La conductivité de la première couche ne suffit pas : la diviser par quatre ne change rien en deçà de 80 mm.

Le mécanisme manquant est l'excès d'infiltration sous-journalier, déjà porté par le clone sous le nom `storm_hours` et activé par `use_hortonian`. Il était inerte sur la ligne québécoise faute d'entrée : le forçage n'a que six canaux, alors que le banc SLSO en a sept dont la durée effective d'orage. Ce canal a été ajouté au constructeur québécois le 2026-09-14, en option, sous la même définition que sur SLSO, la pluie journalière divisée par le maximum horaire bornée entre 1 et 24 heures. Sur la Gaspésie il donne une durée médiane de 3,2 heures les jours de pluie et 47 pour cent des jours sous trois heures.

**Ce que la comparaison a donné.** Six canaux contre sept, en passe avant seule sur un sous-bassin jaugé de Gaspésie : pointes annuelles 1,11 puis 1,12, rapport d'amplitude 0,69 puis 0,70, platitude d'été 6,3 puis 6,6 pour cent. L'effet est nul, et le sous-bassin ne portait de toute façon pas le défaut, son rapport de pointes valant 1,11 avant comme après.

**Leçon de méthode.** Deux fois dans la même journée, un remède a été éprouvé sur un témoin sain : le terme de forme contre des plateaux qu'un témoin sans plateaux ne pouvait pas révéler, puis l'excès d'infiltration contre un déficit de pointes qu'un sous-bassin à 1,11 ne présentait pas. Vérifier que le témoin porte le défaut fait désormais partie des conditions préalables énoncées dans CLAUDE.md.

**Manque connu, écarté le 2026-09-15 : l'enveloppe probabiliste n'existe que sur le scénario géré.** La passe naturalisée du pilote, `ETL_DUMP_NATUREL`, rejoue la simulation avec les prélèvements à zéro et les mêmes poids, mais elle n'écrit que les grandeurs dérivées du débit, sans les sept quantiles ; et le second étage, celui de la tête de quantiles, ne la déclenche pas du tout. Une naturalisation probabiliste, qui dirait que le débit naturalisé vaut tant plus ou moins tant, demanderait d'ajouter les quantiles à la passe naturalisée puis de reprendre le second étage. Environ une heure de travail et une de calcul. Essi l'a jugée non nécessaire pour l'instant.
