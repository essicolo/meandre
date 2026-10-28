# Annexe 2. Où la littérature a placé de l'apprentissage automatique à l'intérieur d'un modèle physique, et ce qui a généralisé (agent, 2026-10-22)

Méthode : chaque référence cherchée sur le web (pages HESS, Copernicus, arXiv, PMC, résumés Wiley via Semantic Scholar, USGS). Les pages Wiley (WRR, GRL) sont bloquées à la lecture directe ; pour celles-là seuls les résumés ont été lus. Vocabulaire : retrait spatial = bassins jamais vus (PUB k-fold, ou régions entières PUR) ; retrait temporel = mêmes bassins, années jamais vues ; fuite temporelle = évaluation spatiale sur les mêmes années que l'entraînement des autres bassins.

## 1. Paramètres spatialisés par réseau à l'intérieur d'un modèle conceptuel (famille dPL / δHBV)

Tsai et al. 2021, Nature Communications 12 (vérifié, résumé). Un réseau lit attributs et forçages et produit les paramètres d'un modèle différentiable (VIC pour SMAP, HBV pour le débit). Retrait spatial et temporel sur CONUS ; surpasse les méthodes évolutionnaires et de régionalisation, ou n'a besoin que d'environ 12,5 % des données pour la même qualité. Limite : le gain croît avec le nombre de sites.

Feng et al. 2022, WRR 58 (vérifié, résumé). Un LSTM produit, pour 16 HBV parallèles par bassin, 11 paramètres statiques et 2 dynamiques (βt, γt) plus 2 paramètres d'hydrogramme unitaire gamma. 671 bassins CAMELS, retrait temporel. NSE médian 0,732 (δ) contre 0,748 (LSTM) avec Daymet. Le CFMAX de fonte est régionalisé par le réseau : seul « apprentissage de la fonte » publié dans cette lignée.

Feng, Beck, Lawson, Shen 2023, HESS 27, 2357-2373 (vérifié, page lue). 531 bassins ; PUB k-fold et PUR (7 régions retirées), même période 1989-1999 (fuite temporelle assumée) ; test temporel séparé. NSE / KGE médians : PUB Daymet δ 0,69 / 0,67, LSTM 0,66 / 0,66 ; PUR Daymet δ 0,62 / 0,63, LSTM 0,55 / 0,51 ; PUR Maurer δ 0,59 / 0,61, LSTM 0,55 / 0,52. Tendances décennales en PUR : R² du débit moyen LSTM 0,64 contre δ 0,88 ; Q98 0,27 contre 0,77. ET non entraînée contre MOD16A2 : corrélation médiane 0,84. Limites : conservation de la masse rend δ sensible au biais de précipitation ; réservoir linéaire blâmé pour l'étiage. C'est la seule configuration de la revue qui gagne nettement en extrapolation régionale : +0,07 NSE, +0,12 KGE.

Song, Bindas, Shen et al. 2025, WRR 61 (δHBV2.0, résumé). Paramétrisation multi-échelle d'un HBV sur unités MERIT de 37 km², routage Muskingum-Cunge explicite ; 2 807 bassins d'entraînement, 4 997 d'évaluation ; NSE médian 0,68 contre 0,48 pour NWM3.0 ; protocole de validation non lisible dans le résumé.

Song, Shen et al. 2024, HESS (adjoint, schéma implicite ; résumé). Rendre différentiable un schéma implicite par méthode adjointe ; gain modéré en KGE, pics et étiages ; corrige des distorsions de paramètres.

## 2. Remplacement de processus internes par un réseau

Jiang, Zheng, Solomatine 2020, GRL 47 (résumé seulement). Modèle conceptuel inséré comme couche récurrente d'un réseau profond ; chiffres non vérifiés.

Höge et al. 2022, HESS 26, 5085-5102 (page lue). Neural ODE : M50, deux petits réseaux remplacent l'ET et la fonction de débit d'EXP-HYDRO ; M100, un réseau à cinq sorties remplace les cinq processus. 569 bassins, calage bassin par bassin, retrait temporel seulement. NSE médian de M50 supérieur d'environ 0,23 à EXP-HYDRO. Limites avouées : relations apprises parfois contre-intuitives (fonte qui décroît quand le stock croît) ; pas de test sur bassin non vu.

Acuña Espinoza et al. 2024, HESS 28, 2705-2719 (page lue). LSTM produisant les paramètres dynamiques d'un modèle à réservoirs (SHM), avec un modèle volontairement faux (NonSense). CAMELS-GB, 669 bassins, retrait temporel seulement. NSE médian : LSTM 0,87, LSTM+SHM 0,84, SHM calé 0,76, LSTM+NonSense 0,80 contre 0,51 pour NonSense seul. Verdict : la structure conceptuelle n'est pas un régularisateur efficace quand ses paramètres varient dans le temps ; l'avantage restant est l'accès aux variables non entraînées (humidité du sol : corrélation 0,86 avec ERA5-Land).

Acuña Espinoza et al. 2025, HESS 29, 1277-1294 (page lue). Architecture δ(γt,βt) ; entraînement sous la crue de 5 ans, test au-dessus. NSE médian LSTM 0,75, hybride 0,71, HBV 0,64 ; différences inférieures à la variabilité entre graines ; l'hybride produit des pics plus hauts aux événements les plus rares ; βt dynamique collé à ses bornes aux fortes intensités.

Kraft et al. 2022, HESS 26, 1579-1614 (H2M, page lue). Un LSTM produit cinq coefficients variables par cellule de 1° : fractions de partage de l'eau liquide à somme un (softmax), coefficient de fonte, fraction évaporative ; neige, bilan et nappe restent des équations. Cibles GRACE, FLUXCOM, GRUN, GlobSnow en z-score. Retrait spatial par cellules entrelacées et retrait temporel. Limites : cibles elles-mêmes produits de modèles ; variabilité interannuelle de l'ET surestimée.

## 3. LSTM à conservation de masse et sa critique

Hoedt et al. 2021, ICML (MC-LSTM) : portes conservatives ; états internes corrélés au SWE. Frame et al. 2022, HESS 26 : le LSTM bat tous les modèles sur les extrêmes hors échantillon ; les contraintes de bilan de masse ont réduit la performance hors échantillon. Frame et al. 2023, Hydrological Processes : deux forçages en désaccord sur la précipitation ; la fermeture du bilan réduit la compétence, le réseau s'adapte aux biais des données. Lecture pour meandre : un modèle conservatif paie le biais de la pluie ; c'est le mécanisme du constat sur CaSR, défaut du forçage, pas raison d'ouvrir le bilan.

## 4. LSTM régionaux : ce que leur protocole prouve

Kratzert et al. 2019, WRR 55 : 531 bassins, k-fold spatial, NSE médian 0,69 contre 0,64 pour SAC-SMA calé localement ; mêmes années en entraînement et test.

Arsenault et al. 2023, HESS 27, 139-157 (page lue). 148 bassins HYSETS du nord-est, leave-one-out ; NSE médian LSTM 0,78 contre 0,58 à 0,63 pour 18 combinaisons modèle-régionalisation et 0,67 à 0,68 pour les modèles calés sur le bassin cible. Fuite temporelle confirmée : tout 1979-2018 en entraînement et en évaluation, aucun retrait temporel décrit ; les auteurs reconnaissent que le filtre « longues séries » favorise le LSTM, qu'il interpole bien mais est imprévisible hors domaine. Seule preuve indirecte de l'inflation : Feng 2023, où le LSTM perd 0,11 de NSE et 0,15 de KGE entre PUB et PUR.

Kratzert et al. 2024, HESS 28, 4187-4201 (opinion, page lue). 81 % des articles 2021-2023 entraînent sur un seul bassin ; recommandations : centaines de bassins, dix graines, découpage 1999-2008 / 1980-1989 / 1989-1999, test de Wilcoxon ; aucune combinaison d'attributs ne prédit quel modèle sera le meilleur dans un bassin donné.

## 5. Routage appris

Bindas et al. 2024, WRR 60 (résumé). Un réseau estime le Manning n et la géométrie à partir d'attributs de tronçon dans un Muskingum-Cunge différentiable, entraîné sur les hydrogrammes aval. En synthétique, la géométrie est non identifiable, n l'est avec précision modérée ; gain réel pour les sous-bassins de plus de 2 000 km². Hydrogramme unitaire appris : Feng 2022 (gamma), δHBV2.0. « Convolution conservative » : aucun résultat, non vérifié.

## 6. Mémoire récurrente contre cumuls

Aucune comparaison directe GRU contre cumuls dans un hybride physique. Indices : Hsu et al. 1997, réseau direct à entrées retardées égal au récurrent ; Koya et Roy 2025 (arXiv), le LSTM ne gagne rien avec la longueur de séquence ; Acuña Espinoza 2024, la mémoire récurrente qui pilote des paramètres dynamiques compense une structure fausse. Non prouvé dans un sens ni dans l'autre.

## 7. Neige, partage pluie-neige, gel

Fonte apprise : « Fang et Shen » non vérifié. Ce qui existe : CFMAX régionalisé (Feng 2022), coefficient de fonte variable (Kraft 2022), fonte remplacée dans M100 (Höge 2022, contre-intuitive) ; Charbonneau, Deck, Schneider 2025 (AIES) : équation différentielle neuronale sous contraintes physiques pour le manteau, plusieurs sites SNOTEL, erreur médiane sous 9 % sur la hauteur, transfert à des sites non vus, environ 12 % avec le SWE.

Partage pluie-neige : Jennings et al. 2018, Nature Communications 9 (vérifié) : seuil à 50 % en moyenne 1,0 °C, de −0,4 à 2,4 °C, l'humidité relative contrôle. Jennings et al. 2025, Nature Communications 16 (vérifié) : réseau, forêt aléatoire et XGBoost gagnent au plus 0,6 % d'exactitude sur les meilleurs seuils ; échec entre 1,0 et 2,5 °C. Conclusion : inutile d'y mettre un réseau.

Porte de gel apprise : rien en hydrologie différentiable. Adjacent : schémas de perméabilité du sol gelé dans Noah-MP, gains de 20 à 57 % en débit journalier selon le schéma.

## 8. Demande évaporative apprise

Koppa et al. 2022, Nature Communications 13 (vérifié) : un réseau entraîné sur covariance des turbulences et flux de sève apprend le stress de transpiration, inséré dans GLEAM. GLEAM4 (Miralles et al. 2025, Scientific Data) applique ce schéma : évaporation réelle = potentielle × facteur de stress par réseau, 0,1°, journalier, 1980-2023. Dans tous les cas la place qui marche : une fraction ou un stress borné [0, 1] qui multiplie une demande physique, jamais l'ET elle-même.

## 9. Règle de gestion des réservoirs

Ouyang et al. 2021, J. Hydrology (vérifié) : LSTM pur, 3 557 bassins dont 83 % avec barrage ; NSE médian 0,74, 0,64 en forte régulation ; les réservoirs de moins d'un mois de débit moyen sont absorbés, les grands demandent un module explicite. Yassin et al. 2019, HESS 23 : règle paramétrique par morceaux dans MESH, pas d'apprentissage. Règle apprise dans un modèle différentiable : un résumé EGU 2026 seulement (KGE 0,53 vers 0,59) ; non évalué par les pairs.

## 10. Spatialisation par réseau de coordonnées

Tancik et al. 2020, NeurIPS (vérifié) : biais spectral d'un MLP sur coordonnées, corrigé par traits de Fourier ; preuve en vision. En hydrologie : aucune publication qui spatialise des paramètres par réseau de coordonnées à traits de Fourier ou représentation implicite (trois recherches). Adjacent : GeoINR (GMD 2023) en géologie 3D ; SpLIIF (arXiv 2025) pour la descente d'échelle météo ; une thèse de Tübingen note qu'un encodage positionnel par l'altitude explique la performance en domaine et les problèmes de généralisation spatiale. La régionalisation éprouvée passe par les attributs du bassin, pas par les coordonnées : un réseau de coordonnées est une interpolation.

## 11. MOD16 et produits alternatifs

Critiques : Wang et al. 2015, J. Hydrometeorology 16 : MODIS incertain en saison froide et végétation éparse sur le Canada. Velpuri et al. 2013, RSE : incertitude jusqu'à 25 % à l'échelle des bassins. Mu et al. 2011 : biais journalier −0,29 mm/j aux tours AmeriFlux. Paramètre β fixé à 0,2 kPa pour tous les biomes. Aucune validation de MOD16A2 publiée sur le Québec ou les tours boréales canadiennes (trois recherches).

Alternatives : GLEAM4 (0,1°, journalier, 1980-2023) ; SSEBop v6.1 (1 km, décadaire, depuis 2012, l'USGS avertit qu'un bilan absolu demande une correction locale) ; PML-V2 (500 m, 8 j, 2000-2023, semblable à MOD16 sur sites agricoles) ; FLUXCOM-X (0,05°, horaire, surestime en zones sèches) ; ETMonitor (non vérifié). Aucune n'a de validation boréale canadienne publiée trouvée.

## 12. Synthèse

Places AVEC preuve de généralisation hors échantillon : (1) paramètres statiques spatialisés par réseau depuis les attributs, physique différentiable en aval (seule place qui gagne en retrait de régions entières) ; (2) fraction ou stress borné [0, 1] pour l'évaporation, multipliant une demande physique ; (3) partage de l'eau liquide par fractions à somme un, bilan fermé ; (4) Manning n appris depuis les attributs de tronçon, identifiable avec précision modérée, gain au-dessus de 2 000 km².

Places SANS preuve ou avec preuve contraire : (1) paramètres dynamiques pilotés par LSTM ou GRU (compensent une structure fausse) ; (2) remplacement de processus entiers par réseau calé bassin par bassin ; (3) partage pluie-neige par réseau (gain ≤ 0,6 %) ; (4) porte de gel apprise, noyau de transfert conservatif appris, règle de gestion apprise différentiable : aucune publication ; (5) réseau de coordonnées pour spatialiser des paramètres hydrologiques : aucune évaluation ; risque documenté de bonne interpolation et mauvaise généralisation ; (6) LSTM pur régional : preuve d'interpolation sur les mêmes années seulement.

Protocoles d'épreuve rapide recommandés : deux retraits distincts (régions entières et décennie non vue) ; juger par la CHUTE de score entre interpolation et extrapolation ; vérifier les variables non entraînées contre une source indépendante avant tout entraînement ; test de structure fausse (si un composant appris améliore autant un modèle absurde que le vrai, il compense) ; test d'identifiabilité synthétique (simuler avec paramètres connus, réapprendre depuis le débit) ; dix graines et test de Wilcoxon ; stress d'extrêmes (entraîner sous la crue de 5 ans) ; deux forçages en désaccord.

Non vérifié : Fang et Shen sur la fonte ; modèles δ de neige ou d'ET ; règle de gestion apprise différentiable ; convolution conservative ; porte de gel apprise ; ETMonitor ; SSEBop v6 (Senay 2023) ; chiffres de Jiang 2020, Tsai 2021 et Koppa 2022 au-delà du résumé ; absence de sublimation dans MOD16 ; toute validation canadienne boréale des produits d'ET.

Sources lues : hess.copernicus.org/articles/27/2357/2023 ; arxiv.org/abs/2203.14827 ; osti.gov/pages/biblio/2555891 ; hess.copernicus.org/articles/26/5085/2022 ; hess.copernicus.org/articles/28/2705/2024 ; hess.copernicus.org/articles/29/1277/2025 ; hess.copernicus.org/articles/26/1579/2022 ; hess.copernicus.org/articles/26/3377/2022 ; hess.copernicus.org/articles/27/139/2023 ; hess.copernicus.org/articles/28/4187/2024 ; repository.library.noaa.gov/view/noaa/63586 ; hess.copernicus.org/articles/23/3735/2019 ; pmc.ncbi.nlm.nih.gov/articles/PMC5861046 ; arxiv.org/pdf/2006.10739 ; bg.copernicus.org/articles/21/5079/2024 ; proceedings.mlr.press/v139/hoedt21a.html.
