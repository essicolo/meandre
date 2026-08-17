# Prétraitement des données météorologiques pour la modélisation hydrologique — revue de littérature

Préparée le 2026-08-17 (demande d'Essi, contexte : écarts de KGE de 0.05 à 0.15 entre produits de forçage sur les 15 régions). Toutes les références ont été vérifiées en ligne ; les exceptions sont marquées [à vérifier]. Les pièges de citation relevés sont en fin de document.

## 1. Interpolation des stations

Le message central : la DENSITÉ du réseau prime sur la méthode, et l'apport de l'élévation dépend du pas de temps. Goovaerts (2000, J. Hydrology 228) : au mensuel et à l'annuel, KED et cokrigeage avec élévation battent tout, et le krigeage ordinaire bat déjà Thiessen et IDW. Ly et al. (2011, HESS 15) : au JOURNALIER, l'élévation n'aide pas (OK et IDW gagnent) ; performance stable de 8 à 70 postes, effondrement à 4. Tobin et al. (2011, J. Hydrol. 401) : en terrain alpin le KED redevient payant, surtout via les gradients de température pour la partition pluie/neige. Hofstra et al. (2008, JGR 113) : le contrôle dominant du skill est la densité.

Canada : grilles NRCan = ANUSPLIN (Hutchinson et al. 2009, JAMC 48 ; McKenney et al. 2011, BAMS 92). Validation Hutchinson 2009 : MAE 1.1 °C (Tmax), 1.6 °C (Tmin), 2.9 mm/j, et ATTÉNUATION DE 8 % DES 95es CENTILES de précipitation, un biais de lissage directement pertinent pour les crues. PRISM (Daly et al. 1994 ; 2008) n'existe pas pour le Québec. Abbasnezhadi et Wang (2024, Atmosphere-Ocean 62) : l'interpolation optimale de CanGRD est la moins précise des méthodes testées dans le nord clairsemé.

Québec : Bajamgnigni Gbambie et al. (2017, J. Hydrometeorology 18), LA référence pour nous : sur 181 bassins avec HSAMI, CaPA gagne au centre et au nord (l'ébauche atmosphérique domine où le réseau est clairsemé), la grille stations MDDELCC gagne au sud (réseau dense). C'est la géographie exacte de nos écarts inter-produits. Aussi : St-Hilaire et al. 2003 (densification -> hydrogrammes modifiés, Mauricie) ; Arsenault et Brissette 2014 (seuil de densité, Côte-Nord) ; Bárdossy et Das 2008 (les paramètres calés sont SPÉCIFIQUES au réseau de calage).

## 2. Sous-captation (undercatch)

Goodison, Louie et Yang 1998 (WMO/TD-872) : référence DFIR ; captation ~60 % à 5 m/s (single-Alter), 15-20 % à 8 m/s [tables vérifiées via source secondaire]. WMO-SPICE (Nitu et al. 2018 ; Kochendorfer et al. 2017, HESS 21) : sous-captation moyenne du solide 34 % sans bouclier, 24 % avec single-Alter ; moins de 50 % de captation au-dessus de 5 m/s. Fonctions de transfert universelles CE=f(vent,T) -> biais résiduel ~2 % (Kochendorfer et al. 2018, HESS 22). Pluie : 3-10 %, jusqu'à 10-20 % en site exposé (Pollock et al. 2018, WRR 54).

Conséquences : correction qui DOUBLE la précipitation annuelle en Arctique (Liljedahl et al. 2017, WRR 53). Pan et al. (2016, The Cryosphere 10), crucial pour le Québec boréal : correction jusqu'à +163 % en site ouvert venté mais MOINS DE 13 % SOUS COUVERT FORESTIER, l'abri forestier module tout. Stisen et al. (2012, HESS 16) : caler sur pluie non corrigée -> paramètres biaisés. Jeux canadiens ajustés : Mekis et Vincent 2011 (Atmosphere-Ocean 49, +48.8 % cumulé à Resolute) ; Wang et al. 2022 (ESSD 14) : ajustements médians ECCC +74 % en janvier, moins de 1 % en juillet.

## 3. Réanalyses et produits sur grille

CaPA (Mahfouf et al. 2007) ; évaluation de référence Lespinas et al. 2015 (J. Hydrometeorology 16, souvent cité à tort « Fortin et al. 2015 ») : surestime les petits événements (moins de 2 mm), sous-estime les gros, meilleur hors été ; en hiver le contrôle qualité rejette les jauges suspectes et l'analyse retombe sur l'ébauche du modèle dans le Nord (Fortin et al. 2018, Atmosphere-Ocean 56). CaSR/RDRS : Gasset et al. 2021 (HESS 25) ; AUCUNE évaluation hydrologique publiée avec KGE sur le Québec [à vérifier], nos mesures internes sont peut-être les premières.

ERA5 (Hersbach et al. 2020) ; biais de bruine documenté par Lavers et al. (2022, QJRMS 148) : précipitation trop fréquente et trop faible, jours humides surestimés, fortes intensités sous-estimées. C'est exactement ce que notre dé-crachinage adresse, et le même patron que CaPA (Lespinas 2015). Tarek et al. (2020, HESS 24, 3138 bassins) : ERA5 calé équivaut aux observations sur l'est du Canada, biais humide au nord. Essou et al. (2016, 2017, J. Hydrometeorology) : l'avantage des observations disparaît quand la densité chute. Daymet V4 (Thornton et al. 2021, Scientific Data 8) : pas de correction d'undercatch, densité effondrée au nord du 55e. EMDNA (Tang et al. 2021, ESSD 13) et EM-Earth (Tang et al. 2022, BAMS 103) : gains maximaux aux hautes latitudes peu jaugées. Intercomparaison canadienne : Wong et al. 2017 (HESS 21, pas Atmosphere-Ocean) : WFDEI[GPCC] et CaPA devant ANUSPLIN ; tous les produits stations sous-estiment sévèrement au-dessus de 60 N ; hiver = pire saison. Rapaić et al. 2015 : dispersion maximale des produits dans l'Arctique canadien.

## 4. Correction de biais et désagrégation

Quantile mapping, trois pièges : (1) grille vers point = inflation de variance, sur-correction du drizzle, extrêmes surestimés (Maraun 2013, J. Climate 26) ; (2) univarié détruit la cohérence inter-variables et spatiale, d'où MBCn (Cannon 2018, Climate Dynamics 50) et R2D2 (Vrac 2018, HESS 22) ; (3) corrompt les tendances des extrêmes en climat futur, d'où QDM (Cannon et al. 2015, J. Climate 28). Critiques d'ensemble : Ehret et al. 2012 (HESS 16) ; Maraun 2016 (Current Climate Change Reports 2). Hydrologie : Teutschbein et Seibert 2012 (J. Hydrology 456-457) : le distribution mapping domine, le scaling linéaire mensuel corrige la moyenne mais ni la variabilité ni les extrêmes et finit dernier. Désagrégation temporelle : familles établies (Koutsoyiannis et Onof 2001, J. Hydrology 246) mais pas d'étude canonique isolant l'impact hydrologique de la désagrégation seule [à vérifier].

Note pour nous : la correction CaSR auto-référencée est conceptuellement un QM par parties ; le piège de Maraun 2013 (aire contre point) s'applique si la cible est ponctuelle plutôt qu'une moyenne de zone.

## 5. Erreur de forçage dans le calage

Cadre BATEA : Kavetski, Kuczera et Franks (2006, WRR 42, deux papiers) : traiter l'erreur de pluie comme du bruit de sortie produit des paramètres biaisés et des intervalles surconfiants. Renard et al. 2010 (WRR 46) : la décomposition erreur d'entrée / erreur structurelle est NON IDENTIFIABLE sans prior indépendant sur l'erreur de pluie ; Renard et al. 2011 (WRR 47) la rendent fiable via simulation géostatistique conditionnelle. Vrugt et al. 2008 (WRR 44) : même constat par DREAM. Oudin et al. 2006 (J. Hydrology 320) : l'erreur systématique de pluie est absorbée par le recalage AU PRIX de paramètres biaisés ; le débit est bien moins sensible aux erreurs d'ETP qu'aux erreurs de pluie. Andréassian et al. 2001 (J. Hydrology 250) : la variabilité des paramètres calés diminue avec la qualité de la pluie. Beven et Westerberg 2011 (Hydrological Processes 25) : périodes désinformatives (coefficients d'écoulement impossibles) à exclure du calage ; Kauffeldt et al. 2013 (HESS 17) à grande échelle.

Elsner et al. 2014 (J. Hydrometeorology 15) : les différences entre jeux de forçage sont PLUS GRANDES que les différences calage/validation ; les paramètres calés sur un jeu ne sont pas optimaux sur un autre ; Mizukami et al. 2014 confirment en montagne. Fonde notre protocole de recalage par forçage, et explique la compensation forçage-paramètres observée toute la semaine.

## 6. Circularité et fuite d'information hydrométrique (le soupçon SIMAT)

La chaîne complète est citable. (1) Le débit CONTIENT l'information de précipitation : Kirchner 2009 (WRR 45, « doing hydrology backward ») reconstruit la pluie de bassin depuis les fluctuations de débit aussi bien que deux pluviomètres entre eux ; Herrnegger et al. 2015 (HESS 19) : l'inversion échoue en période NIVALE, le solide ne laisse pas de signal immédiat ; Manoj J et al. 2025 (HESS 29) : la refont par LSTM sur 1800 bassins en discutant explicitement la circularité. (2) Des produits majeurs INGÈRENT délibérément le débit : MSWEP (Beck et al. 2017, HESS 21 : sous-captation et orographie inférées de 13 762 stations de DÉBIT ; Beck et al. 2019, BAMS 100) ; PBCOR (Beck et al. 2020, J. Climate 33 : Budyko sur 9 372 bassins) ; Adam et al. 2006 (J. Climate 19 : +6.2 % de précipitation terrestre globale par bilan). (3) Les évaluations hydrologiques de ces produits ne sont donc PAS indépendantes : Abbas et al. 2026 (HESS 30) classent MSWEP premier (KGE médian 0.78) en reconnaissant explicitement l'objection. Standard violé : Klemeš 1986 (Hydrological Sciences Journal 31), valider exige des données non utilisées, directement ou indirectement, dans la construction.

Implication : une correction de forçage informée par les débits (multiplicateurs calés sur Q, sélection du produit au KGE, bilan ajusté sur Q) contamine toute validation sur ces mêmes débits. Notre correction -hyb est défendable parce qu'auto-référencée (timing jour-local, dé-crachinage, bilan SANS jauges hydrométriques). La frontière est là.

## 7. Partition pluie/neige

Jennings et al. 2018 (Nature Communications 9, 17.8 millions d'observations) : seuil moyen à 50 % = +1.0 °C, plage -0.4 à +2.4 °C pour 95 % des stations, climats continentaux secs = seuils plus chauds. Un seuil fixe à 0 °C misclassifie une part substantielle des précipitations près du gel au Québec continental. Les méthodes intégrant l'humidité battent les seuils de température, surtout entre 0.6 et 3.4 °C sous saturation. Harder et Pomeroy 2013 (Hydrological Processes 27, méthode psychrométrique) ; 2014 : le choix de méthode déplace le pic de SWE jusqu'à 160 mm et l'enneigement jusqu'à 36 jours [via citations secondaires]. Jennings et Molotch 2019 (HESS 23), nuance clé pour nous : les sites FROIDS continentaux sont PEU sensibles au choix (moins de 65 mm, 1.8-4.0 %) contre plus de 200 mm en maritime chaud ; la fraction tombant entre 0 et 4 °C explique 80 % de la variance de sensibilité. L'enjeu québécois se concentre sur les saisons de transition. (Cohérent avec notre mesure : le seuil calibré à -2.2 °C n'a rien changé aux scores.)

## 8. Recommandations pratiques

1. Correction d'undercatch : nécessaire en climat froid, mais MODULÉE PAR L'ABRI FORESTIER (Pan 2016) ; fonctions SPICE (Kochendorfer 2017/2018) ou jeux déjà ajustés (Mekis-Vincent 2011, Wang 2022), pas de facteur uniforme.
2. Dé-crachinage CaSR : appuyé par la littérature (Lavers 2022, Lespinas 2015) ; cible météorologique ou de bilan, JAMAIS les débits de validation.
3. Tout changement de forçage exige un recalage (Elsner 2014) : pas de transfert naïf de paramètres.
4. Valider un forçage SANS les débits : cohérence de Budyko à long terme (mécanique de PBCOR ; Greve et al. 2015), tours de flux et MODIS (MAE ~24 %, Velpuri et al. 2013), SWE des lignes de neige (CanSWE : Vionnet et al. 2021, ESSD 13, 2 607 sites dont le Québec ; Meyer et al. 2012 : le SWE révèle la sous-captation que les jauges cachent), triple collocation (Alemohammad et al. 2015 ; Massari et al. 2017), fermeture multi-sources (Wong et al. 2021, J. Hydrometeorology 22).
5. Exclure du calage les périodes désinformatives identifiées indépendamment du modèle (Beven et Westerberg 2011).

## Table synthèse

| Choix | Risque | Référence |
|---|---|---|
| Interpolation en réseau clairsemé (nord) | sous-estimation sévère, lissage des extrêmes (-8 % au 95e centile) | Hutchinson 2009 ; Gbambie 2017 ; Wong 2017 |
| IDW/OK sans élévation au journalier | faible hors terrain alpin | Ly 2011 ; Tobin 2011 |
| Ignorer l'undercatch neige | 24-34 % de solide manquant, plus de 50 % par vent fort ; paramètres biaisés | Kochendorfer 2017 ; Stisen 2012 |
| Correction uniforme (sans abri forestier) | sur-correction massive sous couvert boréal | Pan 2016 |
| Réanalyse brute | bruine trop fréquente, gros événements sous-estimés | Lavers 2022 ; Lespinas 2015 |
| QM grille vers point | inflation de variance, extrêmes faussés | Maraun 2013 ; Ehret 2012 |
| Changer de forçage sans recaler | paramètres non transférables | Elsner 2014 ; Bárdossy-Das 2008 |
| Corriger ou sélectionner le forçage avec Q | CIRCULARITÉ (standard de Klemeš violé) | Kirchner 2009 ; Beck 2017/2020 ; Klemeš 1986 |
| Seuil pluie/neige fixe 0 °C | seuil réel +1.0 °C (-0.4 à +2.4) ; enjeu = saisons de transition | Jennings 2018 ; Jennings-Molotch 2019 |
| Calage sur périodes désinformatives | bilans impossibles, paramètres biaisés | Beven-Westerberg 2011 |

Limites : chiffres internes de quelques papiers paywallés non extraits (Lespinas 2015, Oudin 2006, Andréassian 2001, Teutschbein-Seibert 2012, Harder-Pomeroy 2013) ; pas d'évaluation hydrologique RDRS-Québec publiée trouvée. Pièges de citation relevés : Lespinas et al. 2015 (pas « Fortin et al. 2015 ») ; Wong et al. 2017 dans HESS (pas Atmosphere-Ocean) ; Thornton et al. 2021 dans Scientific Data (pas ESSD).
