# Annexe 3. Ce que les observations disponibles peuvent juger (agent, 2026-10-22)

Lecture du registre, du journal, des deux bancs d'identifiabilité, inventaire de D:/meandre-data (en-têtes seulement), revue de littérature. Les numéros R renvoient au registre.

## 1. Sources présentes sur le disque

| source | où | période | densité | état |
|---|---|---|---|---|
| Débit (drapeaux CEHQ reconstructed) | 15 bases quebec/<reg>.duckdb, table observations | 2000-01 à 2026-02 | 8 700 (cnda) à 242 000 (slso) jours-stations | complet |
| MOD16A2GF (modis_et) | 15 bases ; HDF dans modis8/ | 2000-01 à 2024-12, pas de 8 j | 3,9 M lignes outv, 4,5 M gasp ; outm version annuelle ancienne | complet sauf outm |
| GRACE (grace_tws) | 13 bases ; grace/CSR_RL0603_mascons.nc | 2002-04 à 2026-05, mensuel | une série par bassin | absent de slso et outm |
| CanSWE (snow_obs) | 14 bases ; canswe/CanSWE-CanEEN_1928-2025_v8.nc | 1980 à 2025-07 | outv 45 000, sagu 42 700, slno 41 700, gasp 6 300 ; mont 391, slso 34, cnde 3 | réseau absent au sud |
| NEISIM | neisim/ (2,5 km) ; cibles derives/auxiliaires/neisim-<reg>.npz pour 15 territoires | 2000-2024 journalier | outv 9 132 jours × 3 412 nœuds | complet ; c'est un modèle |
| RSESQ (puits) | derives/auxiliaires/rsesq-niveaux-journaliers.parquet, rsesq-puits.parquet | 1968 à 2026-08 ; 5 000 jours en médiane par puits sur 2001-2024 | 981 561 jours-puits, 185 puits rattachés (mont 46, slso 53, gasp 20, outv 27) ; 419 000 lignes en nappe libre | chargé à côté des bases |
| SWOT | swot/ : 99 granules L2 RiverSP Reach | 2023-08 à 2024-05, tous les 21 jours | 1 008 tronçons de plus de 100 m sur six territoires (335 outv) | téléchargé, pas de table, pas de terme |
| SSEBop v6.1 | ssebop/v61/ : 176 zips mensuels | 2012-02 à 2025-06, 1 km | | comparaison seulement (R358) |
| Humidité du sol ESA CCI / C3S | esa_cci_sm/ | 2005-2024 mensuel, 0,25° | | comparaison seulement (R362) |
| Barrages | table dams dans 13 bases ; sources/npdb | sans date | | géométrie ; aucune série de niveau |
| Prélèvements et rejets io-eau | table withdrawals (trois versions archivées) | 2001-2024 mensuel | 1 700 à 121 600 lignes | complet |
| Température de l'eau | rien | | | aucune source (O10 ouvert depuis 08-24) |
| Niveaux aux jauges (fichiers _N du CEHQ), bathymétrie, LiDAR | sources présents, _N non ingérés | | | O9 ouvert |
| MOD10A1 couverture nivale | modis10/, table modis_snow | 2000- | | réfuté comme contrainte (R24) |

## 2. Les deux bancs d'identifiabilité

information_des_donnees.py, espace des SÉRIES : moyennes mensuelles de chaque source, rang effectif à 95 %, apport propre. Derniers résultats (Outaouais, 2026-09-22) : réseau de neige 12 directions, puits 7, débit 5, NEISIM 2, MOD16 1 (une seule courbe saisonnière), GRACE 1 ; apport propre : neige +7, puits +4, débit +2, GRACE, MOD16, NEISIM 0. Le témoin de bruit (R181) a défait le classement chiffré ; ce qui survit : les produits sur grille sont une direction modulée dans le temps. SWOT (R176) : 19 directions à l'élévation, 19 à la pente, contre 8 pour les six sources réunies ; reproductibilité 0,38. Limite : le pas mensuel détruit la forme des décrues et la date de fonte.

sensibilite_des_observables.py, espace des PARAMÈTRES : un multiplicateur par champ (43), passe avant par blocs, rétropropagation de quatre résumés saisonniers par observable, rang par observable. Appelé par le pilote (MEANDRE_SENSIBILITE=1, ETL_EPOCHS=0). Dernier résultat (R178, 2026-09-22, Outaouais) : dix champs à sensibilité nulle (theta_fc_2, theta_wp_2, theta_wp_3, f_root_1/2/3, f_vert_1/3, k_gw, T_gw) ; champs les plus vus : porosity_3 6,9, theta_fc_3 5,4, K_c 4,6, K_musk_hours 1,9, C_f 0,87 ; les cinq observables couvrent trois directions sur 43 ; seule la neige ajoute une direction. Limites : facteur uniforme par champ (rien sur le motif spatial), quatre résumés saisonniers plafonnent chaque famille. Aucune mesure consignée après le 22 septembre : le résultat vaut pour la physique d'avant le schéma semi-implicite et la couche 3 déclarée.

## 3. Observation par observation

| observation | ce qu'elle contraint vraiment | ce qu'elle ne contraint pas | circularité | verdicts |
|---|---|---|---|---|
| MOD16 (8 j, 500 m) | la FORME saisonnière de l'ET hors neige ; le niveau annuel là où il s'accorde au bilan P − Q ; le niveau de K_c pondéré hors neige ; la reproductibilité entre graines (dispersion d'ET divisée par sept sur 052805, R281) | le contraste spatial (une direction, R173) ; l'ET de décembre à avril (produit non fiable sous neige, R278) | aucune (Penman-Monteith sur LAI/FPAR MODIS et météo GMAO) | forme jamais niveau (R24) ; Châteauguay 1,82 contre 1,48 au bilan sous forçage Budyko (R202) ; en forme 0,59 vers 0,80 de KGE, en absolu la physique corrigée perd (R232) ; deux défauts de code du terme centré corrigés (R273, R310) ; sous CaSR brut MOD16 et P − Q s'accordent à 5 % en Montérégie et Outaouais (R359), ±4 % dans 9 territoires sur 14, excès +6 à +18 % en Gaspésie et Côte-Nord (R378) ; sans MOD16 K_c n'est retenu par rien (R331) et l'été du sud se vide davantage (R385) |
| GRACE (mensuel, bassin) | la PHASE du stockage total ; l'amplitude saisonnière en comparaison | le niveau (25 mm d'incertitude) ; la répartition sol, neige, nappe, retenues ; les réservoirs de barrage de SAGU et OUTV sont dans la masse | aucune | en niveau le terme devient l'objectif et dégrade (R47) ; en forme 15 à 40 % de la perte, empêche d'apprendre (R171-R182) ; retiré du modèle figé le 10-10 (R366). Statut : juge seulement |
| CanSWE | la DATE de disparition du manteau (juste à 0-2 jours) ; l'accumulation appariée site et jour ; le seuil pluie-neige dérivé de la masse (R35) ; 7 à 12 directions là où il existe | le niveau de masse du bassin (sites en clairière ou cuvette) ; rien au sud | aucune | représentativité des sites (R167) ; à 6 % de la perte ne déplace pas le manteau (R170) |
| NEISIM (modèle) | le manteau là où le réseau se tait (slso, mont) ; arbitre de territoire (Gaspésie seule en cause) ; le poids hors neige du terme MOD16 | rien de plus que le réseau là où il existe (R172) | produit de modèle, vérifié contre 246 sites (rapport 0,98, corrélation 0,87) | modèle 16 à 36 % au-dessus de NEISIM au sud (R319) ; sublimation coûte 0,10 de KGE réentraînée (R332) |
| Puits RSESQ | la PRÉSENCE des mécanismes de nappe et la PHASE (extraction depuis la zone saturée bascule la corrélation saisonnière de −0,47 à +0,12, R123) ; transfère aux puits jamais vus (R185) | la VALEUR des paramètres (0,12 de corrélation, R123, R235) ; l'amplitude | aucune | fluctuation reproduite SANS le terme, première corroboration indépendante (R163) ; poids nul dans le socle, retiré du pilote le 10-22 |
| SWOT | élévation et pente de surface libre par tronçon de plus de 100 m : Manning, Muskingum, géométrie, tarage des lacs | le niveau absolu (altitude du lit inconnue) ; rien avant août 2023 | aucune | ÉTABLI dans l'espace des séries (R176) ; sensibilité jamais mesurée (le banc est prêt si la géométrie des tronçons est fournie) ; aucune table ni terme |
| Température de l'eau | rien aujourd'hui | | | O10 ouvert ; aucune source inventoriée |
| Niveaux de lacs et retenues | rien aujourd'hui ; les retenues gérées déplacent l'eau au printemps | | niveaux de retenue seraient l'observable propre | O9, O13, O14 ouverts ; dams ingéré sans séries |
| Prélèvements io-eau | entrée du modèle ; signature d'un prélèvement de 5 % vingt fois la dispersion entre graines mais de l'ordre du biais d'étiage (R233) | | aucune | réingestion 09-25 ; jamais vus par l'entraînement avant le 10-05 |
| Débit observé | tout le reste ; mais de décembre à mars 60 à 87 % des valeurs sont des RECONSTRUCTIONS du CEHQ (R19) | la partition verticale (0,79 sans une goutte d'eau souterraine, R11) ; le plancher d'étiage (R237) | cartes de recharge interdites | |

## 4. MOD16 : établi chez nous, littérature, recommandation

Établi chez nous : le mode absolu est nuisible là où MOD16 contredit le bilan P − Q et indifférent là où ils s'accordent ; sous CaSR brut ils s'accordent à 5 % en Montérégie et Outaouais, l'excès se concentre en Gaspésie et Côte-Nord ; le « 30 % trop haut » venait en grande part du forçage calé ; le terme centré a eu deux défauts de code qui poussaient K_c vers le bas, et plusieurs verdicts de septembre portent ce biais ; pondéré hors neige il fixe le niveau d'ET et divise par sept la dispersion entre graines ; avec les trois autres contraintes il coûte 0,032 de KGE et divise par quatre la dispersion ; dans l'espace des séries il n'apporte rien, dans l'espace des paramètres il regarde K_c, troisième champ le plus sensible, qu'aucune autre source ne regarde ; sous la neige il n'est pas fiable.

Littérature : la seule comparaison canadienne complète est Wang, Pan, Lu, Yang 2015 (J. Hydrometeorology 16, 1540-1560) : MOD16 +32 mm/an soit +7,7 % aux 14 tours Fluxnet-Canada, R² 0,52, pente 0,68, +8 à +24 % aux conifères boréaux, +24 % à Chibougamau (440 contre 355 mm/an), +35 mm/an par bassin contre P − Q de 138 bassins ; biais concentré en saison froide, forme estivale bonne. Causes : météo GMAO à 0,5°, VPD seul stress, paramètres par biome fixes. Nijzink et al. 2018 : MOD16 contraint moins que GRACE, l'humidité du sol et la neige. Stratégies recommandées pour un produit biaisé : jamais le niveau ; forme temporelle centrée ou normalisée ; motif spatial insensible au biais (SPAEF) ; volume contrôlé par le débit ; déclarer le produit et en tester deux (Dembélé 2020). Alternatives sur le Québec : GLEAM4 (0,1°, journalier, 1980-2025, sublimation séparée) pour la forme ; PML-V2.2 (500 m, 8 j) pour le contraste spatial ; EALCO de RNCan (5 km, mensuel, 2000-2023, biais −2 % aux tours, données ouvertes du Canada) comme référence canadienne de niveau ; SSEBop v6.1 déjà sur le disque. Aucune n'a de validation aux tours du Québec.

Lecture du désaccord : le milieu québécois juge MOD16 sur son NIVEAU en forêt boréale, trop haut de 10 à 25 %, et c'est confirmé ; Essi mesure son utilité comme contrainte de FORME sur K_c, autre usage, confirmé aussi. Les deux sont vrais.

Recommandation : garder le terme centré pondéré hors neige comme contrainte de forme et de reproductibilité sur K_c, jamais en absolu ni sous la neige. Ancrer le NIVEAU annuel de l'ET sur l'ET implicite des bassins jaugés (pluie CaSR moins débit observé, déjà calculée par fermeture_bassins.py). Les deux termes posés ensemble, gradients sur K_c sondés avant tout entraînement. Vérifier à Chibougamau (CA-Qfo) et contre EALCO ; un second produit (GLEAM4 ou PML-V2.2) en forme comme témoin. Mesurer la sensibilité SPATIALE, que le banc des paramètres ne perturbe pas.

## 5. Pièces de la physique et leur juge non circulaire

| pièce | observation disponible | ce qu'elle juge | lacune |
|---|---|---|---|
| Fonte | CanSWE apparié site et jour (date de disparition, pertes hivernales) ; NEISIM où le réseau se tait ; débit d'avril mesuré à 90 % | date et vitesse de fonte | aucun site au sud ; MOD10 réfuté |
| Partage pluie-neige | accumulation CanSWE de novembre à février ; NEISIM au sud | le seuil et sa variable | Gaspésie injugeable (R42-R44) |
| Aires saturées | aucune directement ; platitude et variations journalières du débit MESURÉ ; C3S trop grossière | la part rapide et la forme des crues | aucune observation de surface saturée ; SWOT pente à essayer |
| Porte de gel | niveaux RSESQ d'hiver ; débits d'hiver aux seuls jours MESURÉS ou jaugeages datés (R385) ; GRACE en phase | si la nappe cesse de se vidanger sous gel | débits de décembre à mars reconstitués à 60 à 87 % ; profondeur de gel mesurée nulle part |
| K_c saisonnier, phénologie | MOD16 en forme hors neige ; LAI MOD15 pour la phénologie sans simuler (R206) ; ET implicite P − Q pour le niveau | forme saisonnière et niveau annuel | aucune tour de flux dans la perte (Chibougamau existe, non ingérée) |
| Vidange de nappe | récessions observées (Brutsaert-Nieber, b = 2 − 1/n, par station) ; puits pour présence et phase ; Q7min et jours sous Q90 | l'exposant, la présence du mécanisme | conductance et extraction non identifiées (R235, R237) |
| Tarage des lacs, lacs en série | SWOT élévation (1 008 tronçons, depuis 2023) ; niveaux aux jauges _N non ingérés ; retard aux stations aval (R389) | k, beta des lacs, Manning, Muskingum | niveaux de retenue des exploitants non publics ; bathymétrie absente |
| Demande évaporative | ET implicite P − Q pluriannuelle (niveau) ; MOD16 forme hors neige ; GLEAM4 ou EALCO second avis | niveau annuel, forme d'été | aucune mesure de flux dans la perte ; évaporation de l'eau libre hors du diagnostic ETR |
| Stockage total | GRACE en phase (juge seulement) ; puits | que le stock respire au bon mois | GRACE mélange retenues ; rien ne sépare sol et nappe hormis les puits |

Conclusion de l'agent : les observations non circulaires qui contraignent une pièce de façon identifiable sont CanSWE pour la date de fonte et le partage pluie-neige, les récessions pour l'exposant de nappe, les puits pour la présence et la phase des mécanismes souterrains, MOD16 en forme hors neige pour K_c, et le bilan P − Q pour le niveau annuel de l'ET. Tout le reste (aires saturées, gel, tarage des lacs, température, retenues) n'a aujourd'hui aucun juge ou un juge non ingéré (SWOT, niveaux _N, tours de flux). Proposition, sans entraînement : relancer sensibilite_des_observables.py sur la physique d'octobre ; ajouter SWOT au banc des paramètres (code prêt, il manque la géométrie des tronçons) ; ingérer les niveaux _N du CEHQ et la tour de Chibougamau avant toute pièce apprise dans le routage ou la demande évaporative.
