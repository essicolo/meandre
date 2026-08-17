# Enquête documentaire : la grille météo « SIMAT » (Grilles climatiques quotidiennes du MELCCFP) et l'hypothèse de circularité

Date : 2026-08-08. Enquête menée sur sources publiques uniquement (5 angles de recherche parallèles ; 2 angles interrompus par la limite de dépense : littérature INRS/Ouranos approfondie et produits comparables NRCan/CaPA, partiellement couverts par les autres). Rien n'est fabriqué ; ce qui n'a pas été trouvé est marqué [non trouvé].

## 1. Ce qui est documenté

### 1.1 Le produit et son vrai nom

SIMAT n'est pas le nom de la grille : c'est le Service de l'information sur le milieu atmosphérique (aujourd'hui Direction de la qualité de l'air et du climat, DQAC), producteur du produit. Le produit s'appelle « Grilles climatiques quotidiennes » (ou Grille climatique quotidienne, GCQ).

Document de référence, retrouvé sur BAnQ (PDF complet lu) : Bergeron, Onil (2016), Guide d'utilisation 2016 - Grilles climatiques quotidiennes du Programme de surveillance du climat du Québec, version 1.2, MDDELCC, Direction du suivi de l'état de l'environnement, 33 p. URL : http://collections.banq.qc.ca/ark:/52327/bs2545297

Citations exactes (guide Bergeron 2016) :

- Producteur et finalité hydrologique : « Le Service de l'information sur le milieu atmosphérique (SIMAT) de la Direction du suivi de l'état de l'environnement (DSEE) [...] gère le Programme de surveillance du climat du Québec (PSC) » (p. 1) ; « Le Centre d'expertise hydrique du Québec (CEHQ) a collaboré à la création de ce produit afin de répondre à ses besoins de modélisation hydrologique. » (p. 1). Collaboration créditée : « Simon Lachance-Cloutier — Soutien technique et scientifique, Direction de l'expertise hydrique, MDDELCC » (p. iii).
- Variables : « La version 1.2 des données sur grille concerne les précipitations totales quotidiennes (solides et liquides), les températures minimale et maximale quotidiennes et la variance d'interpolation associée à chacun de ces trois phénomènes. » (p. 1).
- Domaine : 0,1° x 0,1° (~10 km), lon -81,5° à -55°, lat 43° à 63°, 1961 à aujourd'hui, mise à jour mensuelle, projection Lambert conique conforme (parallèles 46/60°, GRS80).

### 1.2 Les données de base krigées

- « Les mesures prises aux stations avec observateurs et aux stations automatiques du PSC ont servi de données d'observation pour l'interpolation. » (p. 1). La valeur quotidienne retenue est « la définition de données prioritaire disponible à chaque pas de temps » (hiérarchie du tableau 1 du guide). Journée climatologique du PSC (cumul se terminant le matin, heure locale).
- Ajout de « 41 stations d'Environnement Canada situées au nord du 49e parallèle afin d'assurer une couverture spatiale minimale du Nord du Québec » (p. 1), en journée civile avant 2010.
- Contrôle qualité en deux étapes : « prévalidation » automatique (« plage de valeurs raisonnables, séquence de valeurs constantes, différence attendue entre valeurs successives, entre deux instruments à une même station ou entre stations voisines ») puis validation humaine graphique et cartographique (p. 1).
- Rien d'autre : pas de radar, pas de fond de modèle NWP, pas de réanalyse, pas de covariable hydrologique.

### 1.3 La méthode d'interpolation

- « c'est le krigeage ordinaire qui est utilisé, car il ne requiert pas de connaître la moyenne de l'ensemble du domaine spatial ni de présumer de sa stationnarité » (p. 2).
- Variogramme sphérique réestimé chaque jour (« Le variogramme est construit pour chaque journée séparément »), distance max 200 km, voisinage = les 10 stations les plus proches, estimation des paramètres sans pondération (p. 2-3). Outils : Matlab + gstat 2.5.1 via mGstat 0.991 ; stations à moins de ~100 m fusionnées (option average=1).
- Précipitations sans transformation : « Étant donné qu'aucun consensus ne se dégage de la littérature consultée, aucune transformation n'est appliquée et les valeurs nulles de précipitations sont utilisées. » (p. 3). Post-traitement : précip négatives forcées à 0 ; Tmax forcée >= Tmin ; jours à palier de variogramme nul (précip éparses < 1 mm) forcés à zéro partout.
- Seule correction physique de toute la chaîne : « un gradient thermique correspondant à 0,5 °C par 100 mètres d'altitude est soustrait aux valeurs observées avant l'interpolation et additionné aux estimations après l'interpolation » (p. 3), sur les températures uniquement.
- Validation croisée (années 1961, 1971, 1981, 1991, 2001, 2011) : au sud du 50e, ME < ±0,1 mm et ±0,1 °C, RMSE < ~4 mm (précip) ; au nord du 50e, « les estimations sont beaucoup moins justes et précises » (p. 4-5). Hors Québec : « L'utilisation du jeu de données sur grille pour cette région est déconseillée » (p. 5).

### 1.4 Ce que l'Atlas hydroclimatique documente (forçage et calage)

Rapport technique Atlas 2022 (140 p.) : https://www.cehq.gouv.qc.ca/atlas-hydroclimatique/rapport-atlas-hydroclimatique-2022.pdf ; guide 2022 : https://cehq.gouv.qc.ca/atlas-hydroclimatique/guide-atlas-hydroclimatique-2022.pdf ; Atlas 2018 : https://www.cehq.gouv.qc.ca/hydrometrie/atlas/atlas_hydroclimatique.pdf ; Atlas 2013 : https://cehq.gouv.qc.ca/hydrometrie/atlas/Atlas_hydroclimatique_2013.pdf ; portail : https://www.cehq.gouv.qc.ca/atlas-hydroclimatique/

- Le forçage est bien cette grille : « Les données météorologiques utilisées pour le calage du modèle hydrologique et les simulations hydrologiques de la période de référence sont celles produites par la Direction de la qualité de l'air et du climat (DQAC) du MELCCFP (Bergeron, 2016). [...] Les données météorologiques pour la période 1962 à 2022 ont été extraites des grilles de la DQAC et formatées pour Hydrotel. » (section 2.3, p. 27).
- Calage Hydrotel : 70 stations retenues (sur 533), 2007-10-01 à 2017-09-30, pas 24 h, KGE, optimiseur DDS (500 simulations), calage global 2 régions (Rive-Nord, Rive-Sud, 35 bassins chacune), « KGE médian de 0,79 en calibration globale » ; plateformes LN24HA et MG24HA plus quatre plateformes « ajustées manuellement [...] à l'aide d'un jugement d'expert et par itérations » ; validation 151 stations 1962-2020, KGE médian 0,72 à 0,78 (sections 2.5.2-2.5.3, p. 30-33).
- La compensation des biais météo par le calage est ASSUMÉE : « La calibration a donc pour rôle d'estimer les paramètres et d'ainsi compenser les approximations et les simplifications liées à la modélisation, de même que les données météorologiques imparfaites. » (2.5.2, p. 30-31).
- Deux paramètres calés sur les débits agissent directement sur l'entrée météo : THI, « température seuil de passage de la pluie en neige du modèle d'interpolation des données météorologiques de Thiessen » (borné -5 à +5 °C), et le coefficient multiplicatif d'ETP borné 0,25-1,00 (un bouton de bilan volumique pur). L'annexe D fixe Gradient Température : 0 et Gradient Précipitation : 0 (Thiessen).
- Les séries de débit diffusées (Portrait) assimilent les débits observés : interpolation optimale dont « la seconde étape est la production du champ d'essai, soit une simulation hydrologique » et qui « permet d'honorer les valeurs aux stations » (poids ~4:1 en faveur de l'observation à la station, corrélation nulle au-delà de ~200 km).
- Faiblesse de grille reconnue : « parmi les quelques stations de la Côte-Nord [...] plusieurs ont des scores inférieurs à 0,5. Cela s'explique par un problème connu avec l'interpolation des grilles climatiques pour cette région. »
- Débits d'hiver : « les débits observés ayant été corrigés à cause des effets de glace n'ont pas été considérés. Afin de compenser cette perte d'observations au cours de l'hiver, un poids de 30 jours a été attribué à chaque mesure de jaugeage hivernal ».

### 1.5 Corroboration secondaire

- Rapport Ouranos proj-202025 (Poulin) : décrit la « Grille Climatique Quotidienne (GCQ) » comme générée « par krigeage ordinaire des observations quotidiennes des stations du MELCC » (extrait de moteur de recherche ; fetch direct refusé 403, citation non vérifiée mot à mot). URL : https://www.ouranos.ca/sites/default/files/2023-10/proj-202025-ee-poulin-rapportfinal_0.pdf
- Page U. Toronto (nordata.physics.utoronto.ca, serveur injoignable au fetch) : la DEH aurait une reconstruction 1900-2010 d'apports verticaux « interpolées sur une grille de 10 km », accessible « via une demande à Info-Climat ».

## 2. Ce qui est ABSENT de la documentation publique (information en soi)

- [non trouvé] Bergeron, O., 2014, « Données climatiques sur grille version 1.1 - Choix de la méthode d'interpolation » : cité dans le guide (p. 7) mais introuvable en ligne. C'est le document qui justifierait les choix méthodologiques.
- [non trouvé] Toute version postérieure à la v1.2 (2016). Le guide annonce « Une version ultérieure incorporant des stations limitrophes du Québec sera élaborée », sans trace publique.
- [non trouvé] Le contenu du tableau 1 en détail exploitable (hiérarchie des « définitions de données prioritaires » par station : manuel vs automatique, quels instruments).
- [non trouvé] Toute mention de correction de sous-captation des précipitations (undercatch, neige au nivomètre) dans la grille. Le guide n'en parle pas, donc a priori aucune correction n'est appliquée.
- [non trouvé] Toute page produit ou jeu de données « grille » sur Données Québec ou sur la page des produits Info-Climat (https://www.environnement.gouv.qc.ca/climat/surveillance/produits.htm liste 14 produits, aucune grille). La grille se commande par demande directe à Info-Climat, hors catalogue web.
- [non trouvé] Toute mention, dans les 4 documents Atlas et le guide Bergeron, d'un ajustement de la grille météo (précipitations ou températures) sur les débits observés ou sur un bilan hydrique.

## 3. Verdict sur l'hypothèse de circularité

La circularité forte (grille ajustée sur l'hydrométrie) n'est pas documentée et la méthodologie publiée l'exclut : la grille est un krigeage ordinaire pur d'observations de stations, sans covariable hydrologique, sans correction de bilan. Le CEHQ a collaboré à la définition du produit (Lachance-Cloutier crédité) mais rien dans la méthode décrite n'utilise les débits.

Le paradoxe (grille « bancale » en pluie ponctuelle mais excellents volumes/timing de débit) se résout sans circularité amont, par trois mécanismes documentés ou démontrables :

1. Volume : le krigeage ordinaire est un interpolateur exact et non biaisé de cumuls quotidiens mesurés au sol ; sur un bassin de milliers de km², la lame moyenne de bassin est directement contrainte par les pluviomètres, même si le champ est lissé (portée 200 km, 10 voisins, intensités convectives écrasées). CaSR tire son eau du bilan du modèle GEM corrigé par CaPA, avec biais humide documenté (crachin, queue haute). SIMAT est fausse au point mais juste en moyenne de bassin, exactement la quantité qu'Hydrotel intègre à 24 h.
2. Timing : la journée climatologique (cumul finissant le matin local) est presque alignée sur la réponse hydrologique journalière des jauges CEHQ ; l'agrégation UTC de CaSR décale les événements (lag de pic +2 j mesuré sur CaSR dans meandre, réduit par la correction jour-local).
3. Aval : le résidu de biais est absorbé par le calage (compensation assumée par le rapport 2022 ; THI repartitionne pluie/neige, le multiplicateur d'ETP ajuste le bilan) et, pour les séries diffusées de l'Atlas, par l'assimilation directe des débits observés (interpolation optimale qui honore les jauges). L'excellent débit aux tronçons jaugés vient de là, pas de la seule qualité de la chaîne météo+modèle.

Autrement dit : la grille et les jauges voient le même monde au sol (pluviomètres et débitmètres échantillonnent les mêmes événements réels aux mêmes jours), pendant que CaSR voit le monde d'un modèle atmosphérique. Nuance importante pour CaPA : il assimile bel et bien des stations au sol, mais son contrôle qualité écarte les jauges en précipitation solide par vent (sous-captation), donc l'hiver québécois retombe largement sur l'ébauche du modèle, et les réseaux partenaires provinciaux n'ont été intégrés que tardivement et partiellement selon les versions.

## 4. Inventaire des modules de correction CaSR existants dans meandre

Quatre modules dans .runs/slso/, trois familles, avec leur statut circulaire/non-circulaire :

- build_casr_corrected.py (casr-corr, canonique actuel, held-out 0.678) : jour local UTC-5 (option DST), dé-crachinage horaire (seuil 0,3 mm/h), recalage du volume global sur le bilan flux-tower (1147 mm/an = ET 450 + Q 697). AUTO-RÉFÉRENCÉ, défendable : aucune jauge de débit du domaine n'entre dans la correction (le 697 mm/an est une cible de littérature/bilan, pas un calage par station).
- build_casr_corr_spatial.py (casr-corr2) : recalage du volume PAR SOUS-BASSIN JAUGÉ (cible = lame observée période train + ETR 450, facteur borné [0.75, 1.30]). CIRCULAIRE PAR CONSTRUCTION : l'hydrométrie est injectée explicitement dans P. Défendable seulement parce que c'est borné, train-only et déclaré ; à documenter comme tel dans tout papier.
- build_casr_qm.py / build_casr_qm2.py (qmcasr) : quantile mapping par nœud de la précip CaSR sur la CDF de quebec.zarr (grille krigée de stations, cousine de la GCQ), séquence CaSR préservée ; qm2 ne transfère que la forme (volume CaSR gardé). NON CIRCULAIRE (stations météo seulement) mais climatologique : ne transfère ni le volume ni le timing jour par jour. A perdu contre casr-corr (0.634 vs 0.678, note 2026-07-07), ce qui ne teste que la forme, pas le merging événementiel.

## 5. Chaînon manquant (À FAIRE PLUS TARD, pas maintenant : priorité = modéliser sur PHYSITEL)

Conditional merging journalier (build_casr_merge.py) : ratio JOURNALIER lame krigée stations / lame CaSR au même nœud (même support grille contre grille, donc pas le piège point vs aire qui a coulé la fusion GHCN), appliqué au champ horaire CaSR, en journée climatologique locale. Garde-fous : dé-crachinage avant le ratio, ratio plafonné, jours à zéro krigé forcés à zéro. Résultat attendu : volume et date de chaque événement selon les pluviomètres, intensité sous-journalière de CaSR préservée (nécessaire à l'hortonien). Référence de champ journalier : la GCQ elle-même (demande à Info-Climat) ou un krigeage maison des stations PSC. Non circulaire (stations météo seulement). L'échec du QM ne condamne pas cette piste : le QM alignait les distributions sur 25 ans, le merging aligne chaque journée.

## 6. Questions à poser par courriel à Info-Climat / DQAC (test de l'hypothèse de circularité)

Contact : Service Info-Climat, DQAC (ex-SIMAT), MELCCFP (coordonnées dans le guide Bergeron, p. ii).

1. Version et traçabilité : quelle est la version actuelle des Grilles climatiques quotidiennes (le guide public s'arrête à la v1.2, 2016) ? Y a-t-il eu des changements de méthode depuis (stations limitrophes annoncées, covariables, transformation des précipitations) ? Peut-on obtenir le rapport interne Bergeron 2014 « Choix de la méthode d'interpolation » (v1.1) ?
2. Circularité directe : à aucune étape (interpolation, contrôle qualité, post-traitement, choix des paramètres du variogramme), des données hydrométriques (débits, lames, bilans de bassin) ou des sorties de modèle hydrologique ont-elles servi à ajuster ou valider les valeurs de la grille ? Si oui, sur quelle période et avec quelle cible ?
3. Collaboration CEHQ/DEH : en quoi a consisté concrètement la collaboration du CEHQ (Simon Lachance-Cloutier) créditée dans le guide ? A-t-elle influencé des choix méthodologiques (par exemple le traitement des zéros, le voisinage, la résolution) sur la base de la performance hydrologique des grilles dans Hydrotel ?
4. Sous-captation et neige : une correction de sous-captation des précipitations (vent, neige, type de nivomètre/pluviomètre) est-elle appliquée aux données de station avant krigeage, ou envisagée ? Comment les mesures manuelles (règle à neige, carottier) et automatiques sont-elles hiérarchisées dans les « définitions de données prioritaires » du tableau 1, et cette hiérarchie a-t-elle changé dans le temps ?
5. Incohérences temporelles connues : quelles stations sont en journée civile plutôt que climatologique après 2010, et existe-t-il une liste des changements du réseau (densité, automatisation) susceptibles d'introduire des inhomogénéités dans les grilles 1961-présent ?

## Sources

- Bergeron 2016, guide des Grilles climatiques quotidiennes (BAnQ) : http://collections.banq.qc.ca/ark:/52327/bs2545297
- Rapport technique Atlas hydroclimatique 2022 : https://www.cehq.gouv.qc.ca/atlas-hydroclimatique/rapport-atlas-hydroclimatique-2022.pdf
- Guide Atlas 2022 : https://cehq.gouv.qc.ca/atlas-hydroclimatique/guide-atlas-hydroclimatique-2022.pdf
- Atlas 2018 : https://www.cehq.gouv.qc.ca/hydrometrie/atlas/atlas_hydroclimatique.pdf
- Atlas 2013 : https://cehq.gouv.qc.ca/hydrometrie/atlas/Atlas_hydroclimatique_2013.pdf
- Portail Atlas : https://www.cehq.gouv.qc.ca/atlas-hydroclimatique/
- Produits Info-Climat (aucune grille au catalogue) : https://www.environnement.gouv.qc.ca/climat/surveillance/produits.htm
- Données Québec, organisation MELCCFP (aucune grille cataloguée) : https://www.donneesquebec.ca/recherche/organization/developpement-durable-environnement-et-lutte-contre-les-changements-climatiques
- Jeu de données Atlas 2018 : https://www.donneesquebec.ca/recherche/dataset/atlas-hydroclimatique-2018
- Rapport Ouranos proj-202025 (Poulin, non vérifié mot à mot) : https://www.ouranos.ca/sites/default/files/2023-10/proj-202025-ee-poulin-rapportfinal_0.pdf
- Description données CEHQ/DEH (U. Toronto, injoignable au fetch) : https://nordata.physics.utoronto.ca/fr/annexes-datasets-descriptions/76-river-discharge/runoff-data-from-centre-dexpertise-hydrique-de-quebec-cehq-direction-dexpertise-hydrique-deh/
