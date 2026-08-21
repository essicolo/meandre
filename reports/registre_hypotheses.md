# Registre des hypothèses — méandre

Le journal `experiment_log.md` est CHRONOLOGIQUE et purement additif : 1514 lignes, 103 entrées, où les faits établis, les hypothèses réfutées et les conclusions devenues caduques cohabitent sans distinction de statut. C'est ce qui a permis au « 0.82 d'Hydrotel » de circuler pendant des jours, et ce qui m'a fait raisonner le 10 août sur des verdicts mesurés avec un modèle qui simulait un territoire sans forêt.

Ce registre est l'inverse : un état COURANT, révisé, où chaque ligne porte un statut. Le journal raconte l'histoire, le registre dit ce qui est vrai aujourd'hui. Toute conclusion citée dans une discussion doit venir d'ici.

Statuts : **ÉTABLI** (mesuré, reproductible, toujours valide) · **RÉFUTÉ** (testé, faux) · **CADUC** (mesuré sur une base depuis corrigée, à refaire) · **OUVERT** (test défini, pas encore fait).

Dernière révision : 2026-08-12.

---

## 1. Faits ÉTABLIS

| # | Fait | Mesure | Date |
|---|---|---|---|
| E1 | La neige du clone est fidèle à Hydrotel | stock pondéré 108.09 contre 108.14 mm (OUTV), 141.80 contre 141.39 (GASP), corr 0.99 | 08-10 |
| E2 | Le sol est fidèle sur OUTV | theta1/2/3 à 0.97 / 0.96 / 1.00, corr 0.95-0.999, aux deux saisons | 08-10 |
| E3 | Le routage est fidèle et n'ajoute aucun biais | corr 0.985-0.997 sur débit AMONT et AVAL, rapports amont ≈ aval | 08-10 |
| E4 | Le forçage `-hyb` équivaut à la météo d'Hydrotel | +3.1 % de volume, corr journalière 0.990 (pluie) et 0.999 (T) ; substitution sans effet mesurable | 08-11 |
| E5 | L'ensemble Hydrotel sur OUTV vaut 0.771 (médian par station) | 6 membres de 0.7531 à 0.8299 ; meilleur par station 0.854 | 08-11 |
| E6 | Méandre ANCRÉ atteint 0.739 sur OUTV, sans entraînement | mêmes 16 jauges, même période, même formule | 08-11 |
| E7 | L'ancrage NE TRANSMET PAS la qualité du calage | +0.077 à la source (LN24HA -> MG24HK) donnent +0.0035 à l'arrivée | 08-11 |
| E8 | L'entraînement PERD 0.134 contre l'ancrage sur OUTV | 0.6051 contre 0.7389, écart validation/tenu de côté de 0.23 | 08-11 |
| E9 | Le plafond de sous-pas de Courant mord d'autant plus que le sol est perméable | GASP (ks 4.6× OUTV) : hiver 0.77 -> 0.97 en passant de 48 à 300 sous-pas | 08-10 |
| E10 | La capacité n'est pas le facteur limitant | 145k paramètres libres non régularisés ne gagnent que +0.002 | 08-05 |
| E11 | L'entraîné est battu par l'ancré sur SA PROPRE période d'entraînement | 0.58-0.67 contre 0.73-0.80 sur 2001-2018 ; ce n'est pas du sur-ajustement mais une optimisation qui n'atteint pas la solution physique | 08-13 |
| E13 | Le champ spatial porte K_sat et les porosités SANS PERTE | tout imposer sauf elles donne 0.7389, égal au sol entièrement imposé (0.7368) ; les 0.145 manquants venaient des épaisseurs, fractions de surface et pente, qui sont des DONNÉES | 08-14 |
| E14 | Ajusté sur le champ d'Hydrotel, le réseau s'appuie sur la TEXTURE, pas sur les coordonnées | sensibilité de K_sat à f_sand 0.471 et f_silt 0.399, contre moins de 0.035 pour tous les autres descripteurs (0.0032 avant ajustement) : la relation est transférable | 08-14 |
| E15 | La pédologie vient de PHYSITEL et n'a que 4 classes sur OUTV | loam 52 %, sable loameux 40 %, deux classes marginales ; jamais comparée à SoilGrids ni à l'IRDA (vérification REPORTÉE, priorité à bien modéliser sur PHYSITEL) | 08-14 |
| E12 | La sélection sur validation gonfle la fenêtre de sélection d'environ 0.05 | l'entraîné culmine à 0.7103 sur 2019-2021 contre 0.60-0.67 sur les fenêtres voisines | 08-13 |

**Le déficit de février est un déficit de STOCKAGE, pas de neige ni de routage** (mesuré 2026-08-19, champion OUTV). Année par année : février 2022, le manteau gagne +60 mm et le modèle ne rend que 0.39 de l'observé ; 2023, +39 mm et 0.62 ; 2024, le manteau perd 5.7 mm et le modèle fait 1.00. Quand la neige fond il est juste au centième, quand elle s'accumule il s'effondre. L'observé, lui, coule à 13-15 m3/s dans les trois cas : la rivière réelle est portée en hiver par une réserve remplie à l'automne. Le champion n'en a pas (recharge 0.26 mm/an, nappe = 0.1 % de la production) : son aquifère est présent mais AFFAMÉ, et son gain de +0.007 ne vient pas de la physique souterraine.

**Le taux de vidange souterrain mesuré est 0.0273 /j** (résidence 37 j), sur 1316 récessions hivernales PURES des jauges d'OUTV (segments décroissants >= 5 jours avec Tmax < 0, donc sans fonte ni pluie). Le champion tourne à 0.0645 : 2,4 fois trop rapide, et 7 fois trop rapide que la composante lente (centile 10 à 0.0090, soit 111 j). Forte variabilité entre stations (9,6 à 66 j) : ce taux doit rester un CHAMP, pas un scalaire.

**LA RÉSOLUTION DE L'INSTRUMENT EST ~0.025** (mesuré 2026-08-20). Les poids du NeRF étaient tirés au HASARD à chaque entraînement (`init_from_literature` ne biaise que la dernière couche) : aucune graine globale n'existait. Quatre tirages de la recette du champion, à zéro époque, donnent 0.7443 / 0.7389 / 0.7271 / 0.7191 — étendue 0.025, écart-type 0.011. Le 0.7432 du 16 août tombe dans cette fourchette : il n'y avait aucune régression, seulement un tirage favorable.

CONSÉQUENCES. (1) Toute comparaison NON APPARIÉE tranchée sur moins de ~0.025 ne vaut rien, y compris le +0.007 de l'aquifère (16 août) et le +0.004 du seuil pluie/neige (19 août). (2) Le repère qui porte l'enjeu, 0.7880 contre 0.7711 pour l'ensemble Hydrotel, est un écart de 0.017 mesuré sur UN tirage : il n'est pas réfuté, il n'est pas encore démontré. Il faut au moins 3 graines. (3) Le remède est gratuit : à GRAINE FIXÉE la comparaison devient appariée, le bruit de tirage disparaît et un écart de 0.005 redevient lisible.

RÈGLE. Une graine (`ETL_SEED`, défaut 1234) est désormais posée par le pilote et le déterminisme est vérifié (deux runs de la même graine : 0.7389 et 0.7389, identiques). Tout A/B se fait à graine fixée, un seul changement. Toute annonce de record se fait sur au moins 3 graines, avec la dispersion.

**LIGNES ROUGES DE CIRCULARITÉ** (documenté 2026-08-20, Leonardini et al., Symposium Ouranos 2025, `docs/papers/04_Leonardini_*.pdf`). Un produit qui a vu nos débits ne peut servir ni de forçage, ni de cible, ni de contrainte : il rendrait toute évaluation circulaire (même mécanisme que l'enquête SIMAT).

| produit | a-t-il vu nos débits ? | usage |
|---|---|---|
| **CaSR-Rivers** | OUI — **1704 jauges HYDAT + USGS assimilées** | INTERDIT |
| Cartes de recharge PACES / HydroBudget / HELP | OUI — calage sur le débit de base séparé des jauges | comparaison en discussion seulement, JAMAIS cible |
| **CaSR-Land** | **NON** — simulation hors-ligne SVS, aucune assimilation (tableau p17) | contrainte de TENDANCE (SWE, humidité profonde) |
| Niveaux RSESQ | NON — mesures directes de hauteur d'eau | contrainte de TENDANCE (dynamique de nappe) |
| GRACE, MODIS ET | NON | contrainte de TENDANCE (déjà câblées) |

Champs de surface de CaSR lui-même : à écarter aussi. Sa version ISBA surestime le débit d'été d'un FACTEUR DIX et son assimilation CaLDAS-Screen est jugée inappropriée pour l'hydrologie par ses propres auteurs (p4-5) ; c'est la raison d'être de CaSR-Land. Nous n'utilisons que ses champs météorologiques.

**LE CALENDRIER DE FONTE EST JUSTE ; L'ÉCART À HYDROTEL EST LA VARIABILITÉ** (mesuré 2026-08-20 contre CanSWE, 9643 couples simulé/mesuré sur 62 sites d'OUTV, protocole du pilote).

Date de disparition du manteau, simulée contre mesurée : 2022 **+2 jours**, 2023 **0**, 2024 **0**. La fonte tombe au bon moment. Ce résultat est immunisé contre le biais de représentativité des sites (une date ne dépend pas d'un niveau).

Masse du manteau : pic simulé sur pic mesuré de 0.59 à 0.80 selon l'année, rapport mensuel de 0.55 à 0.72. Déficit apparent de 20 à 40 %. NON ÉTABLI : c'est exactement l'ampleur du biais d'interception attendu (site en clairière contre tronçon forestier, 20-40 % en forêt boréale). Le test qui tranche est la pente du déficit contre la fraction forestière.

Décomposition du KGE, tenue de côté 2022-2024 : méandre r=0.8908 beta=0.9269 gamma=0.9199 ; Hydrotel MG24HK r=0.8990 beta=0.9497 gamma=0.9672. **Notre corrélation égale celle d'un modèle opérationnel calibré à 0.008 près.** L'écart de 0.042 est porté par le RAPPORT DE VARIABILITÉ (-0.047), puis le volume (-0.023). Le chantier n'est donc ni le timing des événements ni la fonte : c'est un débit TROP LISSÉ, le rabotage des pics, désormais chiffré comme l'essentiel de l'écart au meilleur membre.

**LE SEUIL PLUIE/NEIGE FIXE PERD SA VALIDITÉ QUAND L'HIVER SE RÉCHAUFFE** (mesuré 2026-08-20, données seules : CanSWE + forçage, aucun modèle). Rapport entre la neige que produit notre règle et les hausses MESURÉES du manteap, par période, sur OUTV :

| période | n intervalles | accumulation mesurée | neige de la règle | rapport | T moy DJF |
|---|---|---|---|---|---|
| 2000-2011 | 1845 | 30 090 mm | 36 954 mm | 1.23 | -10.03 °C |
| 2012-2021 | 2360 | 41 516 mm | 46 834 mm | 1.13 | -9.87 °C |
| **2022-2024 (tenue de côté)** | 657 | 10 714 mm | 10 083 mm | **0.94** | **-8.50 °C** |

Décroissance monotone avec le réchauffement, et passage SOUS 1 sur la tenue de côté : la règle ne fournit plus assez de neige pour bâtir le manteau observé, avant toute perte. Mécanisme direct : un seuil à -2.2168 °C bascule d'autant plus de précipitation en pluie que l'hiver est doux ; il a été calé dans un climat plus froid.

SEUIL OPTIMAL PAR PÉRIODE (même jour, données seules) : celui qui rend la neige produite égale aux hausses mesurées vaut **-4.00 °C** sur 2000-2011 (borne de recherche atteinte), **-3.50 °C** sur 2012-2021, **-1.75 °C** sur 2022-2024. Dérive de 2.25 °C dans le sens du réchauffement : AUCUNE CONSTANTE ne convient aux trois périodes. Ce n'est donc pas un mauvais réglage, c'est une structure qui ne se transporte pas dans le temps.

CONFONDANTS TESTÉS ET ÉLIMINÉS (tous le 2026-08-20, données seules) : (a) COMPOSITION DU RÉSEAU -- en ne gardant que les 43 sites présents dans les trois périodes, la dérive persiste et s'accentue (-4.25 / -3.50 / -1.75 °C) ; (b) BIAIS DE TEMPÉRATURE du forçage -- il existe (+0.50 °C en hiver contre 82 stations ECCC, 121 691 jours) mais va À CONTRESENS, il est le plus fort dans la période où nous produisons déjà 23 % de neige en trop ; (c) NOS CORRECTIONS DE FORÇAGE -- neutres en hiver aux nœuds portant un site nivométrique (1.029 / 0.998 / 0.992), donc hors de cause pour ce déficit, même si elles retranchent 11 % aux nœuds voisins des stations météo (correction spatialement structurée, à noter pour ailleurs) ; (d) DÉRIVE DE CaSR BRUT -- stable contre stations (1.07 / 1.07 / 1.03).

(e) PROTOCOLE D'OBSERVATION -- les relevés sont devenus 24 % plus fréquents (écart moyen 15.2 -> 13.5 -> 11.6 jours) et la part d'instruments non manuels est passée de 2 % à 29 %, deux effets qui gonflent la somme des hausses captées sans qu'un flocon de plus ne tombe. En restreignant aux durées 12-18 jours ET à la seule méthode manuelle multi-points, la dérive passe de 2.50 à **2.25 °C** : le protocole n'explique qu'un dixième de l'écart. (f) HUMIDITÉ -- le THERMOMÈTRE MOUILLÉ (équation psychrométrique résolue par bissection, saturation sur glace, pression selon l'altitude) donne EXACTEMENT la même dérive de 2.50 °C, simplement décalée de -1 °C. Ce n'est donc pas la variable manquante, et une sigmoïde sur le mouillé ne corrigerait pas la dérive.

TROIS RÉSERVES. (1) Ces optima sont des BORNES INFÉRIEURES : caler la neige produite sur les hausses mesurées suppose zéro perte entre la chute et le manteau, or il y en a toujours ; le vrai optimum est plus chaud. (2) La composition des sites change d'une période à l'autre, et une dérive du biais de température de CaSR produirait le même effet qu'une dérive physique -- indistinguables ici. (3) Des optima entre -4 et -1.75 °C sont PHYSIQUEMENT IMPLAUSIBLES, la littérature plaçant la transition vers 0 à +1.5 °C : qu'aucune période n'approche cette plage suggère que la FORME FONCTIONNELLE compense autre chose, et pas seulement que le point est mal placé.

PORTÉE. Cela relie deux problèmes traités séparément jusqu'ici : la dégradation sur 2022-2024, attribuée depuis juin à la non-stationnarité climatique, est en partie STRUCTURELLE -- un paramètre fixe de la physique cesse d'être valide quand le climat change. C'est un argument de fond pour le papier, et une raison de préférer une partition APPRISE (sigmoïde à centre et largeur par nœud, idée d'Essi) ou fondée sur le THERMOMÈTRE MOUILLÉ (Jennings et al. 2018), plutôt qu'un seuil scalaire figé. Attention : il ne faut PAS remonter le seuil uniformément, la règle produit déjà 23 % de trop en période froide.

RÉSERVES : n plus faible sur 2022-2024 (657) ; la composition des sites actifs varie d'une période à l'autre ; la monotonie avec la température appuie la causalité sans la prouver.

**CaSR NE SOUS-CAPTE PAS LA NEIGE, ET LA GRILLE EST DISCULPÉE** (mesuré 2026-08-20, test FLUX contre FLUX sans simulation). Sur 175 intervalles franchement froids d'OUTV (Tmax < -2 °C sur TOUT l'intervalle, donc aucune fonte possible, 3 à 31 jours, hausse > 5 mm), le manteau gagne 3647 mm d'EEN pendant que CaSR annonce 4018 mm : **rapport 0.91**, médiane par intervalle 0.90, 38 % des intervalles au-dessus de 1 (dispersion, pas biais). Les 9 % manquants s'expliquent par la sublimation, le transport par le vent et le bruit. L'hypothèse de sous-captation de 10 à 50 % (Kochendorfer et al. 2017), que j'avais avancée le même jour, est RÉFUTÉE sur ce domaine.

Propriété du test : une VARIATION contre une VARIATION, même intervalle, même lieu. Il ne dépend d'aucun niveau ni datum, donc la représentativité du site s'y annule en bonne partie -- contrairement à la comparaison de niveaux de SWE, abandonnée le même jour.

CONSÉQUENCE PAR ÉLIMINATION. La grille fournit le bon volume annuel (beta 0.978 à l'initialisation ancrée) ET la bonne neige en période froide (0.91). Ne restent que deux causes possibles au déficit d'avril : (1) les événements PRÈS DU SEUIL, entre -2.2168 et 0 °C, que le modèle compte en pluie -- ce test les exclut délibérément ; (2) des pertes internes au modèle après la chute. Dans les deux cas c'est un PARAMÈTRE DE MODÈLE, pas une propriété de la grille : aucune correction empirique du forçage n'est requise.

**LE PARTAGE PLUIE/NEIGE EST SUSPECT, MAIS NON DÉMONTRÉ** (2026-08-20 ; conclusion AFFAIBLIE le même jour après objection d'Essi, voir la réserve en fin d'entrée). En fraction de la précipitation cumulée depuis le 1er novembre, au nœud de chaque site, médiane sur les sites :

| année | manteau OBSERVÉ | notre APPORT de neige | notre PIC |
|---|---|---|---|
| 2022 | 0.40 | 0.41 | 0.31 |
| 2023 | **0.42** | **0.39** | 0.16 |
| 2024 | 0.32 | 0.35 | 0.18 |

En 2023 le manteau observé DÉPASSE toute la neige que le modèle reçoit : impossible même avec zéro perte. Les autres années, la marge est nulle ou de trois points, alors qu'un manteau réel perd toujours de l'eau avant son maximum. Le seuil du projet (-2.2168 °C) classe 35-41 % de la précipitation en neige ; un seuil à 0 °C en classerait 49-59 %. RÉSERVE DÉCISIVE (Essi, même jour) : cette conclusion repose sur une comparaison de NIVEAUX, déguisée en bilan de masse. Or la règle posée le matin même était contrainte de TENDANCE, jamais de niveau, à cause de la représentativité des sites. La marge de 2023 est de 8 % quand l'incertitude de représentativité d'un site nivométrique est de 10 à 30 % : **l'impossibilité annoncée tient dans la barre d'erreur**. Si les sites tiennent plus de neige que la moyenne du tronçon, ce qui est le cas attendu, il n'y a aucune contradiction. CE QUI RESTE ROBUSTE, parce qu'indépendant de toute échelle : (1) la DATE de disparition du manteau, juste à +2, 0 et 0 jours ; (2) la borne de VOLUME, il tombe 2.5 à 3 fois plus d'eau que le manteau observé n'en contient, donc le forçage n'est pas court ; (3) le déficit d'AVRIL à 0.729, qui est dans le DÉBIT aux jauges, sans question d'échelle. C'est ce dernier, et lui seul, qui est le fait à expliquer. S'y ajoute une perte de 24 à 58 % entre notre chute de neige et notre pic, dont on ne peut pas dire si elle est excessive sans connaître les pertes réelles. Piste connexe, non testée : les pluviomètres SOUS-CAPTENT la précipitation solide de 10 à 50 % selon le vent, et CaSR assimile ces mesures (fonctions de transfert de Kochendorfer et al. 2017).

**LE DÉFICIT DE NEIGE EST RÉEL, ET IL N'EST PAS UN ARTEFACT DE SITE** (mesuré 2026-08-20). Le manteau simulé contient 20 à 40 % d'eau de moins que le mesuré (pic simulé/mesuré 0.59 à 0.80). L'explication commode -- site en clairière contre tronçon forestier, interception par la canopée -- est RÉFUTÉE : le déficit AUGMENTE quand la forêt DIMINUE (rapport 0.58 au quartile le moins boisé, 0.77 au plus boisé), soit l'inverse exact de l'interception. Mécanisme cohérent avec le reste : le seuil de partage pluie/neige du projet (-2.2168 °C) compte comme pluie tout ce qui est au-dessus ; les tronçons les moins boisés sont les plus bas et les plus chauds, donc ceux où le plus d'événements tombent près du seuil. Cela explique aussi l'excès de débit de décembre (1.207) sur la même période. Chantier ouvert : le partage de phase, désormais arbitrable par une mesure INDÉPENDANTE du débit.

**L'EAU NE a pas déplacée : ELLE MANQUE, ET C'EST AVRIL** (mesuré 2026-08-20, protocole du pilote, écarts en VOLUME et non en rapport). Sur la tenue de côté, en m3/s-jours sommés sur les 16 stations et les 3 ans :

| mois | écart | part du total annuel |
|---|---|---|
| **avril** | **-31 973** | **-5.6 %** |
| octobre | -7 271 | -1.3 % |
| décembre | +7 996 | +1.4 % |
| **ANNÉE** | **-37 098** | **-6.5 %** |

Les écarts NE SE COMPENSENT PAS : le total annuel est de -6.5 %, et avril en porte 86 %. L'excédent de décembre ne vaut que le quart du déficit d'avril. RÉFUTE mon énoncé du même jour selon lequel le modèle relâcherait en décembre l'eau d'avril : un RAPPORT mensuel ne dit rien du VOLUME qu'il représente, et 27 % du volume d'avril, mois de crue, pèse quatre fois 20 % de celui de décembre. Le pilote imprime désormais les deux tables.

CONSÉQUENCE : il faut chercher une PERTE, pas un déplacement. Quatre candidats : l'eau n'entre jamais, elle entre et reste stockée, elle s'évapore, ou elle disparaît par une fuite numérique (cf. dette #42, jamais requalifiée). L'audit de FERMETURE DU BILAN tranche, avec une règle nette : il ferme à la précision machine ou il ne ferme pas.

**LE BILAN D'EAU FERME PARTOUT SAUF AU COUPLAGE DU MILIEU HUMIDE** (mesuré 2026-08-20, audit `ETL_BILAN=1`, règle posée AVANT la mesure : <0.1 % = ferme, >1 % = fuite).

| configuration | erreur de fermeture |
|---|---|
| avec milieu humide | **+1.38 %** de la précipitation (médiane par nœud +1.26 %, q90 +2.83 %) |
| **sans milieu humide** | **+0.00 %** (1 mm sur 24 481 ; q90 +0.01 %) |

Le reste de la colonne -- neige, gel, ETP/ETR, sol BV3C2, aquifère, canopée -- conserve la masse à la PRÉCISION NUMÉRIQUE. Et le réservoir de milieu humide conserve EXACTEMENT la sienne (testé sur 4 régimes : normal, débordement, étiage, vide ; écart nul). Le défaut est donc uniquement dans la SUBSTITUTION entre les deux.

MÉCANISME, lu dans le code et vérifié sur la plus petite unité : la colonne retire de la production `prod x wet_fr`, mais le réservoir ne reçoit que `wetflwi = prod x (wet_fr - wetsa/hru)` -- la production de l'EMPREINTE PROPRE du milieu humide est retranchée sans être créditée. En compensation le réservoir reçoit `wetpcp = apport x wetsa`, la précipitation directe sur cette surface, eau que le sol a DÉJÀ traitée sur tout le tronçon. La différence `(apport - prod) x wetsa/hru` est créée ou détruite selon le signe. Sur l'exemple unitaire : 1.2000 mm retirés, 1.1765 mm crédités.

PORTÉE. Le commentaire du code indique un portage ligne-à-ligne de `bv3c2.cpp` l.838-895 : c'est donc la comptabilité d'HYDROTEL, pas une erreur de notre portage. Si cela se confirme dans leur source, Hydrotel ne conserve pas la masse sur les tronçons à milieux humides, proportionnellement à leur fraction. Vérifiable, et à signaler à l'équipe qui le maintient. C'est aussi un exemple net de ce que la différentiabilité apporte : un bilan auditable terme à terme, là où un modèle compilé le rend invisible.

EFFET SUR LE SCORE : sans milieu humide, tenue de côté 0.7929 contre 0.7880. Écart de 0.005, SOUS le bruit de 0.025 : la correction ne coûtera rien, mais ce n'est pas un gain non plus.

## 2. Hypothèses RÉFUTÉES

| # | Hypothèse | Comment elle est tombée | Date |
|---|---|---|---|
| R1 | « Hydrotel fait 0.82 sur OUTV » | JAMAIS MESURÉ : chaîne écrite dans mes scripts et relue comme une donnée. Réel : 0.7531 (membre LN24HA), 0.771 (ensemble) | 08-11 |
| R2 | Le rabotage des pics vient des bornes de K_musk | À K physique (0.35 h) TOUT se dégrade, y compris la fidélité au réseau d'Hydrotel (0.209 contre 0.335) | 08-09 |
| R3 | Le rabotage est la réponse optimale à une erreur de calage temporel | Avec une référence bien spécifiée, la perte retrouve le bon K pour toute largeur d'événement et tout décalage | 08-09 |
| R4 | Hydrotel translate sans atténuer, méandre diffuse | Erreur de banc : j'injectais l'impulsion en apport LATÉRAL. Sur l'eau d'AMONT, le clone atténue de 43 % PAR tronçon | 08-09 |
| R5 | Le forçage est un plafond de performance | Hydrotel atteint 0.75-0.83 avec 959 mm/an, plus SEC que notre forçage | 08-11 |
| R6 | Nos corrections ont asséché le forçage sous le CaSR brut | Fait juste, jugement faux : elles l'ont ALIGNÉ sur les stations (CaSR brut est à +16 %) | 08-11 |
| R7 | Le déficit lacustre est une propriété des bassins à lacs | Confusion régionale : corr -0.35 globale mais -0.11 à région fixée | 08-06 |
| R8 | La régression GASP du 9 août venait d'un correctif | C'était `w_et` : le champion tournait avec la contrainte MODIS DÉSACTIVÉE, moi non | 08-10 |
| R11 (PRÉCISÉ 08-19 : son balayage tournait sur un point de reprise INEXISTANT, donc sur le socle NON ENTRAÎNÉ ; depuis les poids du CHAMPION la réponse diffère, cf. §4) | Le déficit hivernal vient d'un aquifère affamé : libérer la recharge le comblera | Libérée à la valeur mesurée (5e-5), la recharge EFFONDRE le modèle (0.4446, abandon à l'époque 4) : elle draine le sol au profit du réservoir. Le calibré (~1e-7) et le mesuré diffèrent d'un facteur 500 ; le balayage à zéro époque (5e-7 à 2e-5) montre une réponse MONOTONE DÉCROISSANTE : pas d'optimum caché, gain maximal +0.001 à 5e-7. Le déficit de février ne se comble pas par la recharge ; pistes restantes = taux de vidange k_gw, ou nappe régionale absente des deux modèles | 08-17 |
| R10 | Le taux d'apprentissage plein déstabilise les départs à chaud, la douceur les préserverait | RÉFUTÉ dans les deux sens : lr réduit 2e-4 donne 0.7111 sur le socle Linacre (contre 0.7810 au taux plein, et 0.7389 au départ — la douceur fait PIRE que ne rien apprendre) et 0.7519 sur MG24HK (contre 0.7547 au taux plein). Les grands pas ne cassent pas la solution, ils permettent d'en trouver une meilleure ; les petits pas dérivent vers une région bonne en validation et mauvaise en tenu de côté | 08-17 |
| R16 | Le deficit d'avril et l'exces de decembre viennent du partage pluie-neige (seuil du projet a -2.2168 °C, donc de la pluie en decembre) | REFUTE pour AVRIL. Le seuil gouverne l'axe decembre-MAI, pas avril : en le remontant de -2.2 a +1.0, decembre tombe de 1.207 a 0.857 et mai monte de 1.066 a 1.419, tandis qu'avril reste bloque entre 0.729 et 0.751. L'eau retiree de decembre part en MAI. Diagnostic reoriente : la CRUE PRINTANIERE ARRIVE AVEC UN MOIS DE RETARD (avril manque, mai deborde, et plus on met de neige au sol plus mai enfle) -- c'est la VITESSE DE FONTE, pas la phase. Signal secondaire a confirmer par entrainement : seuil -1.0 donne 0.7919 en inference pure contre 0.7880, avec decembre a 1.072 | 08-20 |
| R16b | La crue printaniere est en retard parce que la fonte est trop lente | REFUTE par CanSWE : la date de disparition du manteau est juste a +2, 0 et 0 jours sur 2022-2024. La fonte tombe au bon moment, elle a simplement moins d'eau a liberer. Le deficit d'avril vient donc de la MASSE du manteau (ou de sa representativite), pas du calendrier | 08-20 |
| R14 | Le plus gros ecart mensuel du champion est FEVRIER (0.688 de l'observe) : c'est le chantier hivernal | ARTEFACT DE MESURE. Le 0.688 venait de `diagnostic_ecart.py`, qui ne reproduisait pas le pilote (0.7565 au lieu de 0.7880). Dans le protocole de REFERENCE, fevrier vaut **0.896** et le plus gros ecart est **AVRIL a 0.729**, suivi de decembre en EXCES a 1.207. Le defaut reel est la CRUE PRINTANIERE et un relachement premature en debut d'hiver, pas la nappe. Le pilote imprime desormais le biais mensuel lui-meme, pour que ce chiffre ne depende plus d'un script annexe | 08-20 |
| R15 | Un modele ENTRAINE avec une nappe alimentee rattrapera ce que la greffe perd | REFUTE, et il fait meme PIRE que la greffe : krec 5e-6 donne 0.7660 apres 30 epoques contre 0.7822 greffe ; krec 2e-5 donne 0.7189 contre 0.7263. Contre 0.7880 pour la nappe affamee du champion. Fevrier gagne 0.05 a krec 2e-5 mais avril en perd 0.05 et tout l'ete se degrade : l'aquifere restituant ne paie ni en score ni en profil mensuel. Chantier aquifere CLOS sur negatif | 08-20 |
| R13 | Il existe une recharge intermediaire qui repare fevrier ET tient le score : mon banc la voyait a 2e-5 (fevrier 0.694 -> 0.835, +0.017 de KGE) | REFUTE dans le protocole de REFERENCE. Depuis les poids du champion, la reponse est MONOTONE DECROISSANTE : 5e-6 -> 0.7822, 1e-5 -> 0.7748, 2e-5 -> 0.7263, 3e-5 -> 0.6224, contre 0.7880 robinet ferme. Mon banc restait infidele au champion (0.7565 au lieu de 0.7880) et son CLASSEMENT etait faux, pas seulement son niveau. Lecon : un banc dont la ligne de controle ne reproduit pas la reference ne sert a rien, meme en relatif. RESERVE : l'ecran ne teste que la GREFFE sur des poids calibres SANS recharge, ou le sol a ete ajuste en supposant un debit de base nul ; l'entrainement avec nappe alimentee reste ouvert (en cours) | 08-19 |
| R12 | Le goulot de vitesse est le pilotage Python de la boucle de simulation, et un gain x3-4 dort dans le code | Deux tiers justes, conclusion fausse. Le pilotage domine bien (profil OUTV : colonne 98,0 % du pas, routage 1,7 %, boucle Python 0,3 % ; le routage, coupable de juin a 96 %, est innocente par le mode operateur). Mais le levier est DEJA TIRE : `compile_soil`, actif par defaut dans `etl_run.py`, vaut x17,6 (157,5 -> 9,0 min/epoque). Il ne reste pas de x3-4 : la colonne entiere compilee, une fois son blocage leve, donne x1,17 mesure, INFERIEUR a la variance x2 entre repetitions du meme reglage sous CPU charge. Non etabli, a remesurer machine au repos | 08-19 |
| R9 | La chute validation -> tenu de côté vient du CLIMAT de 2022-2024 | Le modèle ANCRÉ, incapable de sur-ajuster, fait 0.7711 en validation et **0.7748** en tenu de côté, et reste entre 0.731 et 0.803 sur les 8 fenêtres de 3 ans de 2001 à 2024. Aucune anomalie de période : c'est bien l'APPRENTISSAGE. Hypothèse posée puis réfutée par moi-même le même jour | 08-13 |

## 3. Conclusions CADUQUES — mesurées sur la base cassée, à refaire

Tout ce qui a été mesuré AVANT le 10 août l'a été sur un modèle qui recevait 0 % de forêt et 0 % d'eau libre, sans module de milieu humide, avec une ETR couvrant 80 % du territoire et une fuite de masse de 21 % aux crues. **Ces verdicts ne valent plus.**

| # | Conclusion caduque | Pourquoi | À refaire par |
|---|---|---|---|
| C1 | Carte provinciale 0.671 et règle de sélection de champion | base cassée | flotte complète sur base saine |
| C2 | Tous les verdicts sur les lacs (8 hypothèses, ancrage d'exutoire, neutralisation) | base cassée + surface de drainage au lieu d'eau libre | reprise après flotte |
| C3 | « Le multi-objectif MODIS de-collapse f_vert » (28 mai) | f_vert n'est lu par AUCUN module de la colonne actuelle | sans objet : à retirer du champ |
| C4 | Déficit de ruissellement de juin (RC 0.55 contre 0.63), déficit d'été, beta ~0.85 | fuite de masse + ETR partielle | re-mesurer sur base saine |
| C5 | Plafond de forçage à 0.76 sur CaSR | mesuré avant tous les correctifs | re-mesurer |
| C6 | Utilité de la couche d'expérience (codes latents) | compensait peut-être les entrées fausses | A/B sur base saine |
| C7 | Verdicts sur le transfert inter-régions et le zéro-shot | base cassée | après flotte |

## 4. Questions OUVERTES, avec le test qui les tranche

NOTE D'ENJEU (Essi, 2026-08-17) : la RECHARGE n'est pas un simple bouton de calage, c'est un
LIVRABLE du projet (cartes de recharge, gestion de l'eau souterraine, scénarios). Or le débit
seul la préfère quasi NULLE (balayage R11 : monotone décroissant), alors que la recharge réelle
du Québec méridional se compte en dizaines à centaines de mm/an. C'est un problème
d'identifiabilité au sens strict : Q ne contraint pas la partition drainage/recharge, il faut
une contrainte indépendante (GRACE, récessions d'étiage, cartes piézométriques) pour que la
valeur PHYSIQUE de la recharge soit crédible. Toute future calibration devra arbitrer
explicitement entre le score de débit et le réalisme de la recharge, pas l'optimiser en silence.

| # | Question | Test défini | État |
|---|---|---|---|
| O1 | Le champ spatial sert-il en NON JAUGÉ ? | validation croisée spatiale, 4 plis, OUTV | **RÉPONDU (4 plis)** : en groupant les 16 jauges (chacune retirée une fois), moyenne **0.5911** contre 0.6043 pour l'entraînement complet, soit **-0.013**. La régionalisation est donc quasi GRATUITE. Mon verdict après 2 plis (-0.07) était prématuré : à n=4 par pli, un pli a même donné les jauges retirées MEILLEURES que les vues (0.710 contre 0.563) |
| O2 | D'où vient l'excès d'été de 25-40 % ? | ETR d'Hydrotel jour par jour (réexécution instrumentée, 78 h CPU sans écriture) ou bilan ETR par classe | bloqué |
| O3 | D'où vient le déficit d'avril (0.76 sur OUTV) ? | neige EXACTE (E1), donc c'est la restitution de l'eau de fonte par le sol | non commencé |
| O4 | Pluie ou évaporation : qui a tort ? | plan 4 cases forçage (stations / CaSR brut) × contrainte ET (off / on). Stations 959 -> ETR 311 (trop bas) ; CaSR brut 1109 -> ETR 461 (crédible) | non commencé |
| O5 | L'entraînement depuis le socle à la référence améliore-t-il, tient-il, ou dégrade-t-il ? | socle mesuré à 0.7389 (contre 0.6051 pour la recette précédente) ; 30 époques en cours, époque 1 à kge_sta 0.7229 donc pas de fuite immédiate | **EN COURS** |
| O6 | La couche d'expérience apporte-t-elle encore quelque chose ? | A/B avec et sans, base saine | **RÉPONDU : NON.** avec 0.6051, sans **0.6106**. Elle ne rapporte rien et coûte un paramètre par nœud plus la non-reproductibilité. À RETIRER |
| O7 | Linacre ancrée ou module ET appris ? | A/B sur tenu de côté, base saine | non commencé |
| O8 | Combien coûte le plafond de sous-pas en entraînement ? | bloc compilé de K sous-pas appelé N/K fois (refactoring du clone) | conçu, non fait |

## 5. Dette technique qui PRODUIT des fantômes

1. ~~**Trois numérotations de tronçons**~~ **RÉGLÉ 08-12** : conversions centralisées dans `hydrotel_calib` (`id_provincial`, `id_local`, `appariement_provincial`), un appariement vide LÈVE une erreur au lieu de rendre des nombres. 2 tests.
2. ~~**Repères écrits en dur dans les sorties**~~ **RÉGLÉ 08-11** : le « Hydrotel ~0.82 » retiré, remplacé par les valeurs mesurées de l'ensemble.
3. ~~**Entrées statiques à repli silencieux**~~ **RÉGLÉ 08-12** : la colonne imprime UNE FOIS l'occupation qu'elle reçoit et AVERTIT si elle est nulle. Test dédié. Les colonnes `f_*_raw` sont par ailleurs désormais écrites à la source par `physitel_loader` (effectif à la reconstruction des caches).
4. **Journal chronologique sans statut** : ce registre est la réponse, à maintenir à chaque verdict. EN COURS, à réviser à chaque résultat de la file.
5. **13 sorties MORTES du champ spatial** sur 37 (f_root, T_snow, interception, manning_n, f_wetland, rain_hours, vsa_b, theta_fc_2/3, theta_wp_2/3) : capacité gaspillée et diagnostics brouillés. À retirer, mais casse la compatibilité des points de reprise — à faire au moment de la reconstruction des caches.
6. **Un POINT DE REPRISE ne définit pas un modèle** : occupation du sol, milieux humides, phénologie, noyau de versant et lois de lac sont posés à l'EXÉCUTION et absents du fichier. Le même checkpoint vaut 0.6051 avec eux et 0.4449 sans, sans aucune erreur. Stocker les réglages (ou leur empreinte) dans le point de reprise.
12. ~~**Le bilan d'eau ne pouvait pas être fermé**~~ **RÉGLÉ 08-20** : deux termes de la physique du milieu humide n'étaient exposés NULLE PART. `calcul_milieu_humide_isole` ne RETOURNAIT même pas son évaporation `wetev`, et le volume du réservoir `wet_vol` (m3) n'apparaissait dans aucun diagnostic ; le champ `wetland` que les diagnostics affichaient était MORT, recopié tel quel d'un pas à l'autre depuis l'ancienne colonne native. Conséquence : un audit de fermeture lisait une fuite de **1.97 % de la précipitation**, corrélée à **+0.48** avec la fraction de milieux humides et nulle là où il n'y en a pas -- signature exacte des deux termes manquants. Ce n'était donc pas nécessairement une fuite de masse, mais une IMPOSSIBILITÉ D'OBSERVER le bilan. La demande d'Essi d'un bilan hydrique rigoureux sur le Québec méridional était techniquement infaisable tant que ces termes restaient invisibles. Les deux sont désormais remontés jusqu'à `SimDiagnostics` (`etr_mh`, `wet_vol`) SANS changer la physique d'un iota. 193 tests verts. À rapprocher de la dette #42, qui portait peut-être sur cette invisibilité plutôt que sur une vraie fuite.
9. ~~**Le code ne reproduisait plus le champion**~~ **RÉGLÉ 08-19** : le correctif du 17 août libérait `krec` dès que l'aquifère est actif. Vérifié en évaluation pure À TRAVERS LE PILOTE, sur une copie du point de reprise : le champion rendait **0.4912 au lieu de 0.7880**. Toute expérience lancée avec `ETL_AQUIFER=1` depuis le 17 août partait d'une physique méconnaissable. Comportement historique redevenu le défaut (revérifié à l'identique) ; libération désormais EXPLICITE (`ETL_KREC_LIBRE=1`), à n'utiliser qu'avec une recharge choisie et gelée.
10. ~~**Un balayage qui mesure onze fois la même chose**~~ **RÉGLÉ 08-19** : le point de reprise RESTAURE `krec_raw` au chargement, donc `ETL_KREC` n'avait aucun effet en évaluation pure. Le gel vaut désormais de bout en bout. Corollaire de méthode : tout balayage doit porter une ligne de CONTRÔLE dont on connaît la réponse, sinon on ne voit pas qu'on mesure du vide.
11. ~~**Les pièces de recette recopiées d'un script à l'autre**~~ **RÉGLÉ 08-19** : `recette.py` + `courbe_retention_imposee()`. La surface de lac (HydroLAKES, +0.015 mesuré) était posée par le pilote et par AUCUN diagnostic ; la règle d'ancrage de `krec` avait divergé entre les deux. Toute pièce présente dans deux fichiers atterrit là.
8. ~~**Un mode de compilation MORT qui se taisait**~~ **REGLE 08-19** : `compile_column` compilait tout le pas de la colonne ; sa compilation echouait (`torch.as_tensor` sur une liste de tenseurs 0-d, dans `_pheno_tensors`, appele depuis le pas) et un `except` fourre-tout la faisait retomber en eager EN SILENCE, pour 1,5x plus lent que l'eager franc. Mode retire, preparation de la phenologie remontee dans `set_static`, cle filtree au rejeu des points de reprise par `purger_kwargs_obsoletes`.
7. **`pgrep` et `ps` du shell POSIX ne voient PAS les processus Windows** : cause de 4 incidents de contention en 3 jours, dont une nuit perdue et trois copies d'une même file prêtes à démarrer ensemble. Toute vérification de processus doit passer par PowerShell ou `tasklist`.
