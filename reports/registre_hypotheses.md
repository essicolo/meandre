# Registre des hypothèses — méandre

Le journal `experiment_log.md` est CHRONOLOGIQUE et purement additif : 1514 lignes, 103 entrées, où les faits établis, les hypothèses réfutées et les conclusions devenues caduques cohabitent sans distinction de statut. C'est ce qui a permis au « 0.82 d'Hydrotel » de circuler pendant des jours, et ce qui m'a fait raisonner le 10 août sur des verdicts mesurés avec un modèle qui simulait un territoire sans forêt.

Ce registre est l'inverse : un état COURANT, révisé, où chaque ligne porte un statut. Le journal raconte l'histoire, le registre dit ce qui est vrai aujourd'hui. Toute conclusion citée dans une discussion doit venir d'ici.

Statuts : **ÉTABLI** (mesuré, reproductible, toujours valide) · **RÉFUTÉ** (testé, faux) · **CADUC** (mesuré sur une base depuis corrigée, à refaire) · **OUVERT** (test défini, pas encore fait).

Dernière révision : 2026-09-15.

---

## 0. Où en est le chantier « il manque de l'eau en avril » — synthèse au 2026-08-22

Les entrées R18 à R30 ont été écrites en 36 heures et **plusieurs se corrigent les unes les autres**. Cette section dit ce qu'il faut en retenir, pour qu'on n'ait pas à reconstituer la cascade.

**Le symptôme.** Avril manque de 28 978 unités de volume, soit 5.1 % du total annuel à lui seul, et c'est la plus grosse erreur mensuelle. Novembre, décembre et janvier sont en excès de 15 915, soit 55 % du déficit d'avril. Rapports simulé/observé : avril 0.753, mai 1.138, décembre 1.235. C'est un DÉPLACEMENT d'eau dans l'année.

**Ce qui est éliminé, mesuré, et ne doit plus être rouvert.** Le domaine modélisé est bon (R18 : l'aire drainée au nœud de la jauge vaut 1.005 fois l'aire officielle sur 160 stations). Le bilan de masse ferme à 0.01 %. Les prélèvements et rejets sont ingérés et ne changent rien (R17). Le printemps n'est jamais limité par l'apport (R20 : l'écoulement d'avril-mai vaut 0.31 à 0.45 de la neige plus la pluie). L'amplitude totale du stockage est juste (R22 : 146 mm contre 143 pour GRACE). La date de fonte est juste à deux jours près (R16b).

**Ce qui reste, et c'est DEUX leviers distincts pour le même transfert, pas un.**

1. *La vitesse de sortie de la réserve lente* (R29). `krec` hérité du calage Hydrotel vaut 1.3e-7 m/h, ce qui donne 0.0036 mm/j de capacité de drainage pour la couche profonde : robinet fermé, L3 épinglée à saturation, nappe à 0 mm toute l'année, aucun endroit où faire attendre l'eau plusieurs mois. Le taux de vidange mesuré sur 1316 récessions vaut 0.0273 /j quand le modèle tourne à 0.0645. **Ce diagnostic était écrit dans un commentaire du pilote depuis le 17 août sans être relié au trou d'avril.**
2. *La rétention du manteau en début d'hiver* (R30). Le seuil pluie-neige du projet, -2.2168 °C, compte comme pluie tout ce qui tombe à -2 degrés. Le modèle a 0.55 de la neige mesurée en décembre et 0.66 en novembre. Le seuil qui produit exactement le déficit manquant est +0.3 °C, valeur que la littérature attend. Déficit de manteau et excès de débit de décembre sont le MÊME phénomène.

**Pourquoi ça n'avait jamais abouti.** Chaque tentative de donner une réserve lente au modèle a été jugée AU DÉBIT, et le débit seul préfère qu'il n'y en ait pas (R11) : le modèle atteint 0.79 sans une goutte d'eau souterraine. C'est l'énoncé exact du problème d'identifiabilité. Ce qui a changé le 21 août, c'est que la phase de GRACE est devenue lisible (R23) : elle exige +45 mm de réserve en mai quand le modèle est à -6, et elle ne dépend pas du débit.

**Trois mises en garde sur la qualité des mesures, toutes du 21-22 août.** (a) De décembre à mars, 60 à 87 % du débit observé est une reconstruction du CEHQ et non une mesure (R19) ; avril, lui, est mesuré à 90 %, donc le trou d'avril est réel. (b) Cela n'invalide PAS la comparaison avec Hydrotel (R26). (c) **Trois conclusions successives sur le manteau neigeux ont été publiées puis défaites, toutes pour la même raison : des agrégations mal appariées** (dette #17). La seule mesure valide est l'appariement site-et-jour, qui donne 0.65-0.72, et elle était imprimée dans chaque journal depuis toujours.

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

**LE COUPLAGE CONSERVATIF FERME LE BILAN SUR TOUTES LES RÉGIONS TESTÉES** (2026-08-21) :

| région | fraction de milieux humides | formulation d'Hydrotel | couplage conservatif |
|---|---|---|---|
| OUTV | 5.4 % | +1.38 % | **+0.01 %** |
| LABI | 25.4 % | +3.44 % | **+0.01 %** |
| ABIT | 35.5 % | +3.77 % | **+0.01 %** |

Mon extrapolation linéaire à partir d'OUTV (9.1 % attendus sur ABIT) SURESTIMAIT d'un facteur 2.4 : la relation est fortement SOUS-LINÉAIRE, 6.6 fois plus de milieux humides ne donnant que 2.7 fois plus de fuite. Logique, puisque la fuite dépend de la surface d'eau libre du milieu humide, qui sature, et non de sa fraction de territoire. Ampleur réelle de la requalification : de ~0.2 % (Côte-Nord) à 3.8 % (Abitibi). Substantielle, pas catastrophique.

**LE BILAN D'EAU FERME PARTOUT SAUF AU COUPLAGE DU MILIEU HUMIDE** (mesuré 2026-08-20, audit `ETL_BILAN=1`, règle posée AVANT la mesure : <0.1 % = ferme, >1 % = fuite).

| configuration | erreur de fermeture |
|---|---|
| avec milieu humide | **+1.38 %** de la précipitation (médiane par nœud +1.26 %, q90 +2.83 %) |
| **sans milieu humide** | **+0.00 %** (1 mm sur 24 481 ; q90 +0.01 %) |

Le reste de la colonne -- neige, gel, ETP/ETR, sol BV3C2, aquifère, canopée -- conserve la masse à la PRÉCISION NUMÉRIQUE. Et le réservoir de milieu humide conserve EXACTEMENT la sienne (testé sur 4 régimes : normal, débordement, étiage, vide ; écart nul). Le défaut est donc uniquement dans la SUBSTITUTION entre les deux.

MÉCANISME, lu dans le code et vérifié sur la plus petite unité : la colonne retire de la production `prod x wet_fr`, mais le réservoir ne reçoit que `wetflwi = prod x (wet_fr - wetsa/hru)` -- la production de l'EMPREINTE PROPRE du milieu humide est retranchée sans être créditée. En compensation le réservoir reçoit `wetpcp = apport x wetsa`, la précipitation directe sur cette surface, eau que le sol a DÉJÀ traitée sur tout le tronçon. La différence `(apport - prod) x wetsa/hru` est créée ou détruite selon le signe. Sur l'exemple unitaire : 1.2000 mm retirés, 1.1765 mm crédités.

PORTÉE. Le commentaire du code indique un portage ligne-à-ligne de `bv3c2.cpp` l.838-895 : c'est donc la comptabilité d'HYDROTEL, pas une erreur de notre portage. Si cela se confirme dans leur source, Hydrotel ne conserve pas la masse sur les tronçons à milieux humides, proportionnellement à leur fraction. Vérifiable, et à signaler à l'équipe qui le maintient. C'est aussi un exemple net de ce que la différentiabilité apporte : un bilan auditable terme à terme, là où un modèle compilé le rend invisible.

EFFET SUR LE SCORE : sans milieu humide, tenue de côté 0.7929 contre 0.7880. Écart de 0.005, SOUS le bruit de 0.025 : la correction ne coûtera rien, mais ce n'est pas un gain non plus.

**LE PASSAGE À L'ÉCHELLE DU QUÉBEC TOURNAIT SANS PRÉLÈVEMENTS NI REJETS** (constaté et corrigé 2026-08-21). Aucune des 15 bases régionales n'avait de table `withdrawals` ; seul le banc SLSO en avait une. La donnée existait pourtant dans le dépôt (`data/io-eau-meandre.parquet`, 1 082 964 lignes, 2001-2024 mensuel, tout le Québec méridional, avec `IDTRONCON` déjà apparié) et le chargeur aussi. Seule l'ingestion n'avait jamais été faite.

Apport net anthropique après ingestion, contre le débit moyen aux jauges :

| région | débit moyen | net anthropique | rapport |
|---|---|---|---|
| **MONT** | 11.0 m3/s | **+49.05** | **448 %** |
| SLNO | 22.6 | +15.41 | 68 % |
| SLSO | 19.3 | +6.67 | 35 % |
| OUTV | 37.4 | +7.71 | 21 % |
| GASP | 26.0 | +1.69 | 6 % |
| ABIT | 164.5 | +0.78 | 0 % |

Sur MONT le terme ignoré vaut QUATRE FOIS ET DEMIE le débit moyen d'une station. Or MONT est notre pire région (0.4869 contre 0.6631 pour Hydrotel), échec que j'attribuais à la régulation par barrages : l'explication la plus simple est qu'il manquait le terme dominant du bilan.

CONSÉQUENCES. (1) Le champion a été AJUSTÉ sans ces flux : il les a absorbés dans ses paramètres, une conductivité ou une recharge compensant une prise d'eau. C'est exactement ce que l'identifiabilité, promesse centrale du projet, doit empêcher. (2) La comparaison à Hydrotel est faussée dans les régions habitées, puisque leurs plateformes les incluent vraisemblablement. (3) Tout run postérieur au 2026-08-21 les charge automatiquement : les résultats d'avant et d'après ne sont PAS comparables.

RÉSERVE : le net est sommé sur tout le bassin alors que le débit moyen est celui d'une station, et toutes les stations ne sont pas en aval de tous les rejets. Le rapport situe un ordre de grandeur, il ne se lit pas comme une proportion exacte.

MÉTHODE : appariement EXACT par `IDTRONCON` (format provincial) converti par `id_local`, et non par rattachement géographique. Le rattachement au nœud le plus proche à 10 km écartait 879 sites sur 1 661 et en plaçait d'autres à 6.7 km. Avec l'appariement exact : ZÉRO ligne perdue sur les 15 régions.

**LES JAUGES SONT AVEUGLES A 80 % DU FLUX ANTHROPIQUE** (mesuré 2026-08-21, remontée du graphe amont). La plupart des stations ont bien des prélèvements en amont (25 sur 28 à MONT), mais l'UNION des bassins jaugés ne capte qu'une fraction du terme anthropique total :

| région | stations | avec prélèvements en amont | part captée |
|---|---|---|---|
| MONT | 28 | 25 | **19.1 %** |
| OUTV | 16 | 13 | 17.5 % |
| SLSO | 41 | 34 | 33.5 % |
| SLNO | 33 | 25 | 15.4 % |

CONSÉQUENCE POUR LE LIVRABLE MINISTÉRIEL : on ne peut PAS valider un produit de renaturalisation contre les jauges, puisque l'essentiel de l'effet à quantifier ne passe par aucune station. Cela vaut pour TOUT modèle, pas seulement le nôtre. La validité du résultat ne peut donc venir que de la STRUCTURE : que le terme anthropique soit explicitement représenté et non absorbé dans des paramètres calés. C'est la distinction entre un modèle identifiable et un modèle ajusté, et elle est démontrable chez nous par la corrélation entre paramètres appris et intensité des prélèvements par nœud -- test qu'un modèle sans paramètres continus ne peut pas passer.

### R45. Onze runs régionaux ne font pas un modèle du Québec, et dix membres ne font pas un modèle probabiliste

Deux corrections d'architecture posées par Essi le 2026-08-25, à deux niveaux de la même erreur.

La première : « il n'y a qu'une seule région dans le concept (le Québec), non ? ». La flotte livrait onze entraînements séparés, donc onze champs NeRF appris indépendamment, recollés ensuite pour couvrir la province. Rien n'y garantissait la continuité aux frontières : deux tronçons voisins de part et d'autre d'une limite de plateforme Hydrotel pouvaient recevoir des paramètres arbitrairement différents, alors que la thèse du projet est justement un champ continu qui dissout les calages régionaux. J'avais retiré la région de l'interface après le premier rappel, sans voir qu'elle structurait encore le CALCUL. Le remède existait déjà dans le dépôt : `joint.py`, un seul HydroModel, les latents dimensionnés au total des nœuds et tranchés par plateforme au chargement, un entraîneur par base lié au même optimiseur. Il ne lui manquait que la recette d'août, maintenant câblée sous `JOINT_V10`.

La seconde, immédiatement après : « et il n'y a pas de membres, seulement un modèle probabiliste ». Le plan traitait l'incertitude de forçage par répétition externe, dix runs déterministes sur dix tirages PyGMET dont on prenait ensuite l'enveloppe. Ce n'est pas un modèle probabiliste, c'est un modèle déterministe passé dix fois : l'objet livré n'est pas une distribution prédictive mais la dispersion d'un échantillon d'exécutions, et sa largeur croît mécaniquement avec le nombre de membres au lieu de converger. Un min/max de dix tirages n'est aucun quantile. L'incertitude de forçage doit entrer comme distribution d'ENTRÉE du modèle unique et ressortir par la même tête quantile, en une passe ; les membres PyGMET ne servent plus qu'à calibrer et vérifier cette entrée. Le contrat du store zarr a été corrigé en conséquence : plus de percentiles 0 et 100 fabriqués à partir de runs.

Ce que les deux partagent : dans les deux cas j'avais obtenu la bonne PROPRIÉTÉ visible (une carte provinciale, une enveloppe autour de l'hydrogramme) en assemblant après coup des objets qui ne l'avaient pas. La propriété doit tenir par construction, sinon elle ne survit pas au premier endroit où on ne l'a pas assemblée à la main.

### R46. La mise à l'échelle a révélé deux défauts que douze runs régionaux ne pouvaient pas voir

Le passage à la province a été traité comme un changement d'échelle du SCORE. C'était un changement d'échelle du COÛT et de la CONFIGURATION, et les deux ont mordu.

Le coût, d'abord, mesuré et non estimé. `joint.py` gardait la plateforme comme unité de calcul et faisait tourner les quatorze en rotation, une simulation par plateforme et par epoch. Résultat sur le run du 2026-08-25 : 5 h 36 de temps écoulé, 19 403 s de CPU sur un seul cœur, GPU à 0-1 %, epoch 0 inachevé. Le modèle est limité par le LANCEMENT de noyaux, pas par le calcul, donc le coût suit le NOMBRE de boucles et non leur largeur. Les réseaux étant disjoints, leur union est un graphe bloc-diagonal et l'epoch devient une seule boucle sur 25 656 tronçons vectorisés. Deux obstacles réels ont dû tomber au passage : le forçage provincial pèse 5.6 Go, les deux tiers de la carte, et ne laissait plus de place aux activations d'un chunk couvrant tous les tronçons (il reste en RAM hôte, seule la tranche du chunk monte au GPU) ; et GRACE, qui moyennait le stockage sur tous les nœuds, aurait confronté quatorze bassins mêlés à une seule série satellitaire (agrégation par bassin, y compris pour le biais climatologique, désormais accumulé par mois ET par bassin).

La configuration, ensuite, et c'est le plus coûteux. L'occupation du sol, les milieux humides, la fonte calée et la phénologie ne sont PAS dans les caches DuckDB : ils viennent des paquets de plateforme Hydrotel et n'étaient chargés que par `etl_run.py`. `joint.py` ne les posait pas. Tous les entraînements conjoints, ceux du 2026-08-25 comme les pilotes antérieurs, tournaient donc sur un territoire sans forêt, sans eau libre et sans milieu humide, tout en classe découvert, la plus fondante. L'écart mesuré sur OUTV à intrants identiques et paramètres figés est de 0.482 contre 0.749 de KGE aux jauges, soit plus que la somme de tous les gains de la campagne d'août. Toute comparaison conjoint contre régional faite avant ce correctif comparait deux physiques différentes, pas deux architectures.

Un troisième écart, mineur mais de la même famille, est apparu : douze caches portent quatre champs physiques, ABIT et LABI en portent treize (milieux humides, c_prod, ksat_bs). L'intersection est le seul choix sûr dans un domaine fondu, mais elle retire à ces deux plateformes des ancrages qu'elles avaient en régional, donc leurs scores fusionnés ne sont pas directement comparables à gen1 tant que les caches n'auront pas été rebâtis sur un schéma commun.

LA LEÇON, et c'est elle qui compte pour la suite : les trois défauts sont des artefacts de PRODUCTION, pas de modélisation, et ils avaient tous la même forme, une configuration héritée en silence par chemin d'exécution. Douze runs régionaux ne pouvaient pas les révéler puisque chacun héritait de la sienne. Fondre les caches en un domaine unique a été un révélateur autant qu'une optimisation, ce qui est l'argument le plus fort en faveur de la consigne d'Essi sur les régions : ce qui n'est jamais unifié n'est jamais confronté.

### R47. Un garde-fou insatisfaisable devient l'objectif, et le borner ne suffit pas

Mesuré le 2026-08-26, deux fois, sur deux échelles.

Sur la province à huit epochs, la perte d'entraînement passe de 4.4 à 48.9 pendant que le débit se dégrade : val_kge 0.7276 puis 0.6510, médiane par station 0.3662 puis 0.2756, et la tenue de côté finale tombe à 0.4518 contre 0.6193 pour le run de quatre epochs. Le modèle n'optimise plus le débit mais GRACE, dont les termes pesaient déjà 966 % du total. Le repli de divergence a mordu deux fois sans suffire. ALLONGER L'ENTRAÎNEMENT DÉGRADE LE MODÈLE, ce qui invalide au passage toute comparaison d'architecture au-delà de quelques epochs.

Deux causes ont été corrigées, et aucune n'a suffi. La tolérance du terme climatologique valait 5.4 mm, soit 25 sur racine de 21, l'incertitude d'une climatologie sur vingt et un ans ; en accumulant le biais par mois ET par bassin, chaque case ne voyait plus qu'une fraction de ces échantillons, et juger une moyenne bruitée à la tolérance d'une moyenne longue fabrique un écart énorme. Et rien ne bornait le z-score : un écart de 240 mm contre 5.4 de tolérance donne un terme de plusieurs milliers ET un gradient proportionnel, donc une rétroaction positive.

LE TEST APPARIÉ DE LA BORNE EST NÉGATIF. Six epochs sur GASP et MONT, borne de Huber active à trois écarts-types : les termes GRACE croissent quand même, tws de 13 à 49.9 et tws_clim de 2.64 à 46.5, et le modèle se dégrade monotonement de 0.5204 à 0.2447. Une borne empêche le gradient d'EXPLOSER, pas de POINTER TOUJOURS DANS LE MÊME SENS : si le modèle ne peut structurellement pas satisfaire la contrainte, un gradient borné mais constant pousse indéfiniment. Le correctif était nécessaire, il n'est pas suffisant.

LA QUESTION QU'IL FALLAIT POSER D'ABORD, et qu'aucun réglage de poids ne remplace : l'écart est-il d'AMPLITUDE, de PHASE ou de DÉRIVE ? Les trois appellent des remèdes opposés. Une amplitude simulée trop grande veut dire que la colonne respire plus que ce que le satellite voit, et aucun poids ne réconciliera sans aplatir la dynamique qui fait le débit. Une phase décalée se corrige par la recharge et la vidange. Une dérive est un problème de référence, pas de physique. D'où `diag_stockage.py`, qui simule sans entraîner et agrège le stockage EXACTEMENT comme le fait la perte.

REMÈDE PRÉPARÉ MAIS NON POSÉ : GRACE en FORME plutôt qu'en niveau, chaque côté divisé par son propre écart-type. C'est la discipline que le projet applique déjà à MODIS ET (R24, la donnée sert pour la forme, jamais pour le niveau) et qui n'avait jamais été étendue à GRACE. Laissé inactif par défaut tant que le diagnostic n'a pas parlé, parce qu'activer un remède avant d'avoir le diagnostic est exactement ce qu'on reproche à un calage.

### R49. Deux sorties pour un seul degré de liberté : le champ exploite la direction sans gradient

Le 2026-08-27, le gel a reçu des propriétés thermiques par nœud apprises par le champ, en remplacement de trois scalaires globaux identiques sur les 25 656 tronçons. Le résultat sur le score est franc, sur GASP et MONT, six epochs, tout identique par ailleurs : le témoin s'effondre (0.5204, 0.4531, 0.3353) alors que la version apprise monte (0.5470, 0.5666, 0.5945). C'est le premier entraînement depuis deux jours qui s'améliore d'un epoch à l'autre, et il lève le blocage décrit en R47, où la contrainte GRACE prenait le contrôle faute pour le modèle de pouvoir la satisfaire.

MAIS L'AUDIT DU CHAMP, fait avant de lire le score, a trouvé conductivité et capacité ANTI-CORRÉLÉES à -0.920 d'un tronçon à l'autre. C'est thermodynamiquement impossible : les deux croissent avec la teneur en eau, l'eau étant à la fois bien plus conductrice et bien plus capacitive que l'air qu'elle remplace.

CE CONTRÔLE NE DÉPEND D'AUCUNE CARTE, ce qui importe puisque les textures viennent de PHYSITEL sans documentation et que les pédotransferts sont très approximatifs (O15). L'hypothèse symétrique, le modèle apprenant la BONNE physique sur de FAUSSES étiquettes, auquel cas l'inversion aurait diagnostiqué la donnée et non le modèle, est donc écartée sans avoir à trancher sur la qualité des cartes. C'est le genre de test à chercher en priorité : celui qui ne dépend d'aucune des entrées suspectes.

LA CAUSE EST UNE PARAMÉTRISATION MAL POSÉE, PAS UNE TRICHE. La relaxation de Rankinen ne dépend du sol que par le RAPPORT conductivité sur capacité : dt·kt/(ca·(2z)²) = dt·alpha/(2z)². Exposer les deux grandeurs laissait une direction SANS AUCUN GRADIENT, augmenter les deux ensemble ne changeant rien au gel, où le champ pouvait dériver au gré du bruit d'entraînement. C'est une définition assez exacte du surajustement : de la capacité qui ne sert pas la prédiction mais qui bouge quand même. Une seule sortie désormais, `diff_gel`, initialisée à 1.6e-7 m²/s, la valeur du C++.

CE QUI RESTE ACQUIS, et qui est le vrai résultat : le déplacement MOYEN était physiquement juste. Conductivité -20 %, capacité +20 %, donc diffusivité -34 %, c'est-à-dire un front de gel qui remonte plus lentement au printemps et retient le drainage plus longtemps, exactement ce que GRACE réclamait après l'échec des quatre leviers de réservoir lent (R48). Le mécanisme est bon, c'était sa paramétrisation qui ne l'était pas.

HYPOTHÈSE D'ESSI À TESTER : la contrainte devrait améliorer la généralisation et réduire le surajustement. Critère de lecture retenu, l'écart entre validation et tenue de côté plutôt que le niveau atteint.

### R50. Cinq hypothèses tombent en cascade sur le déficit de manteau, aucune ne l'explique

Chaîne du 2026-08-27, chaque étape testée avant de passer à la suivante. Point de départ : `calendrier()`, ajouté au banc neige-seule, montre pour la première fois les DEUX courbes mensuelles côte à côte plutôt qu'un ratio. Sur SAGU, le manteau simulé est déjà à 78 % du mesuré en DÉCEMBRE (68 mm contre 84), avant toute fonte, puis perd un quart de sa masse en avril quand le manteau réel n'en perd presque rien (185→141 contre 241→238). Deux défauts distincts, pas un : un déficit d'ACCUMULATION qui court tout l'hiver sur certains bassins, et une perte d'AVRIL qui s'y ajoute.

RÉFUTÉ : la rétention du sol. Nappe 3x et 10x plus lente, drainage L3 non linéaire, leur combinaison : quatre leviers, quatre échecs contre la climatologie GRACE (R48).

RÉFUTÉ : la modulation saisonnière de la fonte (amplitude 0.5, validée sur le débit en R36). Balayée de -0.25 à +0.50 contre la MASSE CanSWE au lieu du débit : le rapport d'avril reste bloqué à 0.60-0.61 quel que soit le signe. Elle ne déplace pas le défaut, elle n'a jamais été le bon levier pour ce juge.

REDIRIGÉ PUIS RÉFUTÉ : le gel thermique appris avait semblé lever le blocage GRACE (R49) avant que l'audit ne découvre une paramétrisation redondante (conductivité/capacité anti-corrélées à -0.920). Corrigé à une diffusivité unique, le test apparié retombe sur la trajectoire du témoin : le gain venait de la direction sans gradient, pas d'un mécanisme physique.

INNOCENTÉ (à la marge) : la précipitation totale de CaSR. Comparée à 21 stations ECCC en hiver (DJF, 2010-2020, <25 km d'un nœud) : rapport médian CaSR/ECCC = 0.948. Un déficit de 5 % ne peut pas expliquer un déficit de manteau de 22 %, et la réserve connue (les jauges au sol sous-captent elles-mêmes la neige sous le vent) va dans le sens de renforcer cette innocence, pas de l'affaiblir.

RÉFUTÉ, ET À L'ENCONTRE DE L'HYPOTHÈSE : le seuil du bulbe humide. Si le partage envoyait trop d'eau vers la pluie, l'abaisser (plus de précipitation classée neige) devrait ENGRAISSER le manteau. C'est l'inverse qui se produit : -0.8 (actuel) donne 0.73-0.77 sur décembre-février, -2.6 donne seulement 0.66-0.67. Le seuil actuel est déjà le MEILLEUR des quatre testés pour la masse, pas le pire.

CE QUI RESTE, et qui commence à ressembler à un défaut structurel plutôt qu'à un paramètre mal réglé : ni la précipitation totale, ni le partage pluie-neige, ni la modulation de fonte, ni la rétention du sol n'expliquent le déficit de décembre. Piste non explorée ce soir, à regarder en premier : l'ETI (Pellicciotti), déjà réfuté sur le DÉBIT le 24 août (R44), ressort à 0.81-0.89 sur la MASSE dans ce même balayage, contre 0.73-0.77 pour le degré-jour au bulbe humide -- un juge différent pourrait avoir tranché la mauvaise formulation de fonte pour la mauvaise raison.

### R51. L'ETI, jugé sur la MASSE et non le débit, décale le calendrier de fonte de quatre à six semaines

MÉCANISME CORRIGÉ LE 2026-08-28 (voir R53) : le texte ci-dessous attribuait le gain à une « fonte hivernale fantôme » que le degré-jour produirait en décembre. C'EST FAUX, mesuré par décomposition : l'écart de fonte entre les deux formulations est de 0 à 3 mm/mois en décembre-janvier. Le vrai effet est au PRINTEMPS. Le FAIT (l'ETI améliore le calendrier de manteau) tient ; l'explication a été refaite.

Suite directe de R50. Cinq hypothèses réfutées sur le déficit de manteau ; la seule piste laissée en réserve était l'ETI (Pellicciotti), rejeté le 24 août (R44) mais UNIQUEMENT sur le débit. Testé cette nuit sur la MASSE CanSWE, calendrier complet, coefficients de LITTÉRATURE (tf=1.2e-3, srf=9.4e-6), sans aucun calage.

SAGU : nette amélioration sur TOUTE la saison, pas seulement le printemps. Rapport mensuel décembre à mai : 0.86, 0.86, 0.83, 0.83, 0.89, 1.28, contre 0.77, 0.76, 0.73, 0.60 pour le degré-jour au bulbe humide (avril seul passe de 0.60 à 0.89). Le mécanisme est cohérent : en décembre les jours sont courts et le rayonnement faible, donc l'ETI y produit PEU de fonte quelle que soit la température, alors que le degré-jour pur fond dès qu'un redoux passe, sans regarder si l'énergie disponible le justifie. C'est probablement le vrai mécanisme du déficit de décembre : pas un manque d'accumulation, une fonte hivernale fantôme que rien ne limitait.

OUTV : amélioration nette du printemps (avril 0.61 → 1.25, mai 0.11 → 2.44) mais avec un SURCROIT net à partir de février (rapports 1.00 à 1.25), là où le degré-jour était en léger déficit. Les coefficients de littérature ne sont donc pas directement transférables d'un bassin à l'autre : SAGU les veut presque tels quels, OUTV demande moins de fonte radiative que la littérature n'en donne.

GASP : ETI aggrave massivement (rapports 1.43 à 2.40, jusqu'à NaN en octobre-décembre par absence de données). Cohérent avec le diagnostic déjà établi (R44) : le forçage côtier de GASP est en cause, pas la formulation de fonte, et ETI ne peut pas corriger un défaut de rayonnement ou de température en amont de lui.

MONT : aucune donnée CanSWE exploitable par ce banc (déjà noté, un seul site retenu par les filtres de représentativité).

CE QUI CHANGE : R44 avait conclu que l'ETI ne valait pas la peine pour la 1.0, EN JUGEANT SUR LE DÉBIT SEUL. Le débit ne pouvait pas voir un défaut de calendrier de deux à trois mois qui se compense en partie dans le bilan annuel. Le bon juge pour une question de CALENDRIER est un calendrier, pas un débit intégré. La conclusion de R44 doit donc être révisée : l'ETI n'est pas un raffinement optionnel, c'est probablement une pièce manquante du DIAGNOSTIC de décembre, sur les bassins où le forçage radiatif est fiable.

PROCHAINE ÉTAPE : verifier que l'ETI corrige AUSSI le decalage de stockage GRACE (R47-R49) en inference, avant de payer le moindre entrainement -- si la fonte trop precoce est la vraie cause commune, la meme correction devrait ameliorer les deux juges a la fois.

### R52. Le terme turbulent est réfuté sur la MASSE aussi, et il n'est pas sélectif

Test du 2026-08-27, suite de R51. Prédiction posée AVANT de mesurer : un terme physique réel doit corriger OUTV et GASP, où l'ETI seul laisse trop de neige, SANS dégrader SAGU, où l'ETI seul est déjà bon. Un paramètre de calage déguisé, lui, agirait partout de la même façon.

RÉSULTAT : le terme agit partout, uniformément, et dégrade SAGU de façon monotone. Rapports d'avril à SAGU : 0.89 sans le terme, puis 0.72, 0.56 et 0.40 pour tf_wind de 2e-4, 5e-4 et 1e-3. Sur OUTV il aide le printemps (avril 1.25 → 0.84) mais creuse l'automne (novembre 0.91 → 0.79). Sur GASP il ne suffit jamais, l'excès restant de 1.37 à 2.19 même au coefficient le plus fort.

POURQUOI IL NE PEUT PAS ÊTRE SÉLECTIF : la vitesse de vent hivernale varie peu entre les régions (SAGU 4.57, OUTV 3.98, GASP 4.56 m/s) et peu dans la saison. Le produit tf_wind × u2 est donc quasiment une CONSTANTE ajoutée à tf : mathématiquement, ce terme augmente le facteur de fonte global, il ne distingue rien. Une vraie fonte advective demande le gradient de pression de vapeur et la pluie-sur-neige, pas la seule vitesse du vent. Le terme est donc réfuté sur la masse comme il l'avait été sur le débit, mais cette fois avec sa raison.

DIAGNOSTIC SÉPARÉ POUR GASP, et il confirme R44. L'ETI y donne 162 à 301 mm quand CanSWE en mesure 104 à 159 : un EXCÈS de 50 à 100 %. Le degré-jour y était déjà en excès (1.91/1.79/2.02, R44). Les deux formulations de fonte sur-accumulent donc, ce qui n'est pas un problème de fonte mais d'ENTRÉE : le forçage côtier met trop de précipitation solide sur ce territoire. Aucun réglage de fonte ne corrigera ça, et il faut cesser d'en chercher un.

CE QUI SORT DE LA SÉQUENCE R50-R52 : l'ETI aux coefficients de LITTÉRATURE, sans calage et sans terme ajouté, est la meilleure formulation de fonte disponible sur les bassins au forçage fiable (SAGU 0.81-0.89 sur toute la saison contre 0.60-0.77 pour le degré-jour). OUTV le sur-corrige légèrement en fin d'hiver, ce qui reste à comprendre mais ne le disqualifie pas. GASP est un dossier de forçage, pas de fonte.

### R53. Ce n'est pas une fonte fantôme d'hiver, c'est une fonte trop précoce de printemps

Correction demandée par Essi le 2026-08-28, après qu'il a relevé deux narrations avancées sans vérification. La première : j'avais décrit OUTV et GASP comme « plus venteux » alors que le tableau que je venais d'afficher donnait OUTV à 3.98 m/s contre SAGU à 4.57 -- OUTV est le MOINS venté des trois. J'ai bâti l'hypothèse advective sur un contraste inexistant, lancé un test dessus, puis expliqué son échec par l'uniformité du vent, qui était lisible AVANT de lancer. La seconde : j'expliquais le gain de l'ETI par une fonte de redoux non freinée, ce qui prédit un déficit maximal là où les redoux sont les plus fréquents -- c'est OUTV (35.7 % de jours à Tmax>0), et c'est justement là que le degré-jour accumule le mieux. Le mécanisme prédisait l'inverse de l'observation.

POURQUOI J'AI PU ME TROMPER DEUX FOIS SANS ÊTRE CONTREDIT : je regardais des RATIOS DE STOCK mensuels, qui ne montrent que la différence cumulée entre accumulation et fonte. Ils ne portaient pas l'information permettant de départager les deux, donc aucune narration ne pouvait être réfutée par eux.

LA DÉCOMPOSITION (`decompo_fonte.py`, somme des hausses et des baisses journalières de SWE, par mois, par formulation) tranche sans interprétation.

L'ACCUMULATION est identique entre les deux formulations : écarts de 0 à 7 mm/mois. La formulation de fonte n'a donc AUCUN effet sur l'accumulation, ce qui était attendu mais jamais vérifié.

LA FONTE se décale, elle ne diminue pas. Écart ETI moins degré-jour, en mm/mois : SAGU -1 en décembre, 0 en janvier, -2 en février, puis -25 en mars, -70 en avril, +92 en mai. OUTV : -1, -3, -10, puis -23 en mars, +27 en avril, +20 en mai. L'ETI ne fond pas moins, il fond QUATRE À SIX SEMAINES PLUS TARD.

LE MÉCANISME, cette fois cohérent avec les chiffres : au printemps, la température repasse au-dessus du seuil de fonte bien avant que le rayonnement n'ait fourni l'énergie nécessaire. En mars le soleil est encore bas mais l'air se réchauffe déjà. Le degré-jour, qui ne voit que la température, fond dès ce moment ; l'ETI attend l'énergie. C'est ce décalage que CanSWE réclamait et que GRACE voit sur le stockage.

CE QUI RESTE ENTIER, ET QUI CHANGE DE NATURE : le déficit de manteau de SAGU en DÉCEMBRE (0.77 du mesuré) ne s'explique ni par l'accumulation, identique entre formulations, ni par la fonte, quasi nulle à ce moment. Il n'est donc pas un problème de fonte du tout. Avec la précipitation totale innocentée contre ECCC (0.948) et le seuil de partage réfuté (R50), il reste sans explication et doit être traité comme une question distincte de celle du calendrier printanier.

### R54. La fonte mesurée directement : le degré-jour n'explique RIEN, et personne ne prédit l'intensité

Renversement de méthode proposé par Essi le 2026-08-28 (« la fonte devrait néanmoins être captée par la météo et le terrain [...] MLP ou même un ridge ? »). Depuis trois jours on devinait une formulation, on la faisait tourner dans le modèle entier, et on la jugeait sur un stock résiduel qui ne peut réfuter aucune hypothèse sur l'accumulation ou la fonte prises séparément (R53). Or la fonte est OBSERVABLE : CanSWE donne la masse au sol à intervalle médian d'UN jour (96 % des intervalles à 3 jours ou moins sur SAGU), donc une baisse de masse un jour sans précipitation EST une ablation mesurée. 20 998 jours-site utilisables sur SAGU.

Validation croisée PAR SITE (un découpage aléatoire mettrait des jours consécutifs quasi identiques de part et d'autre et donnerait un R² fantôme).

CIBLE COMPLÈTE : degré-jour -0.008, ETI 0.085, ETI+vent+vapeur 0.091, ridge toutes covariables 0.146, forêt 0.510.

Le degré-jour a un pouvoir explicatif NUL sur la fonte quotidienne observée -- moins que la moyenne. Trois jours de réglages portaient sur une formulation qui n'explique rien. L'ETI fait mieux mais reste à 0.085, et le vent plus la vapeur n'ajoutent que 0.006, ce qui réfute le terme turbulent (R52) par une voie indépendante.

DÉCOMPOSITION EN DEUX QUESTIONS (76 % des jours ont une fonte nulle ; un R² unique mélange deux problèmes de nature différente).

  DÉCLENCHEMENT, fond-il ? AUC : degré-jour 0.669, ETI 0.747, toutes covariables 0.856.
  INTENSITÉ, combien les jours de fonte ? R² : degré-jour -0.078, ETI 0.032, ridge 0.136, forêt 0.234.

CORRECTION D'UNE LECTURE TROP RAPIDE : j'avais annoncé le 0.510 de la forêt comme preuve que « l'information est dans la météo et le terrain ». La décomposition montre que ce score était largement porté par la partie facile, prédire les 76 % de jours SANS fonte. Sur les jours de fonte réels, la forêt tombe à 0.234. Personne ne prédit bien l'INTENSITÉ.

CE QUE LES DONNÉES DÉSIGNENT : les covariables les plus utiles à la forêt ne sont dans AUCUNE des deux formulations -- le jour de l'année (0.295), la température moyenne (0.197), et le STOCK de neige lui-même (0.182), loin devant le rayonnement absorbé (0.041). Le stock compte parce qu'un manteau profond est froid et doit se réchauffer avant de libérer de l'eau : c'est le contenu en froid, processus standard en nivologie. Le clone suit bien une variable `chaleur` dans son état, mais le TAUX de fonte n'en dépend pas.

RÉSERVE LEVÉE PAR LA MESURE, VOIR R55 : le plafond de bruit de la cible vaut 0.35 de R². Le 0.234 de la forêt en occupe donc les deux tiers, et la marge réelle pour une physique manquante est de 0.12, pas de 0.77.

### R55. Le plafond n'est pas surtout de la physique manquante : la cible elle-même plafonne à 0.35

Réserve que j'avais posée avant de proposer quoi que ce soit après R54, maintenant chiffrée. Deux sites voisins sous la même maille météo sont deux réalisations du même signal : leur accord borne ce qu'un modèle piloté par ce forçage peut atteindre. Mesure sur 6 plateformes, 126 sites, 64 556 intervalles, ablation normalisée en mm/j et intervalles appariés par recouvrement.

  corrélation entre sites voisins (moins de 40 km) : médiane 0.590, q25 0.518, q75 0.627
  R² implicite : médiane 0.348, q75 0.393, et 0.44 pour la seule paire à moins de 10 km

Seules 3 paires ont assez d'intervalles communs, toutes des stations automatiques (1393 intervalles médians) ; les carottages manuels sont trop épars pour s'apparier. Le chiffre est donc solide en profondeur temporelle et mince en échantillon spatial, à revoir si le réseau automatique s'étoffe.

LECTURE. La forêt de R54 atteint 0.234 sur l'intensité pour un plafond de 0.348. Elle capte les deux tiers de ce qui est captable. Le tiers restant vaut 0.11 de R², ce qui est la taille réelle du gisement de physique manquante sur cette cible, pas les 0.77 que le résidu brut laissait croire. Conséquence de méthode : chercher une meilleure formulation de fonte contre CanSWE ponctuel est un jeu à faible enjeu, et un gain de 0.05 de R² sur cette cible ne se distinguera pas du bruit de représentativité.

CE QUE DIT LA LITTÉRATURE, ET ELLE DIT LA MÊME CHOSE (Université Laval, forêt Montmorency, cherché le 08-28 sur demande d'Essi). Nadeau et coll. 2020 (Water 12:2284) partent du site le mieux instrumenté du domaine, avec précipitation solide corrigée du biais, températures distribuées, EEN, hauteur, température de neige et flux turbulents et radiatifs mesurés, puis testent une à une les variables qu'on ajouterait à un degré-jour : rayonnement sous-canopée, interception, température de surface de neige, sublimation, contenu de froid. Conclusion : à l'exception de la sublimation (Pbias 5.4 %) et de la température de surface de neige, aucune n'améliore le modèle. Le meilleur des trois algorithmes testés reste à 24 % de Pbias et 100 mm d'EEN de RMSE. C'est notre R53 et notre R54 par une autre porte : le rayonnement n'a pas de valeur ajoutée dans une forêt boréale humide, et davantage de données n'y change rien.

Les autres travaux du même site convergent. Sur l'ablation, Pomeroy et l'école Laval attribuent le contrôle de l'accumulation à l'interception par la canopée et le contrôle de l'ablation à la distribution du rayonnement de courtes longueurs d'onde, ce qui semble contredire Nadeau jusqu'à ce qu'on note l'échelle : c'est vrai pour le PATRON spatial entre trouées et sous-couvert, pas pour le TAUX moyen sur une maille de plusieurs km, seule échelle qui nous concerne. Sur le contenu de froid (Cryosphere 15:5371, BEREV), l'état interne qui retarde le déclenchement de la fonte est mesuré sur 53 fosses et quatre structures de canopée. Notre clone d'Hydrotel PORTE déjà ce terme (`chaleur` dans hydrotel_clone/snow.py, avec perte convective, chaleur latente, pluie sur neige et géothermie) : ce n'est pas un degré-jour, et l'un des deux processus que Nadeau retient est donc déjà dedans. L'autre, la sublimation, est implémenté en opt-in mais RÉFUTÉ en l'état par R36 (Kuzmin suppose un vent de plaine dégagée, le manteau s'effondre sous 74 % de forêt) : c'est là qu'un facteur d'abri par canopée aurait une justification externe, et c'est la seule piste de fonte que la littérature du domaine soutient encore.

Sur l'hiver chaud et peu neigeux (HESS 28:2745, 2024, même forêt) : la fonte y débute 23 jours plus tôt et dure 24 à 31 jours de plus. La sensibilité de la DATE au climat est bien plus forte que celle du taux, ce qui recoupe R53 (l'ETI ne fond pas moins, il fond 4 à 6 semaines plus tard) et désigne le déclenchement, pas l'intensité, comme la variable qui porte le signal utile au débit.

### R56. Comment Hydrotel coordonne la fonte : un verrou par classe d'occupation, calibré contre le débit

Question d'Essi le 2026-08-28, après R55 : comment Hydrotel y arrive-t-il si ni meandre ni Laval ne savent capter la fonte ? Réponse trouvée dans le calage lui-même, `simulation/simulation/degre_jour_modifie.csv` des plateformes, jamais lu jusqu'ici pour ce qu'il dit.

Hydrotel porte un SEUIL DE FONTE PAR CLASSE D'OCCUPATION, distinct du seuil pluie-neige. Deux jeux seulement pour tout le Québec, uniformes sur tous les UHRH d'une plateforme et identiques d'un membre de l'ensemble à l'autre :

  sagu, outv, abit : conifère +3.35, feuillu +0.40, découvert -2.55 degrés | taux 7.20 / 8.52 / 10.10
  gasp, mont, slso : conifère +2.26, feuillu +1.92, découvert +1.57       | taux 4.52 / 9.05 / 18.09

MESURE DE CE QUE LE VERROU FAIT, sur le forçage réel (part des jours où l'air moyen du domaine dépasse le seuil) :

  sagu, conifère  : 0.7 % de décembre à mars, 26.3 % en avril, 85.0 % en mai
  sagu, seuil 0   : 4.1 %                      56.7 %          97.2 %
  abit, conifère  : 1.7 %                      29.3 %          85.8 %
  outv, conifère  : 4.0 %                      54.1 %          97.0 %
  gasp, conifère  : 3.0 %                      45.1 %          97.9 %

Sous conifère, le calage divise par SIX le nombre de jours de fonte possibles en hiver et par DEUX ceux d'avril, par rapport au seuil physique. C'est là toute la coordination : ce n'est pas une meilleure loi de fonte, c'est un verrou qui interdit la fonte hivernale sous couvert résineux et retarde le déclenchement du printemps, ajusté classe par classe contre l'hydrogramme.

CE QUE LE VERROU N'EST PAS. Les taux nominaux 7 à 18 mm/j/degC ont l'air hors littérature, ils ne le sont pas : le degré-jour d'Hydrotel les multiplie par `indice_rad * (1 - albédo)`, et l'albédo décroît vers 0.5, donc le taux effectif tombe vers 2 à 3 mm/j/degC, dans la plage de Hock 2003. Le levier est le SEUIL, pas le taux. Vérifié avant d'écrire, contre ma première lecture.

CE QUE LE VERROU EST VRAIMENT, ET POURQUOI LAVAL NE LE CONTREDIT PAS. Un manteau sous canopée résineuse à +2 degrés d'air ne fond effectivement pas : il lui manque l'énergie pour combler son contenu de froid, et la canopée lui coupe le rayonnement. C'est exactement l'état interne que le Cryosphere 15:5371 mesure au BEREV sur 53 fosses et quatre structures de canopée, et exactement le processus que Nadeau et coll. 2020 retiennent (température de surface de neige) quand ils rejettent le rayonnement sous-canopée. Hydrotel encode donc un processus RÉEL, sous forme d'une constante calibrée là où Laval a une variable d'état. Il n'a pas plus de savoir, il a un bouton bien réglé au bon endroit. Et il est réglé contre le débit, pas contre la neige : rien ne garantit qu'il tienne hors de sa période de calage, ce qui est précisément le claim d'identifiabilité de meandre.

CONSÉQUENCE POUR NOUS, ET C'EST UNE DETTE QU'ON S'EST CRÉÉE. Le verrou est calibré EN PAIRE avec le seuil pluie-neige de -2.2168 (R30) : Hydrotel fabrique 35 % de neige en moins puis la verrouille très tard. R35 et R37 ont remplacé la moitié pluie-neige de la paire par le bulbe humide dérivé d'une mesure de masse, ce qui était juste en soi et a réparé l'accumulation, mais a laissé l'autre moitié intacte. On a donc 35 % de neige de plus derrière le MÊME verrou, et mai déborde à 1.235 (R37) : c'est la signature attendue, pas un mystère. La bonne suite n'est pas une énième formulation de fonte, c'est de dériver le verrou de la même façon qu'on a dérivé le seuil pluie-neige, ou mieux, de le remplacer par le contenu de froid que le clone porte DÉJÀ (`chaleur` dans hydrotel_clone/snow.py, avec perte convective, chaleur latente, pluie sur neige et géothermie).

ET ÇA EXPLIQUE R53 SANS RIEN AJOUTER. L'ETI garde le seuil ancré mais remplace le taux calibré par un tf de littérature six à huit fois plus faible, sans le facteur `(1 - albédo)`. Le verrou reste, la libération explosive disparaît. L'ETI ne fond donc pas moins, il étale : quatre à six semaines plus tard, ce qui est le chiffre mesuré. Trois jours passés à moduler l'ETI portaient sur la moitié de la paire qui n'était pas le problème.

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
| R17 | Le mauvais score de MONT (0.4869 contre 0.6631 pour Hydrotel) vient de l'absence des prelevements, qui y valent 448 % du debit moyen d'une station | REFUTE. Test apparie dans le code du jour, MONT a zero epoque : AVEC prelevements 0.4987 median, SANS 0.4986. Un dix-milliemme d'ecart. Deux raisons : (1) le biais de volume vaut deja 1.107, le modele produit 11 % d'eau EN TROP, donc ajouter un apport net ne pouvait qu'empirer -- la moyenne baisse d'ailleurs de 0.5139 a 0.4930 ; (2) mon rapport de 448 % comparait un terme somme sur 179 noeuds au debit moyen de 23 stations, deux quantites qui ne se rencontrent pas. Essi l'avait annonce : globalement mineurs, localement forts | 08-21 |
| R18 | Le debit observe ne correspond pas au domaine modelise : les jauges seraient rattachees a des noeuds qui drainent moins de territoire que la station n'en declare | REFUTE. Sur 160 stations et six regions, l'aire cumulee au noeud de la jauge vaut **1.005 fois** l'aire officielle en mediane (q10 par region entre 0.92 et 1.00). Cinq stations sous 0.9, dix-huit au-dessus de 1.1, aucun biais systematique par region (outv 1.008, gasp 1.014, sagu 1.008, slno 1.002, mont 1.007, slso 0.993). Les jauges sont au bon endroit et le domaine correspond aux bassins reels. AVERTISSEMENT DE METHODE : ma premiere mesure donnait 0.248 et concluait au desastre ; elle lisait `drainage_area_km2` du parquet territorial, qui n'est NI l'aire locale NI l'aire cumulee (elle somme a 14 926 km2 sur outv quand le domaine en fait 83 202). La bonne colonne est `area_km2_physical`, qui donne l'aire CUMULEE directement, sans parcours de graphe. Cinquieme colonne mal nommee de la serie -- voir dette #13 | 08-21 |
| R19 | Le debit OBSERVE d'hiver n'est pas une mesure : sous couvert de glace la relation hauteur-debit ne tient plus et l'organisme ESTIME les valeurs par interpolation entre jaugeages manuels | **ETABLI** sur 7 regions et 25 ans, sans aucune simulation. Part des jours situes sur un segment PARFAITEMENT rectiligne (derivee seconde nulle a 1e-3 pres de l'echelle locale) : fevrier **21.7 %**, janvier 14.7, mars 12.8, decembre 6.6, puis avril 1.5 et **0.3 a 0.6 % de mai a novembre**. Un debit reel ne suit jamais une droite exacte plusieurs jours de suite ; une courbe tracee a la main entre deux jaugeages, si. C'etait un PLANCHER, et **la mesure DIRECTE, obtenue le meme jour en telechargeant les fichiers du CEHQ avec leur colonne Remarque, est bien plus forte**. Codes `E` (donnee estimee) et `R` (debit corrige pour effet de REFOULEMENT, c'est-a-dire sous glace ou embacle), sur les 16 stations d'OUTV, **tenue de cote 2022-2024** : janvier **85.4 %**, fevrier **87.3 %**, mars 60.8 %, decembre 47.0 %, avril 9.5 %, et **zero de mai a octobre**. Soit **21.3 % de toute la tenue de cote**. Sur 2000-2024 : 24.7 %. Le code dominant est `R` (4633 occurrences contre 21 pour `E` sur la station 040830) : ce n'est donc pas une interpolation pure mais une correction de refoulement appliquee par l'organisme, ce qui reste une reconstruction et non une lecture de courbe de tarage. Ma detection indirecte donnait 14-22 % en fevrier contre 87 en realite. CONSEQUENCE : de decembre a mars nous n'ajustons pas le modele sur une mesure mais sur la reconstruction d'un hydrologue qui n'avait ni le forcage ni le manteau sous les yeux ; Hydrotel est cale sur les MEMES series, ce qui peut expliquer une part de son avance hivernale. **LIMITE, et elle est DECISIVE : avril est un mois MESURE** (9.5 % de valeurs reconstruites en tenue de cote, contre 87 en fevrier). Notre pire mois, 0.753, porte donc sur des jours reellement observes et le trou d'avril reste ENTIER. R19 disqualifie le chantier hivernal comme cible d'ajustement, il ne disqualifie rien du chantier printanier -- il le renforce. Drapeaux ingeres en base par `.runs/quebec/ingest_cehq_flags.py` (colonnes `remark` et `reconstructed` de `observations`), module `meandre/data/cehq_loader.py` | 08-21 |
| R20 | Le deficit d'avril vient de la MASSE du manteau neigeux (conclusion laissee ouverte par R16b apres que la DATE de fonte se soit averee juste) | **MAL POSE**, budget de printemps ferme sans modele sur 24-25 ans. (1) Aucune penurie d'apport : l'ecoulement d'avril+mai vaut 0.45 de (pic de manteau + pluie d'avril-mai) sur outv, 0.43 sur sagu, 0.31 sur mont. Il entre plus du DOUBLE de ce qui sort ; le printemps est un probleme de PARTAGE, pas d'approvisionnement. (2) La neige est MINORITAIRE dans l'apport de printemps : 0.42 outv, 0.52 sagu, 0.29 mont. La pluie d'avril-mai (339 mm outv, 387 mm mont) est le plus gros terme. Viser la masse du manteau, c'etait viser la minorite de l'apport. PIEGE DE MESURE evite de justesse : ma premiere passe prenait la mediane de MARS pour le pic (99 mm sur outv au lieu de 238) et annoncait un rapport ecoulement/manteau de 2.09, physiquement impossible. Le pic se prend au maximum annuel du manteau median du reseau sur janvier-avril | 08-21 |
| R21 | Decembre relache l'eau d'avril : dans lequel de ses reservoirs le modele la perd-il ? | **REPONDU : dans AUCUN. Elle ne s'arrete jamais.** Cycle saisonnier des stocks sorti du champion OUTV en inference pure (`ETL_STOCKS=1`, controle a 0.8088 median, bilan ferme a +0.01 %). Le modele n'a qu'UN SEUL reservoir saisonnier, le manteau neigeux. Nappe **0 mm toute l'annee**, canopee 0. Sol L3 : 1044 mm, amplitude annuelle **14 mm** et **+1 mm de decembre a avril** -- a theta 0.394 pour une porosite 0.434, il est plein a 91 % et ne bouge plus : masse morte, pas reservoir. L2 respire 23 mm, L1 8 mm, milieu humide 5 mm. Sur 146 mm d'amplitude totale du stockage, le manteau en fait **121** ; sol et nappe reunis moins de 25. Consequence lisible dans les flux : en decembre 47.7 mm atteignent le sol et 45.0 mm ressortent en production, rien n'est retenu. Volumes : avril manque de 28 978 unites (5.1 % du total annuel a lui seul, la plus grosse erreur de l'annee) et novembre-decembre-janvier sont en exces de 15 915, soit **55 % du deficit d'avril**. Le transfert est reel et plus qu'a moitie identifie. **CORRECTION DANS L'HEURE par GRACE (voir R22) : la formule 'le modele n'a nulle part ou mettre l'eau' est FAUSSE au niveau du total.** L'amplitude saisonniere du stockage du modele (146 mm) vaut 102 % de celle que GRACE mesure (143 mm). Ce qui restait de R21 etait la COMPOSITION (121 des 146 mm en manteau) -- **et cet argument tombe AUSSI, le 08-22 avec R28** : le manteau reel vaut ~108 mm pour une amplitude GRACE de 143, donc la realite met ~35 mm dans le sol et la nappe quand le modele en met 25. C'est proche. Ne survit de R21 que le TRANSFERT DE VOLUME mensuel (avril -28 978 unites, novembre-decembre-janvier +15 915) et le constat que L3 ne respire pas. A RAPPROCHER de R11 (le debit seul prefere une recharge quasi nulle) et de l'effondrement de f_vert : le modele atteint 0.79-0.81 SANS AUCUNE eau souterraine, ce qui est l'enonce concret du probleme d'identifiabilite -- le debit ne contraint pas le stockage | 08-21 |
| R22 | Le modele manque de capacite de stockage saisonnier (corollaire attendu de R21) | **REFUTE sur le total, ETABLI sur la phase.** Cycle saisonnier du stock du modele confronte a GRACE (255 mois, 2002-2026, quality_ok), meme protocole sur 4 regions. (1) AMPLITUDE JUSTE : 146 mm pour le modele contre **143 mm** pour GRACE sur outv, soit 102 %. Le modele stocke la bonne QUANTITE. Repere robuste : les 4 regions donnent le meme cycle GRACE, maximum en mars ou avril, minimum en septembre, amplitude 143 (outv, sagu) a 170 mm (mont, gasp). (2) PHASE DECALEE D'UN MOIS : le stock du modele culmine en MARS a +90 mm, GRACE en AVRIL a +64. Surtout, en MAI GRACE tient encore +45 quand le modele est retombe a -6 -- **51 mm d'ecart, le plus gros de l'annee**, suivi de juin a 43. Le modele vide sa reserve pendant qu'un bassin reel la garde pleine. (3) Ca se recolle au debit : mai est le mois ou le modele produit TROP (1.138 de l'observe) juste apres avoir manque avril (0.753). Il ne relache pas trop tot dans l'ANNEE, il relache trop tot dans le PRINTEMPS. Ce qui reste de R21 est la COMPOSITION du stock (121 des 146 mm en manteau, sol et nappe sous 25 mm) : bon total par le mauvais chemin | 08-21 |
| R23 | GRACE n'est pas encore utilise comme contrainte : il faut le brancher (ce que supposait la tache 50) | **FAUX : il est branche depuis toujours, et c'est PIRE.** `w_tws = 0.2` est actif par defaut dans `gasp-v4.toml`, la config de base de TOUTES les regions ; le champion l'a donc eu pendant ses 30 epoques et n'en a rien appris. Raison mesuree : `tws_anomaly_loss` compare des mois INDIVIDUELS avec `sigma=25.0`, l'incertitude d'UNE estimation mensuelle GRACE. Le residu climatologique du modele vaut **26.2 mm** en moyenne quadratique, soit **1.05 ecart-type** : la perte lit un modele deja d'accord avec le satellite et ne pousse plus. Or nous avons ~21 ans par mois calendaire, donc l'incertitude sur le mois CLIMATOLOGIQUE est 25/sqrt(21) = **5.4 mm**, et le meme residu vaut alors **4.8 ecarts-types** (mai 9.2, juin 7.5, mars 6.6, fevrier 4.8). Une erreur saisonniere SYSTEMATIQUE moyennee sur vingt ans ne se juge pas a l'incertitude d'un mois : la contrainte se dilue dans une tolerance calibree sur le mauvais bruit. Profil du residu, avance de phase nette : +26 fev, +35 mars, -49 mai, -42 juin, -24 juil, +26 sep, +26 oct. CORRECTIF PROPOSE : terme climatologique (12 moyennes mensuelles, sigma ~ 25/sqrt(n)) EN PLUS du terme mensuel, pas a la place. DEUX DEFAUTS MINEURS releves au passage dans le stockage vu par la perte (trainer.py ~1015) : `theta1 * 0.30` code en dur alors que `z1 = 0.15` (double la respiration de la couche 1, 8 mm) et `_diag_chunk.wetland`, le champ MORT, au lieu de `wet_vol` (constant donc annule au centrage, mais les 5 mm reels du milieu humide manquent). Petits devant 26 mm | 08-21 |
| R24 | Les trois contraintes auxiliaires disent la meme chose et il suffit de les brancher | **NON : deux d'entre elles se CONTREDISENT au printemps, et la reparation naive aurait empire le modele.** Audit `ETL_AUX=1` sur le champion OUTV, meme moule pour les trois (part systematique du residu, significativite contre la dispersion interannuelle). **MODIS ET** : 77 % systematique, jusqu'a 27 ecarts-types ; le modele sous-evapore de janvier a aout, -0.32 mm/j en moyenne soit ~117 mm/an. Debranche (`w_et=0`). **J'avais justifie ce zero par un double ancrage -- le module de demande MLP etant entraine sur MOD16 -- et Essi a releve que le MLP avait ete ECARTE. Verification : exact, `ETL_ETP=linacre` pose `etp_channel = None` (etl_run.py l.313), donc la recette du champion tourne sur Linacre CALEE et la demande apprise est chargee puis ignoree. Il n'y a AUCUNE information MOD16 dans le champion : l'evaporation y est entierement libre, et ces 117 mm/an sont une contrainte reellement inutilisee, pas un doublon.** Le double ancrage ne vaut que pour le mode `appris`. Reserve qui demeure : MOD16 est biaise haut de 15-30 % a l'est, donc une part des 117 mm appartient au satellite. **MODIS couverture nivale** : 89 % systematique, 38 ecarts-types en mars ; le modele voit **PLUS** de neige que le satellite (+0.47 de fraction en mars, +0.45 en avril), janvier-fevrier et l'automne entierement masques par les nuages. **Activer w_snow=0.3 apres avoir repare la dette #14 aurait donc pousse le modele a ENLEVER de la neige en mars-avril, donc a fondre plus tot -- l'inverse exact de ce que demande GRACE.** Le bug etait accidentellement protecteur. ARBITRAGE : MODIS mesure une reflectance et sous-estime la neige sous couvert (OUTV est boise a 74 %, 32 % de coniferes), GRACE mesure une masse et ignore la canopee ; sur ce bassin GRACE est le temoin. **LE VOLET NEIGE DE R24 EST VOID (voir R28).** J'y ai ecrit que le modele avait la moitie de la neige reelle, sur un pic CanSWE de 238 mm qui etait un ARTEFACT D'AGREGATION. Le pic mesure correct sur OUTV vaut **108 mm** et le modele en accumule 121 : il en a legerement PLUS que la mesure. Ce qui reste de R24 est l'audit des trois contraintes et le fait que MOD10 pousse dans le mauvais sens ; la conclusion sur la masse du manteau, non. C'est la version chiffree de R21 (composition) et la cible du chantier d'identifiabilite | 08-21 |
| R25 | Le residu saisonnier de l'ET vient de la phenologie (K_c par GDD) | **FAUX en partie, et Essi l'avait dit : l'identifiabilite de l'ET passe aussi par la meteo et le SOL.** Nouveau diagnostic `ETL_AUX=1` (rapport ETR/ETP et saturation par couche, par mois). RESULTAT CENTRAL : la couche profonde L3 est a **1.00 de saturation DOUZE MOIS SUR DOUZE** (A CONSOLIDER : ce 1.00 est une moyenne arrondie, et la table des stocks donne 1044 mm pour une epaisseur mediane de 2.65 m, soit plutot 0.91 -- les deux se reconcilient si l'epaisseur MOYENNE differe de la mediane, mais ce n'est pas verifie. Sortie ajoutee au pilote le 08-21 : theta et porosite en valeur absolue, plus la part des couples noeud-jour epingles a la borne, qui distingue un reservoir *plein* d'un reservoir *presque plein*. A relire au prochain run). Ses 1044 mm ne sont pas un reservoir mais un TUYAU : tout ce qui y entre doit en ressortir. L1 respire entre 0.57 et 0.66, L2 entre 0.59 et 0.71, L3 jamais. Consequence directe : ETR/ETP est PLAT toute l'annee entre 0.81 et 0.88, le modele n'est JAMAIS limite par l'eau, en aucune saison -- alors qu'une foret boreale l'est en fin d'ete. Le residu d'ET a donc DEUX causes distinctes, pas une : (1) juin-juillet, deficit de -0.59 et -0.53 mm/j, cote DEMANDE, le rapport plafonne a 0.88 et le modele ne peut pas evaporer plus (plafond de K_c ou d'ETP) ; (2) septembre-octobre, exces de +0.51 et +0.66, cote OFFRE, le modele continue d'evaporer parce qu'il puise dans une couche profonde inepuisable. Ce lien explique aussi R21/R22 (aucun stockage saisonnier dans le sol : L3 est plein, il ne peut rien stocker) et R11 (la recharge ne peut pas descendre dans un reservoir sature). UNE MEME CAUSE STRUCTURELLE derriere trois chantiers separes | 08-21 |
| R26 | L'avance hivernale d'Hydrotel vient de ce qu'il est cale sur les MEMES reconstructions du CEHQ que celles contre lesquelles on le juge (consequence redoutee de R19) | **REFUTE, sans faire tourner meandre** (`.runs/quebec/hydrotel_vs_flags.py`, ensemble des 6 membres contre les observations drapeautees, tenue de cote). PREMIER TEST, ECARTE PARCE QUE MAL CONSTRUIT : jours mesures 0.7990 contre reconstruits 0.5935 sur OUTV, ecart -0.217 cohérent sur les 6 membres -- mais les jours reconstruits SONT les jours d'hiver, donc ce contraste oppose deux saisons, pas deux qualites de donnee. TEST VALIDE, INTRAMENSUEL (decembre, mars, avril, novembre contiennent les deux classes en quantite) : ecart median **-0.061** sur 14 couples region x mois (6 regions), de -0.47 a +0.18, **cinq des quatorze POSITIFS**. Autrement dit : a saison egale, le drapeau ne predit RIEN sur la qualite de l'accord. Aucune trace d'un ajustement d'Hydrotel a la reconstruction, et **pas davantage de preuve que les jours reconstruits seraient plus difficiles** -- une premiere passe a 10 couples donnait -0.135 et je l'avais lu ainsi ; a 14 couples le chiffre tombe a -0.061, sous la dispersion. Consequence pratique : le drapeau change ce qu'on peut DIRE d'un score hivernal (il ne mesure pas un accord a la mesure) mais ne justifie aucun reponderage, et surtout **ne touche pas au CLASSEMENT entre modeles**. RESERVE : 14 couples, 5 a 19 stations chacun, et a l'interieur d'un mois les jours drapeautes restent les plus froids. Le contraste brut, ecarte parce que confondu avec la saison, valait -0.135 sur 78 couples region x membre | 08-21 |
| R27 | Le modele a moitie moins de neige que la realite (lecture de CanSWE faite plus tot dans R24) | **MAL ATTRIBUE. Le modele ne peut pas avoir plus de neige qu'il n'en recoit.** Borne independante calculee sur le forcage lui-meme : au seuil pluie-neige du projet (-2.2168 degres), CaSR livre **174 mm** de neige en mediane sur OUTV de novembre a mars (q10 102, q90 213 ; 337 mm de precipitation totale sur la periode). Meme en comptant TOUT ce qui tombe sous 0 degre, on ne depasse pas **232 mm**. Le pic simule de 121 mm vaut donc **70 % de la neige effectivement recue**, ce qui est physiquement normal (une part fond en cours d'hiver). Mais le pic CanSWE de 238 mm **DEPASSE la neige disponible** : aucun reglage de physique nivale ne peut y amener le modele. DEUX EXPLICATIONS, toutes deux recevables et non departagees : (a) CaSR SOUS-CAPTE la precipitation solide -- le sous-captage des pluviometres par vent est de 20 a 50 % et bien documente, et il va exactement dans ce sens ; (b) les sites CanSWE sont places la ou la neige s'accumule et ne representent pas la moyenne du bassin. CONSEQUENCE OPERATIONNELLE IMMEDIATE : le run aux-B, prevu avec la masse CanSWE comme cible de NIVEAU, a ete reconfigure AVANT demarrage en ablation du terme climatologique GRACE. Pousser le modele vers 238 mm l'aurait mis contre son propre bilan de masse. La cible CanSWE reste cablee et testee ; elle attend d'etre posee en ANOMALIE (la forme, pas le niveau) comme l'ET, ou que le sous-captage soit corrige en amont. A rapprocher de [reference_casr_wetbias_et_litterature] : CaSR sur-estime la pluie ET sous-capterait la neige, ce qui n'est pas contradictoire | 08-21 |
| R28 | Le pic de manteau mesure vaut 238 mm sur OUTV (chiffre utilise par R24 puis R27) | **ARTEFACT D'AGREGATION, ET TOUTE LA CONCLUSION NIVALE TOMBE AVEC.** Le 238 etait le maximum sur l'hiver de la MEDIANE JOURNALIERE du reseau. Or les sites qui rapportent changent d'un jour a l'autre, et le rapport pic/neige-tombee correle a **+0.58 avec l'altitude du site** : cette statistique retenait le jour ou les sites les plus enneiges rapportaient. Statistique correcte (pic par site et par saison, puis mediane sur les sites, chacun apportant ses saisons completes) : **108 mm** sur OUTV, q25 82, q75 152, sur 71 sites. **Le modele en accumule 121 : il a donc LEGEREMENT PLUS de neige que la mesure, pas la moitie.** Ce qui reste ETABLI par ailleurs : le rapport pic mesure / neige TOMBEE au noeud vaut **0.75** en mediane, avec 53 sites sur 69 sous 1.0. Le forcage d'OUTV livre donc assez de neige pour expliquer ce qui est mesure, et le deficit de 30 % du modele est un probleme de RETENTION du manteau -- fonte ou sublimation hivernale de trop -- pas d'apport manquant. CE QUI SURVIT AILLEURS : sagu (mesure 258 pour 213 tombes, rapport 1.21) et slno (244 pour 195, 1.25) montrent un pic mesure SUPERIEUR a la neige disponible, donc un vrai probleme de forcage ou de representativite -- mais pas sur la region ou tous les diagnostics ont ete faits. CONSEQUENCES EN CASCADE, toutes a porter au registre : le volet neige de R24 est VOID ; la premisse de R27 (CanSWE 238 > forcage 174) est VOIDE, meme si sa decision operationnelle -- ne pas imposer CanSWE en NIVEAU -- reste la bonne pour sagu/slno ; et **R21 perd son argument de COMPOSITION** : avec un manteau reel de ~108 mm pour une amplitude GRACE de 143, la realite met ~35 mm dans le sol et la nappe quand le modele en met 25, ce qui est proche. CE QUI RESTE SOLIDE : la PHASE (R22), modele au pic en mars contre avril pour GRACE, 51 mm d'ecart en mai. LECON DE METHODE : j'ai utilise ce 238 dans trois raisonnements successifs sans jamais verifier comment il etait calcule. Une statistique qui porte une conclusion doit etre recalculee d'une SECONDE facon avant d'etre citee | 08-22 |
| R29 | Apres avoir elimine la QUANTITE d'eau sous toutes ses formes, que reste-t-il ? | **LA VITESSE DE SORTIE, et le verrou est identifie : `krec` a 1.3e-7 m/h, herite du calage Hydrotel.** Chaine complete. (1) Dans bv3c2, la couche profonde n'a QU'UN exutoire : `q3 = krec x z3 x theta3` (l.173), qui alimente directement `prod_base`. (2) Sans `ETL_KREC_LIBRE`, `imposed_retention_curve` inclut `krec` dans la courbe imposee, et la colonne FUSIONNE le calage PAR-DESSUS la sortie du champ (hydrotel_column l.379) : la valeur du NeRF n'est pas utilisee. (3) Le calage Hydrotel porte krec ~ **1.3e-7 m/h** -- chez Hydrotel c'est une fuite jamais restituee, que son calage etrangle faute d'aquifere. (4) La capacite de drainage de L3 vaut donc **0.0036 mm/j** : elle ne peut pas se vider, d'ou une L3 epinglee a saturation (R25). (5) L'`AquiferModule` de meandre est alimente par ce meme `pb` : robinet ferme, reservoir vide en aval, d'ou la nappe a **0 mm toute l'annee** (R21). (6) Sans reserve lente, le modele n'a AUCUN endroit ou faire attendre l'eau plusieurs mois -- en decembre, 47.7 mm atteignent le sol et 45.0 ressortent. Exces en novembre-janvier, trou en avril, exces en mai : c'est la signature. CE QUI S'AJOUTE, MESURE : le taux de vidange souterrain d'OUTV vaut **0.0273 /j** (residence 37 j) sur 1316 recessions hivernales pures, composante lente a 0.0090 (111 j) ; le champion tourne a 0.0645, soit 2.4 a 7 fois trop vite. POURQUOI CA N'A JAMAIS ETE REPARE : chaque tentative de donner une reserve lente a ete jugee AU DEBIT, et le debit seul prefere qu'il n'y en ait pas (R11) -- le modele atteint 0.79 sans une goutte d'eau souterraine. C'est l'enonce exact du probleme d'identifiabilite. CE QUI A CHANGE : la phase GRACE (R22, +45 mm en mai contre -6) est une observation qui exige le contraire et ne depend pas du debit, et elle n'etait lisible qu'a partir du 08-21 (R23). AVERTISSEMENT : ce diagnostic etait DEJA ECRIT dans un commentaire d'`etl_run.py` depuis le 17 aout -- « reservoir branche en aval d'un robinet ferme » -- sans que personne le relie au trou d'avril. Test en file : `file_residence.sh`, krec APPRIS par le champ avec sa moyenne ancree (prior_on_krec), k_gw a 0.0273 puis 0.0090, GRACE laissee HORS de la perte pour rester juge | 08-22 |
| R30 | Le seuil pluie-neige du projet (-2.2168 degres, herite de la plateforme Hydrotel) est un parametre physique qu'on peut importer tel quel | **NON, et il explique A LUI SEUL le deficit de manteau.** Trois chiffres qui convergent, tous independants du debit. (1) A site et a jour egaux, le modele a 0.65 a 0.72 de la neige mesuree par CanSWE : il lui en manque ~35 %. (2) Sur OUTV, le seuil qui produit exactement **+35 % de neige** est **+0.3 degre** (cumuls nov-mars : 174 mm a -2.22, 210 a -1.0, 232 a 0.0, 254 a +1.0). (3) **+0.3 degre est la valeur que la litterature attend** pour un partage pluie-neige, la plage usuelle allant de 0 a +2. Un seuil a -2.2 compte comme PLUIE tout ce qui tombe a -2 degres, ce qui est physiquement invraisemblable : c'est tres probablement une COMPENSATION calee pour le modele de fonte d'Hydrotel, et l'importer dans meandre, qui a un autre module de fonte, revient a ancrer une compensation au lieu d'un processus -- exactement ce que la loi des ancrages interdit. OU LE DEFICIT EST LE PIRE, novembre 0.66 et decembre 0.55, un seuil a 0 degre donne **+48 %** de neige : decembre remonterait vers 0.81 et novembre vers 0.98. ET CA RECOLLE AU DEBIT : decembre produit 1.235 fois l'observe, c'est-a-dire que l'eau qui aurait du rester au sol part a la riviere -- deficit de manteau et exces de debit sont LE MEME phenomene. Note : R16 avait teste +1.0 en inference pure (0.7919 contre 0.7880, decembre de 1.207 a 0.857) et conclu que le seuil gouverne l'axe decembre-mai sans toucher avril. Ce qui est nouveau ici, c'est que la valeur n'est plus choisie pour le score mais DERIVEE d'une mesure de masse, et qu'elle tombe sur la plage physique. DEUXIEME LEVIER du transfert decembre-avril, distinct de la reserve lente (R29) et non teste : tache 54 | 08-22 |
| R31 | Resultat du run aux-A : GRACE climatologique + correction du stockage + ET en mode anomalie | **NEUTRE SUR LE SCORE, NEGATIF SUR LE VOLUME.** Tenu de cote 0.7932 median contre 0.7880 au champion : +0.005, sous le bruit de 0.025 mesure entre graines. Mais le bilan annuel passe de -3.4 % a **-8.1 %**, avec juillet 0.820, aout 0.843 et octobre 0.755 : l'ET en mode anomalie a pousse l'evaporation de +33 mm/an (420 -> 452, concentre juin-aout a +6/+8 mm/mois), soit un TIERS du chemin vers MODIS qui demandait +17 mm/mois en ete. La direction obeit a la contrainte ; le cout en debit vient de ce que la precipitation, elle, ne bouge pas -- pousser l'ET vers MODIS avec un P deja juste NE PEUT QUE creuser le debit. Levier pour la suite : w_et 0.3-0.5 au lieu de 1.0, ou accepter l'arbitrage si le biais de P est reel (le registre note CaSR humide). Avril reste a 0.754 (inchange), decembre descend de 1.235 a 1.188. La nappe reste a 0 mm tous les mois et le manteau a 0.64-0.70 de la mesure : ni l'un ni l'autre n'etait vise, donc c'est coherent. L'EXCURSION VIOLENTE N'EST PAS LE TERME CLIMATOLOGIQUE : le run aux-B, qui ne l'a pas, subit le MEME redemarrage autopilote (regression 23.4 % contre 30.1 %). Elle vient de l'ET en anomalie ou de la correction du stockage. NOTE DE PROTOCOLE : aux-A a tourne DEUX FOIS par ma faute (fichier de file edite pendant son execution, bash relit avec un decalage en octets) ; les deux passages donnent exactement 0.7932 / 0.7723, ce qui prouve au passage la reproductibilite bout en bout | 08-22 |
| R32 | Le deficit de manteau vient de la fonte hivernale ou de l'interception sous couvert | **NI L'UN NI L'AUTRE : c'est l'ACCUMULATION.** Diagnostic du pilote lui-meme, 3203 intervalles APPARIES (memes dates, memes sites), lu pour la premiere fois le 08-22 alors qu'il etait imprime depuis longtemps. (1) FONTE HIVERNALE PRESQUE JUSTE : le modele perd 27.1 % de ce qu'il accumule de decembre a fevrier, la mesure 22.9 %, et un manteau reel 24.6 %. (2) DATE DE DISPARITION JUSTE : +2, 0, 0 jours (confirme R16b). (3) **L'ACCUMULATION EST LE TROU** : le modele accumule 9248 mm quand la mesure en accumule 16065 sur les memes intervalles, soit 58 %. (4) LE SEUIL EN EXPLIQUE UNE PART : la regle pluie-neige ne fait tomber que 13371 mm, moins que ce que la mesure accumule reellement -- elle rate donc ~20 % de la neige (coherent avec R30). (5) IL RESTE 31 % entre ce qui tombe (13371) et ce qui s'accumule (9248). (6) **CE N'EST PAS DE L'INTERCEPTION** : par quartile de fraction forestiere le rapport simule/mesure vaut 0.55, 0.71, 0.66, 0.76 -- il ne DECROIT pas avec la foret, il monterait plutot. Le test est celui que le pilote documente lui-meme : ratio decroissant = interception attendue, ratio plat = masse reellement absente. Ratios de pic par annee : 0.77, 0.36, 0.53 | 08-22 |
| R33 | Le terme climatologique GRACE (w_tws_clim=0.05) repare la phase du stockage | **NON -- et la raison est structurelle, pas un mauvais poids.** Ablation PAIREE aux-A contre aux-B (meme graine, meme code, seule difference w_tws_clim). SCORE : 0.7932 contre 0.7827, +0.0105 pour le terme -- comparaison appariee, donc plus fine que le bruit inter-graines de 0.025 ; le terme ne coute rien et aide peut-etre un peu. PHASE : biais GRACE de mai **-45.0 contre -45.4**, mars +46.6 contre +45.4, residu systematique 26.3 contre 26.2 -- IDENTIQUES. Le terme n'a pas deplace la phase d'un millimetre. POURQUOI : dans cette recette `krec` est IMPOSE par le calage Hydrotel a 1.3e-7 (robinet ferme, R29) et n'est pas apprenable ; la contrainte GRACE ne peut pas pousser de l'eau dans un reservoir dont la vanne d'entree est hors du graphe de gradient. Aucun poids de perte ne peut corriger une physique qui n'a pas le degre de liberte -- c'est le complement exact de R11 (le debit seul ne VEUT pas de reserve ; GRACE la veut mais ne PEUT pas l'obtenir tant que krec est gele). LE VRAI TEST du terme est avec krec APPRIS -- or les runs res-C1/C2 tournent volontairement avec w_tws_clim=0 (GRACE en juge) : le run combine krec-appris + w_tws_clim reste A FAIRE apres lecture de C1/C2. SECONDAIRE : l'excursion violente (redemarrage autopilote dans A ET B) est confirmee independante du terme ; suspects restants, ET anomalie ou correction du stockage. Et le diagnostic des jours mesures (tache 48) rend son premier chiffre : median 0.7928 sur les jours MESURES contre 0.7827 tous jours confondus, +0.010 -- ecarter les reconstructions REMONTE le score, l'hiver reconstruit nous coutait des points | 08-22 |
| R34 | Ouvrir le robinet (krec appris, moyenne ancree 2e-5) et poser le taux de vidange mesure suffit a donner au modele sa reserve saisonniere | **MOITIE OUI, MOITIE NON -- la reserve EXISTE enfin, mais elle est PLATE, et la raison est en amont.** Runs res-C1 (k_gw=0.0273, residence 37 j) et res-C2 (0.0090, 111 j), 10 epoques, GRACE en JUGE hors de la perte. CE QUI MARCHE : premiere nappe non nulle du projet (C1 : 7-9 mm ; C2 : 28-34, l'echelle suit la residence comme S=Q/k le predit) ; L3 respire enfin (amplitude 37 mm contre 14, epinglage en recul, saturation a 0.96 en septembre) ; et la PHASE GRACE s'ameliore pour la premiere fois : residu systematique **26.3 -> 22.2 mm (-16 %)**, mai -45 -> -35.5 (C1), mars +46.6 -> +37.0 (C2). Obtenu par la PHYSIQUE seule. COUT : C1 0.7584, C2 0.7785, contre 0.7885 -- l'arbitrage score/realisme de R11, chiffre. CE QUI NE MARCHE PAS : la nappe est PLATE (C2 : dec 30 -> avr 29), et culmine aout-octobre quand GRACE culmine en avril. CAUSE STRUCTURELLE : la recharge q3 = krec x z3 x theta3 est quasi CONSTANTE parce que theta3 ne varie qu'entre 0.96 et 1.00 -- un reservoir lineaire alimente en continu est plat par construction, quel que soit k_gw. Pour que la nappe porte la saisonnalite de GRACE (+45 mm tenus en mai), la RECHARGE doit avoir un pic de crue, donc L3 doit se desaturer assez pour que la fonte la re-remplisse. Le levier n'est plus la vidange de la nappe mais la RESPIRATION DE L3 -- krec plus grand encore, ou le reservoir non lineaire (pret, 7 tests) place au niveau du DRAINAGE de L3 plutot qu'a la nappe. NOTE : l'excursion violente + redemarrage autopilote a frappe les 4 runs A/B/C1/C2, avec et sans chaque terme nouveau -- le facteur commun est la montee du taux d'apprentissage vers 5e-4, pas nos ajouts | 08-22 |
| R35 | Le seuil pluie-neige derive de CanSWE (+0.3 degre) repare l'accumulation nivale (hypothese R30) | **CONFIRME SUR TOUTE LA LIGNE, ET C'EST LE RESULTAT LE PLUS PROPRE DE LA CAMPAGNE.** Paire seuil-D1 (+0.3) contre seuil-D2 (controle -2.2168), meme code, memes 10 epoques, attribution nette. LE JUGE (CanSWE apparie site et jour) : novembre 0.66 -> **1.09**, decembre 0.55 -> **0.96**, janvier 0.65 -> 0.95, fevrier 0.66 -> 0.88 -- le deficit d'accumulation de debut d'hiver est EFFACE (le controle D2 reste a 0.52-0.67, donc c'est bien le seuil). Accumulation appariee : 12 415 mm sur 16 065 mesures (77 %, contre 58 % avant) ; pertes dec-fev 25.4 % contre 24.6 de reference -- la fonte hivernale reste juste. LE DEBIT SUIT : decembre 1.235 -> **0.999**, novembre -> 0.988, avril 0.753 -> **0.819**, gamma 0.9245 -> **1.0014**. ET LE SCORE AUSSI : held-out **0.8014** contre 0.7905 au controle apparie et 0.7885 au champion -- la valeur DERIVEE d'une mesure de masse, dans la plage litterature, bat la valeur heritee du calage Hydrotel MEME AU KGE. La loi des ancrages tenait (le seuil -2.2 etait une compensation du modele de fonte d'Hydrotel, pas un processus). CE QUI RESTE : mai passe de 1.138 a **1.292** -- la neige gagnee fond tard et deborde en mai ; mars descend a 0.792. C'est exactement ce que la fonte SAISONNIERE (serie E, amp qui accroit la fonte d'avril) doit redistribuer ; E4 teste seuil + fonte saisonniere + sublimation ensemble | 08-22 |
| R36 | Serie E (inference pure sur le champion, controle E0 exact) : que valent la fonte saisonniere et la sublimation ? | **FONTE SAISONNIERE RETENUE (amp 0.5), SUBLIMATION KUZMIN REJETEE EN L'ETAT.** E1 (amp 0.5) : novembre 0.66 -> 0.70, decembre 0.55 -> 0.60, et surtout MAI 1.14 -> 0.88 -- elle redistribue la fonte vers avril, exactement le complement du seuil (R35) dont le seul defaut etait mai a 1.292. E3 (Kuzmin seul) : le manteau s'effondre partout (decembre 0.40, -0.15 sur tous les mois) -- la formule suppose un vent de plaine degagee, irrealiste sous 74 % de foret ; a reprendre avec un facteur d'abri par canopee, jamais sans. E0 reproduit le champion a l'identique : le banc est valide. GENERALISATION DU SEUIL (remarque d'Essi : un +0.3 calibre sur OUTV est un parametre regional de plus) : la variable physique du partage est le BULBE HUMIDE (Jennings 2018 : les seuils AIR varient de 0.6 a 3.8 degres dans l'hemisphere, la variance est l'humidite). Verifie sur nos 6 regions : le seuil Twb equivalent au +0.3 air d'OUTV tient dans **[-1.0, -0.7]** partout (HR hivernale 77-82 %). Implemente opt-in (`split_mode=wet_bulb`, Stull 2011, e_a du forcage, 3 tests) ; ETL_SEUIL_TWB. En validation F1 puis integration INT1 (tous les gagnants, GRACE enfin apprenable) | 08-22 |
| R37 | Integration INT1 : bulbe humide Twb=-0.8 + fonte saisonniere 0.5 + krec appris + k_gw 0.0273 + GRACE clim actif + w_et 0.4, 30 epoques demandees | **LE SYSTEME NIVAL EST REPARE, SANS AUCUN PARAMETRE REGIONAL, ET LE SCORE SUIT.** Arrete par patience a ~11 epoques (premier arret anticipe du pilote quebecois, dette #16 close la veille). VALIDATION F1 (inference pure) : le seuil en bulbe humide reproduit sur OUTV les gains du seuil air derive (novembre 1.07, decembre 0.95) -- la generalisation demandee par Essi tient. INT1 ENTRAINE : manteau CanSWE a **1.10 / 0.99 / 1.00 / 0.93 / 0.95** de novembre a mars (contre 0.66/0.55/0.65/0.66/0.70 au champion), pertes hivernales 19.7 % (mesure 22.9), dates de disparition +1/0, quartiles forestiers plats a ~1.0. DEBIT : avril 0.753 -> **0.865**, decembre 1.235 -> 0.897, gamma **1.0416**, r 0.8893. SCORE : **0.7963** tous jours, **0.8090 sur les jours mesures** -- au-dessus du champion (0.7885) avec une physique defendable. CE QUI RESTE, et c'est le prochain chantier : (1) MAI deborde encore (1.235) -- le manteau plus gros fond tard malgre amp 0.5 ; (2) la phase GRACE s'est DEGRADEE (mars +58, residu 30.2 contre 22.2 a C1) : CanSWE dit que la neige de mars est juste, GRACE dit que le stockage TOTAL de mars est trop haut -- la compensation reelle est un SOL qui se vide en hiver, et L3 ne respire toujours pas (nappe 9-12 mm, plate : meme le terme GRACE actif ne peut pas inventer une recharge saisonniere, confirmation de R34) ; (3) l'hiver produit maintenant un peu SOUS l'observe (janvier 0.765) -- mais l'observe d'hiver est reconstruit a 85 % (R19), et sur les jours mesures le score monte. PROCHAIN LEVIER STRUCTUREL : faire respirer L3 (drainage non lineaire au niveau de L3, module pret et teste) | 08-22 |
| R38 | Faire respirer L3 par la courbure seule du drainage (exposant n, plafond preserve) | **REFUTE PROPREMENT, puis REPARE en relevant le plafond -- et la phase GRACE tombe a son plus bas historique.** Bancs G0-G6, inference sur le point de reprise int1. (1) L'exposant SEUL ne fait RIEN (G1-G3 identiques a G0, INT2 = INT1 apres entrainement) : le flux sortant de L3 est deja a son plafond (9 mm/mois constant) et l'entree hivernale l'egale, donc theta ne descend jamais sous saturation et la courbure n'agit jamais. Erreur de conception assumee : j'avais preserve le plafond par principe alors que le plafond EST la contrainte. (2) PLAFOND RELEVE + exposant : G5 (krec 1e-4, n=8) fait respirer L3 de **95 mm** (909 en mars, 1004 en mai) et le residu systematique GRACE tombe de 30.2 a **11.2 mm** -- mai +8.7, juin +0.1, le plus bas jamais mesure, la phase est quasi reparee PAR LA PHYSIQUE. G4 (5e-5, n=8) : 19.7 mm, intermediaire. (3) LE PRIX EN INFERENCE : score 0.36 (G5) -- les poids d'int1 n'ont jamais vu cette partition, le debit de base triple sous eux (28-47 mm/mois). C'est un artefact de non-co-adaptation, pas une refutation : INT3 entraine cette nuit (krec init et prior a 7e-5, n=8, tout le reste de la recette int1) pour voir si le NeRF redistribue et ou s'arrete l'arbitrage score/phase | 08-22 |
| R39 | La nappe non lineaire (Q=q_ref*(S/100)^b) restitue la crue absorbee et repare avril | **REFUTE en inference (G7/G8 sur les poids INT3) : avril tombe a 0.47 et la restitution culmine en JUIN-JUILLET, pas en avril.** La chaine souterraine entiere suit la fonte avec 1 a 2 mois de retard (L3 culmine en mai, la nappe apres), donc toute absorption de la crue vers le souterrain VOLE avril, quel que soit le profil de vidange. LA FRONTIERE SCORE/PHASE DU JOUR, mesuree : INT1 0.796 de KGE pour 30.2 mm de residu GRACE ; INT3 0.645 pour 19.8 ; G5 0.36 pour 11.2. Le conflit est structurel : les jauges veulent l'eau de fonte EN AVRIL, GRACE veut du stockage en avril-mai -- les deux ensemble exigent que le modele fasse les deux SIMULTANEMENT, ce que la colonne actuelle ne sait pas (la fonte passe soit en surface tout de suite, soit dans un souterrain qui rend trop tard). HYPOTHESES POUR LA SUITE, a instruire par ANALYSE avant tout nouveau run : (a) le biais de mars de GRACE est en partie un artefact d'EMPREINTE -- le pixel de ~300 km voit du territoire hors domaine ; a verifier en comparant l'empreinte reelle au masque du bassin ; (b) la voie manquante est un stockage RAPIDE-lent (milieux humides, plaines d'inondation, lacs) qui retient la crue QUELQUES SEMAINES -- pas le souterrain profond ; (c) l'arbitrage a retenir pour l'instant est INT1 (nival repare, 0.796) en attendant de departager (a) et (b) | 08-22 |
| R40 | La voie rapide-lente de R39 est le reservoir de milieux humides existant, en allongeant sa vidange (c_prod 10 -> 30-60 j) | **REFUTE NET, et l'agregation GRACE est disculpee au passage.** (1) c_prod x3 et x6 sur les poids INT1 : stock de milieu humide IDENTIQUE (2-8 mm), score 0.7977/0.7980, GRACE 30.1 -- aucun effet, parce que le reservoir ne depasse jamais son volume normal, seul domaine ou c_prod agit ; l'entree est bornee par la fraction drainee PHYSITEL et la geometrie SWAT. Troisieme voie de retention refutee (sol profond R38-39, nappe lineaire et puissance R39, milieu humide ici). (2) GRACE AU MASQUE : l'agregation bbox ne remplit le bassin qu'a 50-59 %, mais la serie ponderee par le masque (mascons CSR 0.25 degre en cache) correle a 0.984-0.999 avec la bbox et ne deplace le cycle d'OUTV que de -0.1 mm en mars et +1.7 en mai : **le mars a +58 d'INT1 n'est PAS un artefact d'agregation**. Seule GASP bouge vraiment (+11.8 mm en avril), a reprendre au scale-up. Les series regionales ne sont pas des clones (corr croisees 0.72-0.94, pics mars/avril, amplitudes 134-195 mm) : la contrainte porte du vrai signal regional. CE QUI RESTE pour le conflit avril/stockage : le decalage residuel ressemble a 2 semaines de retard fonte-vers-debit (avril 0.866, mai 1.21) que ni le routage (borne a 48 h) ni les reservoirs testes ne portent ; candidats restants = forme intra-avril de la courbe de fonte, ou plaine d'inondation au routage (a construire). PRIORITE DEPLACEE : valider la recette INT1 sur d'autres regions -- la generalisation est l'enjeu du projet, pas le dernier centieme d'OUTV | 08-23 |
| R41 | La recette INT1 sans bouton regional (Twb -0.8 unique, fonte saisonniere, krec ancre litterature, k_gw champ provincial) generalise-t-elle hors d'OUTV ? | **OUI au debit, trois regions sur trois ; et le juge nival, lu avec les plafonds de forcage, dit la meme chose sauf en Gaspesie.** Flotte gasp/sagu/slno, 30 epoques avec arret anticipe, AUCUNE valeur derivee de ces regions. SCORES : gasp **0.7134** contre 0.577 a la meilleure reference ancree (+0.14, la plus grosse marche du projet sur cette region) ; slno **0.7631** contre 0.7106 (+0.05) ; sagu **0.7438** contre 0.7587 (-0.015, parite dans le bruit de 0.025). NIVAL : sagu 0.69-0.79 et slno 0.80-0.98 de la mesure -- MAIS R28 avait etabli que le pic CanSWE de ces deux regions DEPASSE la neige que CaSR fait tomber (rapports 1.21 et 1.25) : leur plafond atteignable est ~0.83 et ~0.80, et les runs y sont. Le residu nival y est le SOUS-CAPTAGE DU FORCAGE, pas la recette. L'ANOMALIE RESTANTE EST GASP : manteau simule a **1.8-2.5x** la mesure -- soit le seuil unique casse en climat maritime (enormement de precipitation pres de zero degre, levier maximal), soit l'echantillon CanSWE gaspesien est trop mince pour juger (49 sites, 347 jours avec donnees, contre 4872-5497 ailleurs) ; beta 0.849 y suggere par ailleurs de l'eau enfermee en trop. A departager par la meme derivation de masse que R30, faite SUR GASP. FLOTTE 2, memes conditions : mont **0.6953** contre 0.6243 a la meilleure reference meandre (+0.07) et 0.4987 au diagnostic du 21 aout (+0.20) -- la pire region du projet rejoint la zone de l'ensemble Hydrotel (0.6631 mesure dans le meme protocole le 21) ; abit 0.5244 contre 0.5475 (parite, mais n=3 stations et gamma 0.586 : les lacs abitibiens restent le chantier lacs, tache 40 -- et la sortie neige d'octobre y imprime un ratio aberrant 1e10, denominateur quasi nul a borner). BILAN CINQ REGIONS : gasp +0.14, mont +0.07 a +0.20 selon la reference, slno +0.05, sagu et abit parite. FLOTTE 3 (premieres references de ces regions via ce pilote, 1-2 stations chacune) : cnda 0.7849, cndb 0.6993, cndc 0.7222, cndd 0.8057, cnde 0.5281 ; labi 0.5035 contre 0.5567 avant (n=1, bruit) ; vaud n=0 (aucune station ne passe le filtre de 365 jours valides, region hors comparaison). CARTE PROVINCIALE COMPLETE : douze regions courues avec UNE SEULE recette. LECTURE D'ENSEMBLE : la physique plus le champ appris portent la geographie mieux que les constantes regionales qu'ils remplacent -- premier test grandeur nature de la these du projet, positif | 08-23/24 |
| R42 | D'ou vient le manteau gaspesien a 2x la mesure (anomalie de R41) ? | **TROIS SUSPECTS ELIMINES, DEUX RESTENT, et le debit n'attend pas la reponse.** Elimines par mesure : (1) le seuil pluie-neige -- meme a Twb -3.0, GASP ne perd que 18 % de neige, car 11 % seulement de sa precipitation d'hiver tombe dans la zone de levier [-2,+1] : un facteur 2 est hors de portee du seuil ; (2) la fonte saisonniere -- l'ablation amp 0 -> 0.5 sur le meme point de reprise change les ratios de moins de 0.1 (1.82/1.71/1.94 contre 1.91/1.79/2.02), et l'amplitude AIDE le score (0.7134 contre 0.6948) ; (3) l'appariement site-noeud -- ecart d'altitude median -36 m, 2 sites sur 56 au-dela de 150 m. RESTENT, ET MON HYPOTHESE MARITIME EST DEJA AFFAIBLIE PAR LA MESURE : le forcage donne a GASP MOINS de jours de fonte de mi-hiver qu'a OUTV en moyenne de bassin (dec 2.6 contre 3.8, jan 0.8 contre 1.2) -- l'interieur gaspesien est froid, la fonte maritime de bassin n'explique rien. Les suspects resserres : (a) un biais FROID de CaSR sur les cellules COTIERES ou vivent les sites CanSWE (232 m d'altitude mediane) -- les sites reels fondent en mi-hiver, les noeuds du modele a ces memes endroits non, parce que la maille de 0.25 degre melange cote et interieur ; (b) une surestimation de la precipitation solide cotiere. Le mode ETI reste justifie pour lui-meme (part radiative mesuree : 0.10 en hiver contre 0.20-0.29 au printemps dans les DEUX regions -- le substitut sinusoidal encodait bien ce cycle, la radiation reelle le rend exact), mais s'il ne corrige PAS le manteau gaspesien, le verdict pointera le forcage cotier, pas la fonte. NOTE : l'echantillon CanSWE gaspesien n'a AUCUNE donnee de novembre-decembre (releves janvier-avril seulement), donc le debut d'hiver y est injugeable de toute facon. Et malgre l'anomalie, GASP gagne +0.14 au debit : la question est diagnostique, pas bloquante | 08-23 |
| R43 | Un seuil de bulbe humide UNIQUE porte le partage pluie-neige partout (these de R35/R36, question d'Essi sur la generalisation) | **VRAI SUR LE CONTINENT, FAUX EN GASPESIE, et l'ecart pointe le FORCAGE cotier plutot que la physique du partage.** Banc neige-seule aux noeuds des sites CanSWE (nouveau : .runs/quebec/snow_bench.py, ~50 variantes a l'heure sur CPU, ne juge que le manteau -- le debit et l'apprentissage restent aux runs complets ; controle nu conforme au pilote). Balayage Twb par region : OUTV optimal a **-0.8** (0.83-0.95, se degrade en refroidissant), SAGU pareil (-0.8 le meilleur, niveau bas explique par le sous-captage R28), mais GASP reste a **1.20-1.35 meme a Twb -3.0** -- AUCUNE valeur de bulbe humide ne tient ses sites, alors que le seuil AIR du projet Hydrotel (-3.07) les met a 1.01-1.16. INDICE CONVERGENT : la plateforme Hydrotel elle-meme cale GASP a -3.07 quand toutes les autres regions sont a -2.2168 -- les calibreurs d'Hydrotel avaient DEJA besoin d'un degre de plus de froid en Gaspesie dans LEUR forcage. Le suspect principal devient un biais CHAUD des temperatures cotieres du forcage aux sites (un noeud lu 1-2 degres trop chaud exige un seuil 1-2 degres plus froid pour compenser), a verifier contre les stations GHCN cotieres (donnees en cache). OPTIONS : (a) diagnostiquer le biais T cotier d'abord -- si confirme, corriger le FORCAGE et garder le Twb unique ; (b) seuil en sortie de champ, ancre sur Twb -0.8, libre de devier ou les donnees de masse l'exigent -- la reponse modele qui absorbe aussi (a). L'option (a) est la plus propre epistemiquement : on ne donne pas au modele un degre de liberte pour compenser un defaut de donnee identifiable. AU PASSAGE, LE BANC A DEJA PAYE : il a intercepte mon re-echelle ETI tf=4 (qui detruisait l'hiver, 0.63 en decembre) AVANT le run complet qui etait en file, et designe le bon point tf=1.2 litterature + srf=6e-5 (0.84-1.02 sur tout l'hiver, avril 0.91) | 08-24 |
| R44 | TOPO ETI COMPLET (demande d'Essi avant de sceller la 1.0) : la fonte au bilan radiatif reel remplace-t-elle la sinusoide, et corrige-t-elle la Gaspesie ? | **TROIS VERDICTS, UN PAR QUESTION.** (1) L'ACCUMULATION SANS SINUSOIDE : OUI, parfaite -- OUTV a 1.02/0.97/1.04 de novembre a mars avec les coefficients litterature, la these du cycle radiatif tient (la sinusoide amp 0.5 approximait un cycle FB mesure d'amplitude relative 0.62). (2) LA FONTE DE PRINTEMPS aux coefficients litterature : NON, trop lente -- avril 1.30, mai 4.10, score 0.636 ; Pellicciotti vient de glaciers alpins, ~7 mm/j aux conditions de crue boreale quand le degre-jour cale en libere ~25, et deux scalaires ne remontent pas d'un facteur 3.5 en dix epoques. LE BANC NEIGE-SEULE A TROUVE LE POINT : tf litterature + srf 6e-5 donne 0.84-1.02 sur l'hiver et 0.91 en avril -- il a aussi INTERCEPTE mon re-echelle tf=4 qui detruisait l'hiver, avant le run complet qui l'attendait en file. L'entrainement complet a ce point reste a payer (1.5 h) pour clore. (3) LA GASPESIE : NON, l'ETI ne change RIEN au manteau (1.95/1.86/2.17 contre 1.91/1.79/2.02 au degre-jour, avril 4.21 par la fonte lente en plus, score 0.409). Par elimination complete -- seuil (R42), amplitude (R42), appariement (R42), et maintenant la formulation de fonte elle-meme -- **le FORCAGE COTIER est formellement mis en accusation** : biais froid des mailles ou precipitation solide en trop, a departager contre les stations GHCN cotieres. MA RECOMMANDATION INITIALE (degre-jour + sinusoide en 1.0, ETI en 1.1) a ete REJETEE par Essi, a raison : la sinusoide est la becquille, l'ETI la physique finie, et geler la becquille dans une 1.0 revient a livrer du half-baked. NOUVELLE SEQUENCE : run decisif eti3 au point du banc (tf litterature, srf 6e-5) en file derriere la phase quantile ; si CanSWE ~1.0 ET avril/mai repares ET score au niveau de gen1, la flotte des 12 regions se rejoue sous ETI et la 1.0 se scelle sur CES chiffres. Le dossier gaspesien passe au chantier FORCAGE dans tous les cas | 08-24 |
| O9 | NIVEAUX D'EAU (chantier 1.x, accord d'Essi 2026-08-24) : meandre peut-il donner une idee des niveaux, et la bathymetrie publique suffit-elle ? | OUVERT, plan par etapes arrete. (1) AUX JAUGES : le CEHQ publie les NIVEAUX a cote des debits (fichiers _N du meme depot que les drapeaux, chargeur existant a etendre) -> courbes h(Q) directes, et verite terrain des etapes suivantes. (2) PARTOUT : geometrie hydraulique de Leopold-Maddock (largeur et profondeur en lois de puissance du debit), coefficients et exposants en SORTIE DE CHAMP ancres litterature -- la meme grammaire que tout le projet : loi simple, champ appris, observation independante par processus. Donne des VARIATIONS de niveau et des profondeurs d'etiage aux prises d'eau (la question du mandat), pas d'absolu. (3) OBSERVATIONS INDEPENDANTES : SWOT (elevation de surface, largeur, pente, public, depuis 2023, rivieres > 50-100 m -- couvre la tenue de cote : jouerait pour les niveaux le role que GRACE a joue pour le stockage) ; LiDAR provincial 1 m (Foret ouverte : berges et plaine fines, surface d'eau au jour du vol -> inversion partielle de la geometrie immergee croisee a notre debit du meme jour) ; NONNA du SHC (eaux navigables) ; sections Info-Crue/MTQ (local). LIMITES ASSUMEES : sans bathymetrie sous le plus bas niveau observe, l'absolu reste inconnu hors ancrage SWOT/LiDAR ; la cote de crue centimetrique exige de l'hydraulique complete sur sections mesurees, par troncon cible seulement, jamais a la province | 08-24 |
| O10 | TEMPERATURE DE L'EAU (chantier 1.x, accord d'Essi 2026-08-24) : meandre peut-il porter un modele thermique des troncons ? | OUVERT, et le modele possede deja presque tous les ingredients. ATOUTS EN PLACE : le forcage porte le bilan d'energie complet (radiation reelle FB des caches sw_in, vent, pression de vapeur, T) ; le debit par troncon donne la capacite thermique et, via O9 (geometrie hydraulique), la PROFONDEUR donc l'inertie ; la fraction de DEBIT DE BASE par troncon -- fruit direct du chantier krec/k_gw -- pilote le tampon d'eau souterraine (~6-8 degres) qui stabilise les etes ; f_forest donne l'OMBRAGE ; les lacs sont identifies (rechauffement aval) ; et le graphe de routage porte l'ADVECTION de temperature d'amont en aval comme il porte l'eau. FORMES CANDIDATES, du simple au complet : air2stream (Toffolon-Piccolroaz, 4-8 parametres, la reference du rapport cout-performance) comme socle ; puis bilan d'energie par troncon dT/dt = echanges/(rho*c*profondeur) + advection + melange souterrain, avec coefficients d'echange, facteur d'ombrage et T de nappe en SORTIES DE CHAMP ancrees litterature -- la grammaire du projet. LIEN AU MANDAT, et il est fort : les REJETS sont souvent thermiques, et la temperature est LA variable ecologique des impacts de prelevement (habitat, etiage chaud) ; un module thermique transforme l'analyse des rejets en analyse d'impact reel. OBSERVATIONS INDEPENDANTES a inventorier : thermographes des reseaux de suivi provinciaux et des OBV/rivieres a saumon (Donnees Quebec, couverture a verifier), capteurs aux stations hydrometriques, thermique satellitaire (Landsat/ECOSTRESS, rivieres larges, temperature de peau). LIMITES ASSUMEES : periode de glace, stratification des reservoirs, et les rejets thermiques ponctuels demanderont leur propre inventaire de sources | 08-24 |
| O11 | PETITS BASSINS ET PLUIE RADAR (chantier 1.x, idee d'Essi 2026-08-24) : combler la DISCONTINUITE entre la methode rationnelle et Hydrotel par une approche integree multi-echelles, pluie a haute resolution spatio-temporelle a l'appui | OUVERT, et il prolonge un fil deja ouvert : la cause racine du deficit d'ete (journal 2026-06-30) etait le quickflow d'orage ECRASE par le pas journalier (DT_H=24 lamine l'intensite), et la colonne porte DEJA le canal storm_hours et la porte hortonienne opt-in -- il manque la DONNEE d'intensite. L'IDEE : sur les petits BV, la pratique bascule vers la methode rationnelle Q=C*i*A, discontinue avec le modele continu ; au lieu de deux methodes, UN continuum ou le coefficient de ruissellement EMERGE du champ appris et l'intensite vient d'une pluie resolue en sous-journalier -- le modele devant CONVERGER vers le comportement rationnel quand l'aire tend vers zero (test de coherence d'echelle a construire, avec la diffusion spatiale comme operateur de changement d'echelle explicite plutot que comme artefact). SOURCES DE PLUIE, du plus praticable au plus fin : CaSR est deja HORAIRE (exploite seulement en agregats journaliers) ; HRDPA/CaPA 2.5 km horaire informee radar (archive ~2011+) ; composites radar ECCC 1 km / 10 min (GeoMet, archive courte et hivers aveugles) ; IMERG 30 min en complement. LIMITES ASSUMEES : le radar ne voit pas la neige (chantier d'ete), bande brillante et etalonnage, archives courtes pour l'entrainement long, et la tenue de cote devrait gagner des crues ECLAIR observees -- stations a pas fin (CEHQ 15 min ?) a inventorier comme juge. PRECISION D'ESSI (meme jour) : meme horaire, CaSR est IMPRECIS en convectif -- orages manques, ou deplaces au mauvais endroit et au mauvais moment (le registre en portait deja l'indice : « risque = CaSR mal-date les orages », 06-30, et la correction de timing jour-local existait pour la meme raison a l'echelle journaliere). CaSR horaire n'est donc que l'ECHAFAUDAGE DE VOLUME ; le RADAR devient la reference corrective du convectif. ORDRE REVISE : (1) diagnostic d'evenements AVANT toute correction -- apparier N orages d'ete CaSR/composites radar sur un bassin jauge nerveux, chiffrer les erreurs de position et de chronologie (l'equivalent du travail fait pour le de-crachinage : mesurer le defaut avant de le corriger) ; (2) selon le diagnostic, corriger par appariement d'objets et recalage (advection/warping des cellules CaSR vers le radar), ou n'emprunter au radar que la STATISTIQUE d'intensite sous-journaliere (duree, concentration) si les deplacements sont trop grands pour un recalage deterministe -- le canal storm_hours n'a besoin que de la concentration, pas de la position exacte, SAUF sur les petits BV qui sont justement la cible : d'ou l'enjeu du recalage spatial ; (3) seulement ensuite, le chantier hortonien s'appuie sur l'intensite corrigee. DEUXIEME PRECISION D'ESSI : la DIFFUSION sert aussi a COMBLER LES MANQUES DES INSTRUMENTS eux-memes -- le radar a ses trous (blocage de faisceau, attenuation en distance, interstices entre radars, cecite hivernale). Deux lectures complementaires, a garder toutes deux : (a) diffusion physique/statistique = degradation GRACIEUSE de l'information, le champ s'etalant et s'incertifiant la ou l'instrument ne voit pas, plutot qu'un trou dur ; (b) diffusion GENERATIVE (modeles de score, la reference Song-Ermon deja notee au registre pour le prior NeRF) = reconstruction conditionnelle du champ de pluie coherente avec radar + CaSR + jauges la ou l'observation manque, dont l'echantillonnage produit un ENSEMBLE de realisations d'orages -- le frere convectif des membres PyGMET, qui se brancherait tel quel sur la machinerie de propagation de membres construite le 08-24. L'incertitude de position d'un orage deviendrait alors une dimension EXPLICITE de la chaine probabiliste au lieu d'une erreur muette | 08-24 |
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

### O16. GRACE mesure aussi les reservoirs de barrage, et SAGU/OUTV en sont pleins

Remarque d'Essi (2026-08-27) : « peut etre l'influence des barrages ? Outaouais et Saguenay ont pas mal de petits barrages ». Elle porte sur DEUX enquetes distinctes, et ne s'applique qu'a une seule.

POUR GRACE, ELLE PORTE ET N'AVAIT PAS ETE VUE. GRACE mesure une masse d'eau TOTALE sur son empreinte, sans distinguer sol, neige, nappe ou reservoir de barrage. SAGU compte 348 noeuds lacustres (87 467 km2 draines au plus grand) et OUTV 514 (83 198 km2) : des reservoirs de tete importants, geres de facon coordonnee (O13, DZTR). Si cette gestion retient l'eau plus tard au printemps, une partie de la signature attribuee au SOL, a la NEIGE ou a la NAPPE du modele dans R47-R49 appartient en realite a une gestion de barrage que le modele ne represente pas du tout. TOUTE L'ANALYSE GRACE DES TROIS DERNIERS JOURS A PU ETRE PARTIELLEMENT CONTAMINEE sur ces deux bassins precisement.

Consequence pratique : privilegier GASP et MONT pour tout futur test de stockage contre GRACE, ce sont les deux bassins les moins regules du domaine bien echantillonne. Refaire le diagnostic de stockage (celui qui a produit R47-R49) sur GASP et MONT seuls serait le controle le plus direct.

POUR LE DEFICIT DE MANTEAU MESURE PAR CanSWE, ELLE NE PORTE PAS. CanSWE est une mesure PONCTUELLE au sol, a un site fixe, de la neige accumulee sur le terrain. Le deficit de 22 % constate a SAGU des decembre, avant toute fonte, est mesure a des sites d'observation sans rapport avec un reservoir hydroelectrique. Un barrage ne change rien a la neige qui tombe sur un point de mesure au sol. L'enquete sur le forcage (precipitation solide sous-estimee par CaSR, ou seuil de partage pluie-neige mal calibre) reste entiere et independante de cette remarque.

### O17. Le deficit de decembre correle a l'occupation du sol (agriculture, wetland), pas au site-noeud

En cherchant la cause du deficit de manteau de SAGU (R50), deux controles supplementaires le 2026-08-27.

REFUTE : l'appariement site-noeud. Correlation du ratio hivernal (sim/CanSWE) avec la distance site-noeud (-0.017) et l'ecart d'altitude site-noeud (-0.086), sur 37 stations bien echantillonnees : essentiellement nulles. Le deficit n'est pas un artefact de mauvais appariement geographique.

PISTE OUVERTE, A CONFIRMER : correlation avec l'occupation du sol au noeud, sur 31 noeuds (n modeste, a interpreter avec prudence). Agriculture -0.421, milieu humide -0.292, conifere +0.198, eau libre +0.216. Les noeuds plus agricoles ou plus humides ont un deficit de decembre PLUS GRAND ; les noeuds plus coniferiens un deficit plus PETIT.

REFUTE comme explication : le terme geothermique (taux_fonte_geo) est un SCALAIRE UNIFORME (0.5 partout, verifie), il ne peut pas porter cette variation spatiale.

HYPOTHESE PHYSIQUE PLAUSIBLE, NON TESTEE : les coefficients de fonte degre-jour par classe (taux_c/taux_f/taux_d) sont plus eleves pour la classe DECOUVERT, qui inclut l'agriculture, que pour le conifere. Des redoux hivernaux ponctuels (temperatures positives en decembre-fevrier, frequents mais courts) feraient donc fondre PLUS en terrain agricole qu'en foret, meme avant le printemps -- ce qui est PHYSIQUEMENT ATTENDU (exposition au vent et au soleil) et pas necessairement un defaut du modele. A verifier : la frequence et l'amplitude des redoux hivernaux dans le forcage CaSR sur SAGU, et si le coefficient DECOUVERT est cale trop haut pour ces evenements courts specifiquement.

TEST QUI TRANCHE, non fait ce soir : refaire cette correlation sur OUTV (accumulation correcte, 0.95) et sur GASP. Si la correlation agriculture/deficit est specifique a SAGU, chercher un defaut LOCAL (forcage ou occupation mal calee sur ce territoire) plutot qu'un mecanisme general.

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

### O12. L'agriculture est la variable qui explique le mieux l'échec territorial, et il manque un PROCESSUS, pas une variable

Mesuré le 2026-08-26 sur la première tenue de côté provinciale (champ unique, 3 epochs, 141 stations, médiane 0.6193). Sur les six territoires bien échantillonnés, la relation entre fraction agricole et score est presque monotone : MONT 33.7 % pour 0.4821, SLSO 29.2 % pour 0.5380, SAGU 14.1 % pour 0.6737, GASP 11.0 % pour 0.7361, SLNO 10.8 % pour 0.6999, OUTV 9.2 % pour 0.6519. VAUD, le plus agricole à 56 %, n'a aucune jauge et reste invisible.

Ce n'est PAS un manque de variable : `f_agriculture` fait partie des seize attributs fournis au NeRF, qui peut donc moduler ses paramètres dessus. C'est un manque de processus, et un paramètre ne compense un processus absent que dans son bassin de calage. Trois candidats, par ordre d'effet attendu.

Le DRAINAGE SOUTERRAIN d'abord. La colonne n'a aucun chemin qui sorte de la couche 2 vers le tronçon par un seuil de profondeur, or c'est exactement ce qu'est un drain agricole posé à un mètre avec dix à vingt mètres d'espacement. Il convertit un interflux lent en chemin rapide à seuil : crue printanière plus haute et plus précoce, tarissement caractéristique, étiage estival plus bas. Formulation utilisable : Hooghoudt, continue et dérivable, deux paramètres réellement nouveaux (espacement, profondeur de pose), la conductivité étant déjà au champ. DRAINMOD est la référence complète mais n'est ni léger ni différentiable.

La PHÉNOLOGIE DES CULTURES ensuite, et elle est presque gratuite : l'occupation agricole tombe aujourd'hui dans la classe découvert, c'est-à-dire du sol nu toute l'année, sans levée ni croissance racinaire ni coefficient cultural. Le dépôt porte déjà un modulateur phénologique piloté par les degrés-jours pour les classes forestières ; y ajouter une classe cultivée avec les quatre phases FAO-56 est une extension. Elle est jugeable immédiatement puisque l'anomalie d'ET MODIS est déjà dans la perte.

L'IRRIGATION enfin, sous-estimée : les prélèvements agissent sur le débit du tronçon et sur la nappe, jamais sur le sol. L'eau d'irrigation est pourtant prélevée puis APPLIQUÉE sur le champ, d'où elle s'évapore ou percole. La représenter comme une simple soustraction au cours d'eau perd la moitié du cycle, et c'est la moitié qui compte pour le bilan estival.

CE QUI TRANCHE : ajouter un processus à la fois, avec son juge propre, en commençant par le drainage souterrain sur la Montérégie seule, où le signal est le plus fort. Si l'écart de 0.21 face au champion régional se referme, c'est le processus ; sinon c'est le calage ou la donnée.

REMARQUE DE MÉTHODE : c'est le champ UNIQUE qui a rendu ce défaut visible. Les calages régionaux le masquaient, chacun ajustant ses propres paramètres pour absorber localement un processus absent. Le modèle provincial sert donc d'instrument de diagnostic, ce qui est l'argument d'identifiabilité du projet mis en pratique.

### O13. Barrages : le modèle DZTR et le répertoire du CEHQ (1.x)

Piste posée par Essi le 2026-08-27. Les ouvrages de retenue sont aujourd'hui absents du modèle, alors que le Québec méridional en compte des milliers et que leur gestion déplace de l'eau dans le temps exactement là où le modèle se trompe déjà, au printemps. Deux ressources à mobiliser.

Le modèle DZTR (Dam Zoning and Target Release) donne une règle de lâcher générique à partir de zones de remplissage et d'une cible saisonnière, sans exiger les consignes réelles de chaque exploitant, qui ne sont ni publiques ni homogènes. C'est ce qui le rend applicable à l'échelle provinciale, et sa forme par zones est différentiable par morceaux. Référence : https://www.sciencedirect.com/org/science/article/pii/S031514682200044X

Le répertoire des barrages du CEHQ fournit la localisation, la hauteur, le volume de retenue et l'usage, donc de quoi apparier un ouvrage à un tronçon et l'initialiser. https://www.cehq.gouv.qc.ca/barrages/default.asp

À NE PAS CONFONDRE avec les pseudo-lacs, artefact déjà connu : des retenues importées d'Hydrotel comme réservoirs actifs y produisaient des retards artificiels. Ici il s'agit de représenter une GESTION, pas une géométrie. Et à distinguer aussi des prélèvements et rejets, qui retirent ou ajoutent de l'eau, alors qu'un barrage la déplace dans le temps sans changer le volume annuel.

Le juge doit être choisi avec soin : le débit aux stations en aval mélangerait l'effet du barrage et celui de l'hydrologie amont. Les niveaux de retenue, quand ils sont publiés, seraient l'observable propre, ce qui rejoint O9 sur les niveaux d'eau.

### O14. Laminage de la Richelieu et de l'Outaouais : deux tâches déjà ouvertes, corroborées de l'extérieur

Signalé le 2026-08-27 par un hydrologue d'expérience, via Essi : le laminage est mal modélisé sur le bassin de la Richelieu et sur celui de l'Outaouais. La remarque ne demande pas une tâche nouvelle, elle confirme deux tâches existantes et dit laquelle s'applique où.

VÉRIFICATION FAITE, ET ELLE ÉCARTE UNE FAUSSE PISTE. Le lac Champlain EST dans le domaine : nœud de lac à 44.285 N et -73.406 O, donc au Vermont, 21 283 km² drainés, trente-six arêtes entrantes. Le domaine MONT franchit la frontière et la somme de ses aires locales (38 937 km²) égale son aire cumulée maximale (38 935). Il n'y a donc PAS d'apport transfrontalier manquant, contrairement à ce qu'on pourrait supposer.

RICHELIEU : tout le tamponnement du lac Champlain repose sur UN nœud de lac avec un couple de paramètres appris par le champ. Or son atténuation est gouvernée par le haut-fond de Saint-Jean, donc par une géométrie d'exutoire particulière et fortement non linéaire. Un seul jeu de paramètres appris SUR LE DÉBIT peut difficilement la reproduire, d'autant que le débit n'est plus le terme dominant de la perte (R47). C'est la tâche 40, ancrage d'exutoire et tête apprenante des lacs.

OUTAOUAIS : 514 lacs, le plus grand drainant 83 198 km², donc les réservoirs de tête sont bien présents en tant que plans d'eau. Ce qui manque est leur GESTION. Dozois, Cabonga et Baskatong sont exploités de façon coordonnée pour écrêter les crues, et cette atténuation est une décision, pas une propriété du bassin. C'est O13, le modèle DZTR.

CONSÉQUENCE POUR LE DIAGNOSTIC EN COURS : MONT porte désormais DEUX explications concurrentes pour son mauvais score (0.4821 contre 0.6953 pour son champion régional), l'agriculture (O12) et le laminage de Champlain. Elles ne sont pas exclusives et il faudra les départager, sans quoi on attribuera à l'une ce qui revient à l'autre. Le départage est possible : le drainage agricole agit sur les tronçons à forte fraction cultivée, le laminage sur le seul axe du Richelieu en aval du lac.

### R48. Le temps de résidence de l'aquifère n'est PAS le mécanisme de la descente printanière

Balayage en inférence du 2026-08-27, juge = climatologie GRACE, aucun entraînement. Une nappe trois fois plus lente laisse les rapports d'amplitude quasi inchangés (GASP 1.44 vers 1.42, MONT 0.77 vers 0.73) ; dix fois plus lente ne fait pas mieux et DÉGRADE les corrélations (OUTV 0.834 vers 0.805, MONT 0.890 vers 0.845). Deux points, tous deux négatifs.

L'hypothèse était pourtant motivée : le champ k_gw provincial a une médiane de 0.0856 par jour, soit douze jours de temps de résidence, quand GRACE demande une rétention d'un à deux mois, et il a été estimé sur des récessions de COURS D'EAU, donc sur le compartiment le plus rapide. Elle est réfutée, ce qui renforce d'autant la piste du gel.

### O15. Cartes de sol : PHYSITEL non documenté, IRDA en réserve, et le piège des pédotransferts

Posé par Essi le 2026-08-27, en réponse à ma proposition d'ancrer les propriétés thermiques du gel sur des fonctions de pédotransfert. Deux avertissements, tous deux fondés, qui invalident cet ancrage tel que je le proposais.

Les fractions de sable, limon et argile viennent de PHYSITEL et sont issues d'un calage NON DOCUMENTÉ : rien ne garantit qu'elles décrivent correctement le sol. Et les fonctions de pédotransfert thermiques sont reconnues comme très approximatives, avec des incertitudes de l'ordre de cinquante pour cent. Ancrer sur le produit des deux contraindrait fortement le champ vers une valeur dont on ignore la justesse, ce qui est PIRE qu'un paramètre libre : on remplacerait une liberté visible par une erreur invisible.

L'IRDA dispose de cartes mieux documentées. À NE PAS substituer tout de suite, mais la divergence éventuelle entre les deux jeux est en soi une mesure : si elles s'accordent, la texture de PHYSITEL est utilisable ; si elles divergent, on saura que tout ce qui en dépend (dont les corrélations texture-paramètres, dont la lecture ci-dessous) reposait sur du sable.

CE QUE CELA CHANGE À L'INTERPRÉTATION DE R49. J'avais lu l'inversion des corrélations comme une pathologie du modèle. Elle admet une seconde lecture, symétrique : si la carte de texture est mal étiquetée, le champ apprend la BONNE physique sur des étiquettes fausses, et le signe apparaît inversé sans que le modèle ait tort. Le champ deviendrait alors un instrument qui signale une entrée suspecte, ce qui est l'argument d'identifiabilité du projet plutôt que son échec.

LE TEST QUI DÉPARTAGE NE DÉPEND NI DE PHYSITEL NI DES PÉDOTRANSFERTS. La conductivité thermique et la capacité calorifique volumique croissent TOUTES DEUX avec la teneur en eau, l'eau étant à la fois bien plus conductrice que l'air et bien plus capacitive. D'un tronçon à l'autre, elles doivent donc être POSITIVEMENT corrélées ENTRE ELLES, quelles que soient les étiquettes de texture. Une anti-corrélation forte est thermodynamiquement impossible et prouve un levier utilisé hors de son sens ; une corrélation positive laisse ouverte l'hypothèse de la carte fautive.

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
13. **Cinq colonnes dont le nom ment sur le contenu** : `mean_elevation_m`, `f_forest`, `territorial.drainage_area_km2` et `area_km2_physical` sont NORMALISEES dans la table brute et doivent passer par `get_physical()` ; `drainage_area_km2` du parquet territorial est, elle, une quatrieme quantite qui n'est ni locale ni cumulee. Chacune a produit une conclusion fausse avant d'etre reperee (un biais d'altitude fictif de +186 a +412 m, des quartiles de foret illisibles, et un deficit d'aire drainee de 75 % qui n'existe pas). Regle : aucune colonne d'aire ou d'altitude ne se lit directement dans la table brute. A regler en renommant a la source lors de la reconstruction des caches, avec un test qui refuse une colonne d'aire dont la somme s'ecarte du domaine connu.
14. **La contrainte NEIGE n'a jamais existe dans le pilote quebecois** (trouve le 08-21). `with_forcing` (`.runs/quebec/etl_run.py` l.152) reconstruit `TrainingData` en recopiant `et_obs` et `tws_obs` mais **PAS `swe_obs`**, dont le defaut du dataclass est `None` ; or `_need_snow = (w_snow > 0 and data.swe_obs is not None)` (trainer.py l.924). Le pilote imprime `[etl] w_snow = 0.3 (fonte supervisee MOD10)` a CHAQUE run et la perte neige ne s'evalue jamais. Introduit le **2026-07-21** par le commit 1083662, celui-la meme qui ajoutait la supervision MOD10 phase 2 : la fonctionnalite et sa desactivation silencieuse sont arrivees ensemble, et 219 commits ont suivi. `.runs/slso/slso.py` passe `swe_obs` correctement, donc SLSO n'est pas touche. NON CORRIGE a dessein : reparer changerait la recette de tous les runs quebecois, y compris le champion, ce qui est une decision d'Essi et pas un correctif de diagnostic. Meme famille que la dette #3 (entrees a repli silencieux) : la regle a etendre est qu'un poids de perte strictement positif dont le terme ne s'evalue jamais doit LEVER une erreur, pas se taire.
15. **Un journal qui annonce autre chose que ce qui tourne** (trouve le 08-21, sur une question d'Essi). `etl_run.py` imprimait `etp_channel=6 (demande apprise x K_c NeRF)` EN DUR, y compris sous `ETL_ETP=linacre|mcguinness` qui posent `etp_channel = None`. Tous les journaux du chantier quebecois affirmaient donc que la demande evaporative MLP etait active alors que la colonne l'ignorait, et j'ai moi-meme construit un raisonnement faux dessus (voir R24). Corrige : la ligne lit l'attribut. Meme famille que les dettes #2, #3 et #14 -- la regle commune est qu'aucune sortie ne doit affirmer un reglage sans l'avoir LU dans l'objet qui tourne.
16. **`patience` n'a jamais atteint le pilote quebecois** (trouve le 08-21 en lisant `no_improve=1/0` dans un journal de run). `etl_run.py` construit `TrainingConfig` en ENUMERANT les cles qu'il reprend du TOML, et `patience` n'y figure pas : elle reste au defaut **0**, donc `if self.config.patience > 0 and ...` (trainer.py l.762) ne se declenche JAMAIS et l'arret anticipe n'existe pas. Les 53 configs quebecoises portent pourtant `patience = 8` depuis ce matin, avec un commentaire de dix lignes justifiant la valeur sur 37 runs -- une economie de 28 % de calcul annoncee a Essi qui n'a jamais eu lieu. `.runs/slso/slso.py` la passe correctement (l.928, `tcfg["patience"]`), donc les 74 configs SLSO sont indemnes. NON CORRIGE VOLONTAIREMENT pendant la file aux-A / aux-B : ajouter la cle maintenant ferait tourner B sous un protocole different de A et ruinerait l'ablation. A corriger des la file terminee. TROISIEME instance du meme motif en une journee (dettes #14 et #15) : la regle generale est qu'une CONSTRUCTION PAR ENUMERATION -- de dataclass, de config, de message -- perd en silence tout ce qu'on ajoute ailleurs. Preferer `dataclasses.replace`, un passage en bloc du dict, ou a defaut un test qui compare les cles du TOML aux champs consommes.
17. **Trois erreurs d'agregation de suite sur la MEME question** (08-21/22). Le pic de manteau mesure a ete calcule trois fois de trois facons non comparables : maximum de la mediane journaliere du reseau (238 mm, artefact d'echantillonnage) ; mediane des pics par site (108 mm, correct en soi mais compare a un pic MOYEN-BASSIN simule, donc inutilisable) ; et enfin l'appariement site-et-jour, seul valide, qui donne **0.65-0.72** -- **et qui etait IMPRIME dans chaque journal depuis l'ajout d'ETL_CMP_NEIGE**. Chacune a produit une conclusion differente, ecrite au registre, puis defaite par la suivante : « le modele a la moitie de la neige », puis « il en a plus que la mesure », puis « il en a 70 % ». REGLE A APPLIQUER : avant de comparer un simule a un observe, ECRIRE explicitement l'unite d'agregation des DEUX cotes (par site ou par bassin ; par jour ou par saison) et refuser la comparaison si elles different. Et CHERCHER D'ABORD si la mesure existe deja dans une sortie du pilote : ici la bonne reponse etait a l'ecran, sous une ligne jamais lue.
18. **Puits virtuels de raccordement dans les topologies PHYSITEL** (trouves par Essi SUR la carte feuillage, 2026-08-24 -- premiere paie de l'outil). Les regions cotieres sont des collections de bassins independants que PHYSITEL raccorde a un exutoire ARTIFICIEL unique pour faire un arbre : gasp noeud 1 recoit 167 aretes de 127 km medians (280 max), cndc noeud 1 en recoit 18 ; marques is_lake, avant-derniers de l'ordre topo, aucune station dessus. AUCUN impact sur les resultats (routage aval-seulement, rien d'evalue sur ou sous les puits, aires aux stations verifiees a 1.005x par R18). Impacts reels : un parametre de LAC APPRIS gaspille sur chaque puits fictif (cousin des pseudo-lacs deja consignes), les agregats provinciaux par troncon doivent les exclure, et la carte les rendait en eventails d'aiguilles (filtre 35 km pose dans build_feuillage, qui GARDE le reservoir Baskatong a 20-30 km d'aretes reelles). A distinguer au prochain rebuild des caches : un drapeau is_virtual_sink dans la table nodes, plutot qu'un seuil de longueur.
19. **Un entrainement peut se BLOQUER sans mourir ni diverger visiblement** (eti3, 2026-08-24). L'epoque 16 a dure ~4 h contre 12 min en regime, processus vivant et carte a 33 % : ni plantage, ni NaN, ni message. Le diagnostic n'a ete possible QUE par la ligne des composantes ponderees ajoutee la veille (prior=185, tws_clim=84.5, tws=31.3 contre un total de 2.88 -- le champ spatial en fuite, vraisemblablement krec, avec le garde-fou de divergence boucle en rechargements). Sans cette sortie, on aurait relu des heures d'epoques identiques sans comprendre. CE QUI MANQUE ENCORE, a ajouter : un CHIEN DE GARDE de duree d'epoque (si une epoque depasse ~3x la mediane des precedentes, arreter et rendre le meilleur point de reprise), et un plafond sur les composantes de perte (si une composante ponderee depasse ~20x le total, arreter -- c'est une divergence de parametre, pas un apprentissage). Le meilleur point de reprise etait sauf : rien de mesurable n'a ete perdu, seulement 4 h de carte et la cascade de nuit bloquee derriere.
20. **Un run TERMINE peut ne pas SORTIR et garder la carte** (2026-08-24/25). Mon evaluation du point de reprise eti3 a fini d'ecrire son journal a 21h32 puis n'est jamais sortie : processus vivant onze heures, en concurrence avec la cascade de nuit. Effet mesure : les evaluations de membres sont passees de ~7 min a ~2h15 -- trois membres en huit heures au lieu de dix, et les onze regions jamais commencees. C'est la dette #7 sous une forme NOUVELLE : le probleme n'est plus de detecter les processus (le compteur voyait bien deux runs) mais qu'un run FINI n'est pas mort ; l'attente-par-journal que j'avais adoptee pour contourner le compteur ne protege pas de ca, un DONE ecrit ne garantissant pas la liberation de la carte. CORRECTIFS A FAIRE : (a) trouver ce qui retient le processus apres le dernier print (thread de chargeur de donnees non ferme ? handle CUDA ?) et fermer proprement ; (b) en attendant, terminer etl_run.py par un os._exit(0) explicite apres le DONE ; (c) que les files verifient la LIBERATION de la carte (nvidia-smi ou compteur) et pas seulement le marqueur de fin.

21. **Le pilote PROVINCIAL ne posait aucune graine** (trouvé le 2026-08-28, en comparant deux runs qui auraient dû être identiques). `etl_run.py` pose une graine depuis le 20 août, avec le commentaire qui explique pourquoi ; `province.py`, écrit plus tard, n'en posait aucune. PREUVE : deux runs de la même recette, même chaîne de configuration imprimée et mêmes 100 323 paramètres, donnent kge_med **0.3481 et 0.3566** à l'époque 0, puis **0.4378 et 0.2618** à l'époque 1. Les poids du champ étaient tirés au hasard à chaque lancement. CONSÉQUENCE : aucune comparaison provinciale dite appariée d'avant cette date ne l'était réellement, y compris le verdict du verrou de canopée (0.4357 contre 0.3566, écart plus petit que celui mesuré entre deux runs IDENTIQUES) et le banc local sur SAGU qui passe par le même pilote. Ce qui survit : tout ce qui lit un point de reprise sans entraîner, donc l'étape 0 du dossier barrages. CE QUI A RETARDÉ LA DÉCOUVERTE : un `grep` lancé sur deux fichiers dans la même commande, dont j'ai attribué la sortie au mauvais. Réglé, `ETL_SEED` défaut 1234 appliqué à random, numpy, torch et CUDA, et la graine est IMPRIMÉE au démarrage pour qu'un journal sans elle se repère à l'oeil.

22. **Une sortie destinée à un programme ne doit pas partager sa formulation avec de la prose** (2026-08-28). La file de fin de semaine lisait le score provincial en cherchant `mediane provinciale` dans le journal. Or le chargeur du domaine imprime lui aussi `pose a la mediane provinciale 0.6 (HYPOTHESE)` quand un champ manque. La fonction prenait la première occurrence et rendait donc **0.6**, une valeur de remplissage de milieu humide, au lieu de 0.4203. Toutes les décisions autonomes du week-end s'y seraient prises, et le résultat aurait eu l'air normal. Réglé par un marqueur `PROVMED` qui n'appartient qu'au pilote.

23. **Le garde-fou de divergence ne voit pas les dérives lentes** (2026-08-28). Il déclenche sur une perte à trois fois la moyenne mobile. Le banc provincial a dégradé TOUTES ses métriques de façon monotone sur cinq époques, perte d'entraînement de 4.57 à 7.05, sans qu'aucun mécanisme ne s'en aperçoive. Il faut une détection de TENDANCE, pas seulement de saut. Non corrigé : changer un garde-fou au milieu d'un banc apparié l'invaliderait.

24. **Les prédicteurs compositionnels entrent en pourcentages bruts** (trouvé par Essi le 2026-08-31). Les fractions granulométriques f_sand, f_silt, f_clay somment à 1 : après normalisation, les trois colonnes que reçoit le champ ont un RANG DE 2 (mesuré sur OUTV), un triplet exactement dégénéré. Même situation pour l'occupation du territoire (cinq parts plus un complément implicite jamais fermé). Conséquences : capacité gaspillée, et la sensibilité PAR fraction d'E14 (sable 0.471, limon 0.399) n'est pas bien définie, on ne peut pas varier une part en tenant les autres fixes sur un simplexe — la conclusion qualitative d'E14 (le champ s'appuie sur la texture, pas sur les coordonnées) survit, ses chiffres par variable non. Remède écrit et testé, NON branché : `meandre/spatial/compositions.py`, log-ratios isométriques avec remplacement multiplicatif des zéros (63 tronçons sans sable sur OUTV, urbain nul presque partout). PORTÉE : les compositions comme PRÉDICTEURS seulement ; là où une fraction pondère la physique (classes de fonte, milieu humide), elle reste une fraction. Décision d'Essi : chantier 1.1, banc apparié requis avant adoption puisque N_FEATURES change et que les points de reprise cassent.

25. **Le niveau d'évapotranspiration est PRESCRIT par la correction du forçage, pas identifié** (trouvé par Essi le 2026-08-31, en questionnant Oudin dans le rapport). La correction Budyko cale le volume de précipitation sur lame observée + ET(Budyko-Fu, PET Oudin). Par conservation de la masse, un modèle entraîné au débit sur cette précipitation converge vers CE niveau d'ET : l'entraînement tire les coefficients de Linacre jusqu'à rendre le volume injecté. MODIS étant en mode anomalie (niveau exclu, décision R24), RIEN ne contraint le niveau d'ET indépendamment. CONSÉQUENCE SUR LE CLAIM : l'identifiabilité du partage vertical est bornée à la SAISONNALITÉ et au CONTRASTE SPATIAL de l'ET, jamais à son niveau annuel, et tout texte (rapport, articles) doit le dire. Ironie documentée : l'hybride condamné (dette forçage) avait une précipitation de stations SANS hypothèse d'ET ; le tout-CaSR-Budyko est cohérent entre variables mais son P embarque Budyko. Aucun forçage existant n'est propre sur les deux axes. REMÈDE ACTÉ (décision d'Essi) : v1.0.1 livrée avec la limite documentée ; chantier 1.1 = reprendre CaSR BRUT (tuiles locales à inventorier), décrachinage et agrégation jour-local conservés (corrections mécaniques sans hypothèse), puis correction de P par KRIGEAGE DES RÉSIDUS sur les stations ouvertes — corriger la précipitation avec des observations de précipitation, aucune hypothèse d'ET dans la chaîne des données.

## R57 — La migration vers une grappe de l'Alliance est validée (2026-09-01)

**Statut : établi.** Le même modèle (`slso-etl-canon`, recette du socle, forçage hybride, inférence pure) rend un KGE médian par station de **0,5737 sur 29 stations** pour 2022-2024, à l'identique sur la machine locale et sur Narval, deux exécutions indépendantes sur la grappe. La chaîne complète se charge : occupation du sol, milieux humides, surfaces de lac corrigées par HydroLAKES, noyau de versant, fonte régionale ancrée, phénologie, courbe de rétention par nœud.

**Ce qu'il a fallu.** Quatre paquets viennent de modules et jamais de pip (`arrow` pour pyarrow, `mpi4py` sans lequel netCDF4 refuse de s'importer), le module devant être chargé AVANT l'activation de l'environnement. Triton est absent, donc `torch.compile` du sol échoue : `ETL_COMPILE=0`. La variable des plateformes s'écrivait `MEANDRE_PLATEFORMES` en français dans le code et `MEANDRE_PLATFORMS` dans mes scripts, et le repli silencieux vers le chemin du poste laissait la tâche tourner vingt minutes avant d'échouer ; les deux orthographes sont désormais lues.

**Vitesse.** Une simulation SLSO complète (24 ans, 2900 tronçons) prend **6 min 13 sur A100 avec la compilation du sol**, contre 27 min sans elle et 6 min en local : la colonne est dominée par le lancement de petits noyaux, pas par le calcul, et `torch.compile` y vaut un facteur quatre et demi. Le KGE est identique dans les deux cas (0,5737), donc la compilation ne change que la vitesse. Extrapolation : simulation provinciale des quinze régions en tableau, six à dix minutes de temps réel ; entraînement de trente époques, environ quatre heures au lieu de vingt en séquence.

**Portée.** Les quinze régions sont indépendantes, donc un tableau de tâches les entraîne en parallèle : la nuit de vingt heures en local devient l'attente d'une seule région. Coût nul. Procédure et scripts dans `.runs/quebec/alliance/`.

## R58 — L'Outaouais amont n'échouait pas, son réseau était inversé (2026-09-02)

**Statut : établi et corrigé.** Chaque ligne de `troncon.trl` finit par l'identifiant du tronçon aval, et le repli de `_build_graph` l'utilisait sans contrôle. Sur OUTM, la ligne du tronçon 1 (l'exutoire du domaine, 463 m de large, 62 996 km²) finit par 138, un ruisseau de 6,5 m : l'exutoire a été branché dessus. Ce lien fermait un cycle, le briseur de cycles a coupé la sortie du lac 96, et les 2379 tronçons se sont enracinés sur un lac de 2 079 km².

**Ce que ça produisait.** La station 041902 recevait 40 % de son bassin : rapport de volume 0,51 avec une corrélation de 0,91, et un débit spécifique de 6,5 L/s/km² contre 14 à 18 pour ses voisines. La région valait 0,448 contre 0,770 pour l'ensemble Hydrotel, et j'ai successivement soupçonné les barrages, le donneur, la météo et les paramètres de sol avant de regarder la topologie.

**Après correction** : 041902 passe de 0,448 à 0,639 et son débit spécifique à 15,0 L/s/km², la région passe de 0,448 à 0,639, l'écart à Hydrotel tombe de 0,322 à 0,131.

**Invariant à vérifier désormais sur tout domaine construit** : la racine de l'arbre doit être le tronçon de plus grande aire drainée. Mesuré sur dix régions, OUTM était la seule à 0,03 ; toutes les autres à 1,00.

## R59 — Sept sorties du champ spatial ne sont lues par personne (2026-09-02)

**Statut : établi, correction à faire.** Sur les 42 champs produits par le réseau spatial, un audit de gradient (`.runs/quebec/audit_gradients.py`) montre que `manning_n`, `interception_capacity`, `rain_hours`, `frost_alpha`, `alpha_T`, `f_wetland` et le champ de canopée n'apparaissent dans aucune équation de la physique : gradient exactement nul, aucun usage dans `hydrotel_column`, `routing` ou `model`. Le réseau consacre de la capacité à prédire des quantités que rien ne consomme, et leurs termes de prior tirent sur le tronc partagé.

**À ne pas confondre avec l'imposition légitime.** Les 23 champs imposés par la recette du socle sont bien des paramètres de sol (b, psis, nn, mm, omegpi, fsa/fse/fsi, cin, slope, z1-z3) ; leur gradient nul est voulu. `K_atm` et `alpha_T` sont morts parce que `use_temperature=False`, ce qui est un choix, pas un défaut.

**Conséquence mesurée.** Seuls K_sat (44 % de variation relative), k_gw (10 %) et T_melt (6 %) varient d'un tronçon à l'autre ; les 35 autres champs sont des constantes. Le champ de 42 paramètres se comporte comme un champ de trois, ce qui explique que l'entraînement apporte peu et que le transfert dépende autant du donneur.

## R60 — La surface d'un lac était lue comme une longueur d'écoulement (2026-09-03)

**Statut : établi, corrigé.** Dans `troncon.trl`, le bloc de quatre nombres d'un tronçon de lac est (surface en mètres carrés, profondeur moyenne, coefficient de tarage, exposant), et non (longueur, largeur, pente, quatrième valeur). Deux preuves indépendantes : le quatrième nombre vaut 1,500 pour tous les lacs des six régions inspectées, soit l'exposant de la loi de déversoir ; et la racine carrée du premier vaut 661 à 810 mètres en médiane, la dimension linéaire d'un petit lac. `_parse_troncon` lisait le premier comme `length_m` et le second comme `width_m`.

**Ampleur.** Longueur médiane des tronçons de lac : 437 à 656 km selon la région, maximum 8831 km, contre 3 à 5 km pour les rivières dont le maximum est 82 km. Le défaut est proportionnel à la densité de lacs, donc maximal là où le modèle s'effondre (outm 19 % de lacs, cnde 24 %).

**Ce que ce défaut n'a PAS fait.** Il n'a pas faussé la version 1.0. La topologie reconstruite est rigoureusement identique avant et après correction sur outv, outm, gasp et slso : zéro lien ajouté, zéro retiré. Le temps de parcours en jours, qui atteignait 102 jours sur 14 % des liens d'outv contre 1 jour après correction, n'est consommé que par l'attention temporelle, désactivée (`use_travel_time_attn=False`).

**Ce que ce défaut a fait.** Il a invalidé le banc du Muskingum physique du 2026-09-02, qui calculait le temps de parcours à partir de cet attribut et obtenait 167 heures par lac. Le verdict « l'échelle physique perd sur quatre régions sur six, dont −0,19 en Gaspésie » est donc **NUL** et ne doit plus être cité. Il faisait aussi comparer, au garde-fou de largeur de R58, une largeur de rivière à une profondeur de lac ; le garde ne compare désormais que deux rivières.

## R61 — Les modèles retenus déploient quatre à cinq paramètres figés sans le dire (2026-09-03)

**Statut : établi, cause de la « collapse » du champ.** Le champ compte aujourd'hui 42 sorties. Les modèles retenus en portent 37 ou 38 : ils sont antérieurs à l'ajout des dernières. Au chargement, `HydroModel.load` complète la différence par des ZÉROS, et une ligne nulle rend, après contrainte, le MILIEU des bornes, la même valeur sur tous les tronçons. Le paramètre devient une constante non apprise, et rien ne le signalait avant le 2026-09-03.

| modèle retenu | sorties | figées |
|---|---|---|
| outv-etl-canon, gasp-etl-socle30, sagu-etl-socle30, slno-etl-socle30 | 37 | krec, diff_gel, fs_neige, dT_canopee_feu, dT_canopee_conif |
| mont-etl-gen1 | 38 | diff_gel, fs_neige, dT_canopee_feu, dT_canopee_conif |
| slso-etl-casr | 42 | aucune |

**Contradiction relevée.** La recette du socle pose `ETL_KREC_LIBRE=1` et le journal annonce à chaque exécution que la récession de l'aquifère est apprise par le champ. Sur le champion de l'Outaouais elle vaut 2,0×10⁻⁵ sur les 3412 tronçons, étalement nul.

**Vérification par le témoin.** Le seul modèle complet, slso-etl-casr, apprend bien ces paramètres : krec 12,0 % d'étalement entre déciles, diff_gel 6,5 %, fs_neige 8,5 %, et une seule constante déguisée sur 19 sorties vivantes contre trois pour outv. Le mécanisme est donc la PÉREMPTION du point de reprise, pas un défaut de la physique ni de l'optimiseur.

**Correction.** Le rembourrage nomme désormais les sorties concernées au chargement. La correction de fond est un réentraînement à 42 sorties.

## R62 — Le temps de transfert de Muskingum vaut quinze à vingt-six fois le temps physique (2026-09-03)

**Statut : établi (la mesure), ouvert (le correctif).** Longueur du tronçon médian après correction de R60 : 2,9 à 3,5 km selon la région, soit environ une heure de parcours à 1 m/s. Le champ borne le temps de transfert à [4, 48] h : la borne INFÉRIEURE excède déjà de quatre à cinq fois le temps physique. Valeurs médianes effectivement déployées par les modèles retenus : 16,5 h (mont), 20,7 h (slso), 26,1 h (outv). Chaque tronçon se comporte donc en réservoir de près d'un jour, effet composé le long de la chaîne topologique, ce qui est le mécanisme candidat du retard d'un à trois jours croissant avec l'aire drainée.

**Ce qui a déjà été réfuté, et ce qui ne l'a pas été.** R2 (2026-08-09) a réfuté un K PHYSIQUE FIXE de 0,35 h : tout se dégradait. Le banc du 2026-09-02 est nul par R60. Aucune expérience n'a encore testé une PLAGE physique apprise. Le banc `alliance/muskingum.sbatch` le fait, par `MEANDRE_KMUSK="0.5,12,3"` contre le témoin `"4,48,24"`, sur le forçage cohérent tout-CaSR, trente époques, six régions appariées.

## R63 — Un forçage demandé et introuvable basculait en silence sur un autre (2026-09-03)

**Statut : établi, corrigé.** `joint_data._paths` résolvait `forcing-<reg><suffixe>.nc` et, si le fichier était absent, retombait sans un mot sur `forcing-<reg>-budyko.nc`. Les forçages hybrides n'ayant jamais été téléversés sur la grappe, le banc de routage du 2026-09-03 y a évalué les champions de la version 1.0, entraînés sur le forçage hybride, contre le forçage tout-CaSR : outv notait 0,671 au lieu de 0,780, et l'écart a d'abord été pris pour un effet du mode de routage. Un forçage explicitement demandé et introuvable est désormais une erreur, sauf pour les régions qui possèdent une entrée propre dans `FORCINGS`. Les tâches de grappe fixent le suffixe et refusent de démarrer si le fichier manque.

## R64 — L'entraînement anéantit l'état riche tous les 45 jours (2026-09-03)

**Statut : établi, non corrigé, le plus destructeur des défauts connus.** Diagnostic apporté par un audit externe, vérifié ligne à ligne et reproduit ici. Le trainer découpe la séquence en blocs (`chunk_steps = 45` dans toutes les configurations du Québec) et appelle `model.simulate` une fois par bloc (`trainer.py:1032`). Il transmet bien l'état d'un bloc au suivant, mais `simulate` en rejette la partie riche : `HydrotelColumn.setup_simulate` fabrique un état neuf par `init_state(theta_init=(0,0,0))` et ne recopie que les teneurs en eau, lesquelles sont de toute façon écrasées juste avant par `frac × porosité` (0,9) dans `model.py`. Le stockage des lacs repart aussi de zéro (`lake_storage = torch.zeros`). Neige, gel, milieu humide et lacs sont donc remis à zéro tous les 45 jours.

**Reproduction** : `python tests/test_chunk_state_annihilation.py`, seize secondes. Sort 538,86 mm d'équivalent en eau du manteau effacés à la frontière de bloc, et le profil de gel réinitialisé.

**Conséquence.** L'entraînement n'a jamais simulé d'hiver continu : les gradients du facteur de fonte, du seuil pluie-neige et de la température de fonte sont appris sur des manteaux de deux semaines, alors que l'évaluation juge une simulation continue sur trois ans. La perte et la métrique de sélection ne portent pas sur le même objet. Cela explique, mieux que tout le reste, pourquoi le réentraînement ne paie pas.

**Défauts voisins vérifiés au passage.** `w_kge = 1.0` cohabite avec `chunk_steps = 45` alors que le commentaire deux lignes plus haut dans le même fichier déclare cette famille de critères incompatible avec le découpage (le Nash-Sutcliffe voisin est désactivé pour cette raison, pas le KGE). Le stockage des lacs est détaché en mode opérateur (`operator_routing.py`, deux endroits). `ETL_KGW_FIELD=1` écrase `sp.k_gw` par un tenseur sans gradient au lieu de le moduler. `float(sp.Z2.mean())` tue le gradient de Z2 et Z3 pour la voie racinaire. `w_physics` est un poids fantôme, son argument n'étant jamais fourni par le trainer.

## R65 — L'hydrogramme simulé est plat un tiers à une moitié du temps (2026-09-03)

**Statut : établi, non corrigé.** Constat visuel d'Essi sur les hydrogrammes du rapport, quantifié depuis les séries. Est dit plat un jour dont le débit varie de moins de 1 % par rapport à la veille.

| région | plat simulé | plat observé | plat en hiver | plus longue suite |
|---|---|---|---|---|
| outm | 54,7 % | 17,7 % | 80,4 % | 102 j |
| abit | 46,1 % | 17,7 % | 73,3 % | 123 j |
| sagu | 38,3 % | 14,1 % | 74,9 % | 105 j |
| gasp | 38,0 % | 12,1 % | 53,7 % | 96 j |
| outv | 32,0 % | 8,7 % | 48,6 % | 54 j |
| slso | 21,2 % | 3,3 % | 41,9 % | 70 j |
| slno | 20,2 % | 5,8 % | 27,8 % | 55 j |
| mont | 6,5 % | 2,5 % | 6,9 % | 16 j |

En hiver le modèle descend une récession exponentielle sans qu'aucun événement ne s'y ajoute : ni fonte de redoux, ni pluie sur neige. Le classement des régions par platitude reproduit celui de leur échec. À rapprocher de R61 (le coefficient de récession est une constante figée sur ces régions, donc la décrue est un exponentiel uniforme) et de R64 (aucun hiver continu n'a jamais été entraîné). Mesure : `python .runs/quebec/banc_defauts.py --region <reg>`.

## R66 — Le champ provincial est une mosaïque de quinze champs disjoints (2026-09-03)

**Statut : établi, non corrigé.** Constat visuel d'Essi sur les cartes de paramètres, quantifié. Pour toutes les paires de tronçons distantes de moins de dix kilomètres, écart relatif médian du paramètre selon qu'une frontière de région les sépare ou non (28 035 tronçons, 8 696 paires transfrontalières).

| champ | même région | de part et d'autre | saut |
|---|---|---|---|
| Z2 | 0,2 % | 48,2 % | 286× |
| Z3 | 0,6 % | 73,0 % | 113× |
| K_sat_3 | 2,1 % | 178,4 % | 83× |
| K_sat_2 | 2,0 % | 150,6 % | 75× |
| K_sat_1 | 2,1 % | 121,6 % | 57× |
| T_melt | 3,6 % | 42,8 % | 12× |

Deux tronçons voisins reçoivent des conductivités qui diffèrent de 120 à 180 % parce qu'une ligne administrative passe entre eux, contre 2 % à l'intérieur d'une région. La frontière n'a aucune existence hydrologique : c'est l'artefact de production que le projet refuse d'exposer. Le champ est lisse à l'intérieur de chaque région et discontinu entre elles, donc l'effondrement du champ a deux visages distincts, à ne pas confondre : à l'intérieur, 27 sorties sur 42 sont quasi constantes (R61) ; entre régions, les sorties vivantes sautent d'un facteur 50 à 300. Un champ provincial unique doit ramener ces rapports vers 1. Mesure : `python .runs/quebec/banc_discontinuite.py`.

## R67 — Le KGE de la perte portait sur 45 jours, celui de la sélection sur trois ans (2026-09-03)

**Statut : établi puis CORRIGÉ.** Défaut signalé par un audit externe, quantifié et réparé ici. Le KGE, le Nash-Sutcliffe et le NRMSE sont des statistiques de séquence : moyennes, écarts-types, corrélation. Toutes les configurations du Québec portent `w_kge = 1.0` avec `chunk_steps = 45`, alors que le commentaire deux lignes plus haut dans le même fichier déclare cette famille incompatible avec le découpage. Le Nash-Sutcliffe voisin est désactivé pour cette raison précise, pas le KGE. La perte était donc calculée sur des fenêtres de 34 jours utiles (45 moins 11 de burn-in), la sélection sur la série continue de trois ans.

**Ampleur mesurée**, sur les huit régions du rapport, KGE médian par station.

| région | KGE continu | KGE moyen par fenêtre de 45 j | corrélation entre les deux |
|---|---|---|---|
| gasp | 0,762 | −0,014 | 0,57 |
| outv | 0,780 | 0,140 | 0,67 |
| slno | 0,768 | 0,158 | **0,05** |
| sagu | 0,741 | 0,085 | 0,91 |
| mont | 0,687 | 0,229 | 0,76 |
| outm | 0,639 | 0,272 | 0,99 |
| slso | 0,601 | 0,196 | 0,69 |
| abit | 0,549 | 0,187 | 1,00 |

Sur le Saguenay nord-ouest, la corrélation entre ce que la perte optimise et ce qui sert à choisir le modèle vaut 0,05 : ce ne sont pas deux mesures du même objet à une échelle près, ce sont deux quantités différentes.

**Correction.** La perte accepte un historique DÉTACHÉ des débits déjà simulés dans l'époque, et les statistiques portent sur toute la séquence vue depuis son début. Le gradient ne remonte que par le bloc courant, ce qui est correct : les blocs passés ont été simulés avec des paramètres antérieurs. Coût mémoire négligeable, l'historique ne portant que les stations. `MEANDRE_KGE_CONTINU=0` restitue l'ancien comportement pour comparaison. Vérifié par `tests/test_training/test_kge_continu.py`, dont un test montre qu'une perte calculée en deux blocs avec historique ÉGALE celle calculée d'un coup sur la séquence entière.

## R68 — Le bassin du lac Abitibi n'a pas un défaut d'appariement, mais une couverture tardive (2026-09-03)

Correction du 2026-09-08 : la région `labi` désigne le bassin du LAC ABITIBI, centré à 79,6 degrés ouest et 48,8 degrés nord, et non le Labrador. L'erreur d'étiquette a circulé dans les comptes rendus du 3 au 8 septembre.

**Statut : clos, aucun correctif requis.** Une passe d'entraînement y annonçait « max valid count: 0 » alors que la base porte 3045 jours d'observations dans la fenêtre d'entraînement, ce qui faisait soupçonner un appariement station-tronçon défaillant. Vérification : l'appariement est juste (station 089907 vers le tronçon 174, aire 222 km²) et les 3045 jours sont bien chargés dans `q_obs`. La station commence simplement le 2010-08-31 quand l'entraînement démarre en 2000 : les 86 premiers blocs de 45 jours ne contiennent aucune observation, et l'avertissement était exact.

**Coût mesuré**, part des blocs d'entraînement sans aucune observation après burn-in, période 2000-2018 :

| région | première observation | blocs vides sur 154 | part |
|---|---|---|---|
| labi | 2010-08-31 | 86 | 55,8 % |
| cnde | 2000-01-01 | 15 | 9,7 % |
| cnda | 2000-01-01 | 11 | 7,1 % |
| cndd | 2000-01-01 | 8 | 5,2 % |
| les onze autres | 2000-01-01 | 0 | 0 % |

Sur les quinze régions, 120 blocs vides sur 2156, soit **5,6 % du calcul d'entraînement rendant un gradient nul**. Réel mais modeste, et concentré sur une région. Sauter les blocs sans observation économiserait ce temps ; l'optimisation n'est pas prioritaire. Vaudreuil reste sans aucune station, ce qui était déjà connu et justifie son modèle transféré.

## R69 — Les deux correctifs d'entraînement débloquent l'hiver, et le KGE annuel ne le voit pas (2026-09-03)

**Statut : établi.** Banc apparié sur grappe, trois régions enneigées, deux bras, trente époques, forçage tout-CaSR cohérent, un seul changement entre les bras : la poursuite de l'état riche entre blocs (R64) et le calcul du KGE sur la séquence continue (R67).

**Contrôle de validité.** Le bras ancien reproduit sur l'Outaouais un KGE médian de 0,7130 quand la version 1.0.1 sur le même forçage donnait 0,7127. La comparaison est donc propre.

**Le KGE annuel dit match nul.**

| région | ancien | corrigé | écart |
|---|---|---|---|
| gasp | 0,767 | 0,732 | −0,035 |
| sagu | 0,755 | 0,782 | +0,026 |
| outv | 0,713 | 0,717 | +0,004 |

**Le comportement hivernal dit autre chose.** Part de jours où la simulation varie de moins de 1 % par jour, nombre d'événements hivernaux (montée de plus de 25 % en un jour) et variabilité d'hiver :

| gasp, hiver | ancien | corrigé | observé |
|---|---|---|---|
| jours plats | 57,9 % | 35,2 % | — |
| événements | 12 | 18 | 17 |
| variabilité | 0,76 | 1,34 | 1,12 |
| plus longue suite plate | 96 j | 51 j | — |

Le KGE d'hiver gagne 0,134 en Gaspésie ; la perte annuelle vient entièrement de l'été (−0,150). **Le score annuel récompensait un hiver écrasé.** Sur l'Outaouais la platitude d'hiver recule aussi (13,8 % à 8,3 %) mais le modèle produit désormais 20 événements pour 16 observés : la dynamique est revenue sans être juste. Le Saguenay ne bouge pas en hiver ; son gain annuel vient du printemps.

**Le bilan de masse annuel s'améliore partout.**

| volume simulé / observé | ancien | corrigé |
|---|---|---|
| gasp | 0,88 | 1,09 |
| outv | 1,21 | 1,14 |
| sagu | 1,19 | 1,15 |

L'erreur absolue baisse sur les trois régions. Les correctifs ne créent pas d'eau, ils la répartissent.

**Conclusion.** Les deux correctifs sont conservés. Ils corrigent deux propriétés de JUSTESSE, pas deux réglages : la perte mesure enfin le même objet que la sélection, et l'entraînement voit enfin des hivers continus. Le KGE médian annuel est disqualifié comme juge unique de la question hivernale. Mesure : `python .runs/quebec/banc_hiver.py <dossier>`.

## R70 — L'amplitude de fonte saisonnière a été calée sous une boucle qui effaçait le manteau (2026-09-03)

**Statut : ouvert, banc écrit.** La modulation vaut `s(j) = 1 + amp·sin(2π(j−81)/365)`, donc décembre ×(1−amp) et juin ×(1+amp), moyenne annuelle inchangée. L'amplitude 0,5 de la recette du socle a été calée quand l'entraînement remettait le manteau à zéro tous les 45 jours : le modèle n'avait alors jamais de manteau profond, et brider la fonte de décembre ne coûtait rien.

Depuis la correction de R64, le manteau persiste, et la répartition mensuelle en porte la trace en Gaspésie :

| volume simulé / observé | hiver | printemps | été | automne | annuel |
|---|---|---|---|---|---|
| avant correction | 0,63 | 0,95 | 0,98 | 0,87 | 0,88 |
| après correction | 0,70 | 1,06 | 1,26 | 1,32 | 1,09 |

L'hiver reste sous-produit de trente pour cent et l'excédent ressort en été et en automne. Signature d'une fonte hivernale trop bridée : l'eau reste dans un manteau qui, lui, persiste désormais.

**Prédiction posée d'avance**, pour ne pas interpréter après coup : baisser l'amplitude doit remonter le rapport de volume d'hiver vers 1 et faire redescendre l'été et l'automne, en Gaspésie surtout. Si l'hiver ne bouge pas, la fonte n'est pas le mécanisme et il faut chercher du côté de la récession de l'aquifère, figée sur ces modèles (R61).

**VERDICT DU 2026-09-03 : PRÉDICTION RÉFUTÉE, le mécanisme est négligeable.** Mesuré en trois minutes sur colonne isolée et forçage réel de Gaspésie (`banc_synthetique.py fonte`, huit nœuds, quatre années dont une de mise en régime), sans réseau, sans routage, sans entraînement :

| amplitude | manteau au 1er avril | production hiver | printemps | total annuel |
|---|---|---|---|---|
| 0,50 | 383 mm | 19 mm/an | 55 mm/an | 109 mm/an |
| 0,25 | 378 mm | 21 mm/an | 54 mm/an | 109 mm/an |
| 0,00 | 374 mm | 23 mm/an | 51 mm/an | 110 mm/an |

Supprimer entièrement la modulation ne libère que 4 mm/an d'eau en hiver, contre un déficit mesuré de trente pour cent sur bassin. L'amplitude déplace 10 mm de manteau, rien de plus. **Aucune passe de grappe n'est engagée sur cette piste** ; `alliance/fonte.sbatch` reste écrit mais n'a pas à tourner. Le déficit hivernal vient d'ailleurs, et R61 (récession de l'aquifère figée) est le candidat suivant.

**Méthode validée au passage.** Neuf essais de grappe à huit heures ont été évités par trois minutes de banc. Deux pièges payés en le construisant : un climat entièrement fabriqué était trop froid pour fondre (janvier à −12 °C sans redoux), donc aucun réglage de fonte n'y produisait d'effet ; et `melt_seasonal_amp` n'est injecté que par `params_from_nerf`, si bien qu'un banc passant par `set_static` le contournait et donnait trois résultats identiques au millimètre. C'est cette exactitude impossible qui a trahi l'erreur. Le bon partage est de fabriquer le SYSTÈME (colonne isolée) et de garder la MÉTÉO réelle.

## R71 — REQUALIFICATION : tout verdict obtenu par entraînement est antérieur à une boucle juste (2026-09-03)

**Statut : avertissement permanent, à lire avant de citer un verdict de ce registre.**

Deux défauts de la boucle d'entraînement ont été établis et corrigés le 2026-09-03. Ils ne datent pas d'hier : ils étaient présents dans **tout entraînement jamais lancé dans ce projet**.

- R64, l'état riche anéanti tous les 45 jours. Le manteau neigeux, le profil de gel, le milieu humide et le stockage des lacs repartaient de zéro à chaque bloc, et les teneurs en eau du sol étaient réinitialisées à 0,9 fois la porosité. Aucun entraînement n'a jamais simulé d'hiver continu, ni laissé un lac se remplir.
- R67, le KGE de la perte calculé sur 45 jours quand la sélection le calcule sur trois ans. Sur une région, la corrélation entre les deux valait 0,05.

**Conséquence à porter.** Un verdict de la forme « telle piste a été testée par réentraînement et n'a pas payé » a été rendu par une boucle qui n'entraînait pas ce qu'on croyait. Il ne devient pas faux pour autant, mais il perd son autorité : il dit que la piste n'a pas paye SOUS UNE BOUCLE CASSÉE. Un verdict de mesure, d'inférence pure ou de lecture de code n'est pas concerné.

**Décompte sur ce registre**, sur 44 entrées de tableau : 14 verdicts s'appuient sur un entraînement (R3, R12, R13, R18, R19, R24, R26, R27, R29, R31, R33, R40, R43, R44) et 13 sur de l'inférence pure ou une mesure. Les entrées longues (R45 et suivantes) sont à qualifier au cas par cas selon la même règle.

**Preuve que la requalification n'est pas théorique.** R69 montre que la correction de la boucle fait passer l'hiver gaspésien de 12 à 18 événements pour 17 observés, et sa variabilité de 0,76 à 1,34 pour 1,12 observée, alors que le KGE annuel BAISSE de 0,035. Une piste hivernale jugée sur le KGE annuel sous l'ancienne boucle aurait été écartée deux fois à tort : parce que l'hiver n'était pas simulé, et parce que le juge ne regardait pas l'hiver.

**Règle pratique.** Avant de rouvrir une piste écartée, vérifier deux choses : le verdict reposait-il sur un entraînement, et son critère était-il un KGE annuel médian. Si les deux réponses sont oui, le verdict est à refaire avant d'être cité. Le cas de R32, l'amplitude de fonte saisonnière, est le premier traité sous cette règle (voir R70).

## R72 — Le mode d'évapotranspiration est le plus gros levier du bilan d'eau (2026-09-03)

**Statut : établi, et il retire une fausse alerte.** Colonne isolée sur le forçage réel de Gaspésie, huit nœuds, quatre années dont une de mise en régime, aucun réseau ni entraînement (`banc_synthetique.py bilan`).

| mode d'ETP | ETR | part de la pluie | production | coefficient d'écoulement |
|---|---|---|---|---|
| McGuinness brut | 766 mm/an | 88 % | 110 mm/an | 0,13 |
| Penman | 737 mm/an | 85 % | 150 mm/an | 0,17 |
| Oudin | 526 mm/an | 60 % | 365 mm/an | 0,42 |
| **Linacre calée du projet** | **376 mm/an** | **43 %** | **514 mm/an** | **0,59** |

Précipitation 871 mm/an. Fourchette de référence : ETR boréale 400 à 500 mm/an, coefficient d'écoulement du Québec méridional 0,45 à 0,65.

**Fausse alerte retirée.** Le premier bilan, en McGuinness brut, montrait 88 % d'évapotranspiration et un coefficient de 0,13, ce qui aurait fait de la fuite d'eau le défaut dominant devant toute question de forme d'hydrogramme. C'était un artefact de la colonne NON ANCRÉE. Le coefficient de calage Linacre de la plateforme vaut 0,399 en médiane sur ces nœuds : il divise l'évapotranspiration par deux et demi, et c'est lui qui rend le bilan juste. **Le modèle déployé n'a pas de fuite d'eau.**

**Ce que le résultat établit tout de même.** Un facteur 1,5 sur l'évapotranspiration devient un facteur 4 sur l'écoulement, parce que la production est la petite différence de deux grands termes. Le choix du mode d'ETP domine donc tout autre réglage du bilan, et le coefficient calé de Linacre n'est pas un détail d'ancrage mais la pièce qui rend le modèle physiquement correct. C'est l'énoncé quantitatif de la loi des ancrages, et la raison concrète pour laquelle un point de reprise seul ne définit pas un modèle.

**Conséquence de méthode.** Tout banc sur colonne isolée doit poser les paramètres d'ancrage avant de conclure quoi que ce soit sur des volumes. `banc_synthetique.py` documente le piège.

## R73 — L'équation d'évapotranspiration n'est pas le sujet, une seule constante l'est (2026-09-03)

**Statut : établi.** Question posée par Essi : Hydrotel s'en sort avec McGuinness ou Linacre et un coefficient constant par région ; pourquoi le champ spatial n'y arriverait-il pas ? Mesuré sur colonne isolée, forçage réel de Gaspésie, quatre années dont une de mise en régime.

**Correction de vocabulaire d'abord, elle avait faussé un test.** Une PLATEFORME est une configuration de modèle, pas une région. Hydrotel en a six : LN24HA sur Linacre, les cinq MG24H* sur McGuinness. Vérifier « les plateformes de la Gaspésie, de l'Outaouais, de Montréal » revenait à vérifier cinq régions de LA MÊME plateforme, celle qui est sur Linacre ; l'absence de fichier McGuinness n'y prouvait rien.

**Comparaison loyale**, région Gaspésie, plateforme MG24HK, chaque formule avec SON coefficient :

| formule | coefficient calé | ETR | part de la pluie | coefficient d'écoulement |
|---|---|---|---|---|
| McGuinness brute | — | 766 mm/an | 88 % | 0,13 |
| McGuinness calée | 0,500 | 387 mm/an | 44 % | 0,58 |
| Linacre calée | 0,485 | 457 mm/an | 52 % | 0,50 |

Les deux équations donnent le même résultat une fois calées, et leurs coefficients sont voisins. **L'équation est interchangeable ; la constante multiplicative fait tout.**

**Pourquoi cette constante est difficile.** Balayage du coefficient Linacre, même colonne :

| coefficient | ETR | coefficient d'écoulement |
|---|---|---|
| 1,000 | 816 mm/an | 0,10 |
| 0,800 | 750 mm/an | 0,17 |
| 0,600 | 566 mm/an | 0,37 |
| 0,399 | 376 mm/an | 0,59 |
| 0,300 | 283 mm/an | 0,69 |

Un facteur 2,5 sur la constante déplace l'écoulement d'un facteur 6, la production étant la petite différence de deux grands termes. La direction est extrêmement raide.

**Réponse à la question posée.** Le champ dispose du bon levier : `K_c`, borné [0,3, 1,5], multiplie l'ETP, et la valeur cherchée (0,4 à 0,5) tombe dans cette plage. **Il PEUT la représenter.** L'audit du 2026-09-03 le mesure pourtant à 0,975 de médiane avec 14 % d'étalement, donc collé à 1. Deux raisons, à ne pas confondre :

1. Dans la configuration déployée, Linacre est déjà ancrée à 0,4 : un `K_c` proche de 1 par-dessus est CORRECT, et le champ n'a jamais eu à chercher la constante.
2. Son prior le tire vers 0,85, loin de la réponse, sur une direction où l'erreur de débit varie d'un facteur 6. Un optimiseur qui part à 0,85 et qu'un prior y retient a peu de chances de descendre à 0,45.

Le champ n'échoue donc pas à s'adapter : on ne le lui a jamais demandé, et son prior l'en empêcherait. **Hypothèse à tester** : sans ancrage d'ETP, un entraînement fait-il descendre `K_c` vers 0,45 ? Si non, le prior et les bornes de `K_c` sont mal centrés et c'est un défaut réparable.

## R74 — Avec les paramètres d'Hydrotel, méandre reproduit Hydrotel : le clone est fidèle de bout en bout (2026-09-04)

**Statut : établi.** Question d'Essi : Hydrotel fonctionne, méandre non, nous avons le code d'Hydrotel et ses paramètres, pourquoi le défaut n'est-il pas identifiable ? Réponse par un duel à trois sur le même sous-bassin, même météo, même période : méandre contre Hydrotel isole le défaut de tout ce qui n'est pas méandre (données, jauge, régime), puisque tout le reste est commun.

Sous-bassin de la Gaspésie en amont de la station 021702, 33 tronçons, 223 km², évaluation 2020-2024 (période couverte par le post-traitement d'Hydrotel), `banc_sousbassin.py`, aucun entraînement :

| configuration de méandre | KGE vs observé | KGE vs Hydrotel | r vs Hydrotel | gamma vs observé |
|---|---|---|---|---|
| sol de littérature, sans ancrage | 0,354 | 0,440 | 0,733 | 0,462 |
| sol d'Hydrotel imposé en entier, sans aquifère | 0,673 | 0,853 | 0,898 | 0,879 |
| recette du socle (tout sauf K_sat), aquifère | 0,651 | 0,863 | 0,947 | 0,795 |
| **Hydrotel lui-même** | **0,661** | — | — | **0,863** |

**Ce que cela établit.** Avec les paramètres d'Hydrotel, méandre reproduit Hydrotel à 0,947 de corrélation et l'égale contre l'observé. La colonne clonée, le réseau et le routage sont donc fidèles de bout en bout, pas seulement module par module. Le défaut n'est PAS dans le code de la physique.

**Où naît la platitude.** Le sol de littérature seul donne gamma 0,462 : c'est le chiffre « qui ne vaut rien ». Le socle le ramène à 0,795, à sept centièmes d'Hydrotel, et l'écart restant tient au seul paramètre laissé libre, K_sat. Les hydrogrammes plats du rapport (R65, 38 % de jours plats en Gaspésie) sont ceux de champions ENTRAÎNÉS sous la boucle qui effaçait le manteau : l'entraînement les a rendus plus plats que le zéro époque, ce qui rejoint la loi des ancrages (OUTV 0,739 sans entraînement contre 0,499 entraîné).

**Ce que cela ne dit pas encore.** Le pari du projet est de BATTRE Hydrotel par le champ, pas de l'égaler. Aucun entraînement sous une boucle juste n'a encore montré que le champ rend les hydrogrammes plus nets. Le test est en cours sur ce même sous-bassin (`--entrainer 20`) : si gamma dépasse 0,863 en apprenant K_sat, le champ bat Hydrotel sur ce qui compte ; s'il descend, la réponse est définitive dans l'autre sens.

**Réponse à « six mois à tourner en rond ».** Le modèle n'était pas cassé ; l'appareil pour l'entraîner (R64, R67) et le juger (R69) l'était, et chaque essai coûtait des heures. Les trois verrous ont sauté les 3 et 4 septembre, et ce duel a coûté deux minutes par bras.

## R75 — La validation du trainer démarrait sans manteau neigeux : le sélecteur de champions était faux de 0,43 (2026-09-04)

**Statut : établi, corrigé.** Sur le sous-bassin gaspésien 021702 (`banc_sousbassin.py --entrainer`), le trainer rapportait après une époque un KGE de validation de 0,399 avec un biais de 0,619, alors que le même point de reprise, évalué par une simulation continue indépendante, vaut **0,829** sur la même période 2018-2019 (zéro époque : 0,823). Essi a relevé l'anomalie : « méandre part au moins à 0,6 ».

**Mécanisme.** La validation reprend l'état de fin d'entraînement (chemin rapide) ou une mise en régime de 730 jours (chemin lent), puis appelle `simulate`, qui REFABRIQUAIT l'état riche de la colonne : manteau, gel, milieu humide, lacs. La validation démarrait donc chaque année sans neige et ratait la crue, d'où le biais de 0,62. C'est R64 sous une troisième forme, après les blocs d'entraînement et la mise en régime du banc. Le premier bloc d'entraînement après la mise en régime jetait de même le manteau spinné, ce qui explique les gradients NaN observés au premier bloc de chaque époque (KGE de janvier sur un débit quasi constant).

**Portée.** Ce chiffre faux est celui qui choisit le meilleur point de reprise (`best_metric`) et déclenche l'autopilote (paliers de taux, redémarrages sur régression). **Tous les champions du projet ont été sélectionnés par une validation sans neige**, et les redémarrages de l'autopilote ont pu répondre à des régressions fantômes. C'est une explication directe du motif « l'entraînement dégrade le modèle » (loi des ancrages : OUTV 0,739 sans entraînement contre 0,499 pour le champion entraîné), à ajouter à R64 et R67.

**Correctif.** L'état riche traverse désormais les trois frontières : fin d'entraînement vers validation, mise en régime vers validation, mise en régime vers premier bloc. Même levier `MEANDRE_ETAT_CONTINU`. Tests d'entraînement au vert. Le test du champ sur le sous-bassin est relancé avec une sélection enfin juste ; ses deux premiers essais (validation fausse) sont jetés.

**Règle de méthode confirmée.** Toute mesure du trainer doit être recoupée au moins une fois par une simulation continue indépendante, et l'écart de 0,43 ici a été trouvé en trois minutes par ce recoupement.

## R75 bis — RECTIFICATION : l'écart de validation de 0,43 venait du banc, pas du trainer (2026-09-04)

**Statut : R75 est requalifié.** L'écart entre la validation du trainer (0,399, puis 0,560 après le correctif de frontière) et l'évaluation continue indépendante (0,82) n'était pas dû au manteau jeté à la validation. Il venait d'une **convention du trainer que le banc de sous-bassin violait** : `q_obs[0]` correspond à `forcing[train_slice.start]`, pas à `forcing[0]` (voir `joint_data.py`, `q_obs=q_obs[sl_.start:]`, et `trainer.py`, `obs_offset = 0` puis `q_obs_val = data.q_obs[:n_val]`). Le forçage porte la mise en régime en amont, les observations commencent au premier jour jugé. Le banc passait `q_obs` complet : la boucle comparait la simulation de 2010 aux observations de 2008, et la validation celle de 2018 à celles de 2008. La première version du banc, sans mise en régime, comparait de même 2018 à 2010.

**Ce qui reste de R75.** Le correctif du trainer est conservé parce qu'il est physiquement juste, dans la droite ligne de R64 : `simulate` refabriquait bien l'état riche aux trois frontières, et le manteau était bien jeté. Mais son EFFET sur le score n'est plus mesuré, et la phrase « tous les champions du projet ont été sélectionnés par une validation sans neige » est **retirée** tant qu'une mesure propre ne l'a pas établie : la mesure qui la fondait était confondue par le désalignement.

**Ce que cela n'entame pas.** La méthode a fonctionné : le recoupement par simulation continue indépendante a détecté l'écart en trois minutes, deux fois, et c'est lui qui a conduit à la convention. Le banc est corrigé, les trois essais d'entraînement faits sous le désalignement sont jetés, le test du champ repart sur GPU.

**Leçon consignée dans le code.** Un bloc de commentaire dans `banc_sousbassin.py` porte la convention. Elle devrait être un contrat vérifié dans `TrainingData` (une assertion sur les longueurs) plutôt qu'une convention tacite ; c'est le troisième défaut de cette famille en deux jours.

## R76 — Rabotage et platitude, séparés : le rabotage est partagé avec Hydrotel, la platitude n'existe pas avant entraînement (2026-09-04)

**Statut : établi sur un sous-bassin, à confirmer sur les 59 (grappe, tâche 2394135).** Deux questions d'Essi qu'un gamma bas ne distingue pas : l'hydrogramme est-il raboté, et est-il plat par moments. Mesuré sur la Gaspésie 021702, 2020-2024, `banc_sousbassin.py` (fonction `forme`).

| | pointes annuelles sim/obs | q95 | q99 | jours plats | plats en hiver | plus longue suite |
|---|---|---|---|---|---|---|
| socle, zéro époque | 0,58 | 0,71 | 0,72 | 6,7 % | 9,2 % | 7 j |
| Hydrotel | 0,54 | 0,66 | 0,67 | 5,3 % | 2,2 % | 10 j |
| observé | 1,00 | 1,00 | 1,00 | 10,2 % | 21,9 % | 27 j |

**Raboté : oui, fortement, et Hydrotel autant.** Les deux modèles rendent 54 à 58 % des pointes annuelles et environ 70 % du quantile 95. Le défaut est partagé, donc il n'est pas dans ce que méandre fait de différent ; il est en amont des deux (forçage aux pointes, ou physique commune). Chantier à part.

**Plat : non, pas avant entraînement.** Le socle est MOINS plat que la réalité (6,7 % contre 10,2 % de jours plats ; 9 % contre 22 % en hiver) : l'observé sous glace est réellement constant et aucun des deux modèles ne le reproduit. La platitude de R65 (38 % de jours plats en Gaspésie) est celle des champions ENTRAÎNÉS sous la boucle cassée ; elle est absente du modèle ancré à zéro époque.

**D'où vient alors l'écart de gamma (0,82 contre 0,875) ?** Pas des pointes, que méandre rabote moins. Du NIVEAU MOYEN : biais 0,88 pour méandre contre 0,74 pour Hydrotel à écart-type comparable, donc coefficient de variation plus bas. Le gamma d'Hydrotel est meilleur en partie parce que sa moyenne est plus fausse. **Gamma ne doit plus être lu seul** ; le banc imprime les deux diagnostics à côté.

**Bruit d'initialisation, à connaître avant tout verdict.** Sans graine fixe, trois zéro époque successifs ont donné 0,811, 0,817 et 0,850 de gamma pour la même configuration : l'initialisation aléatoire du champ vaut l'écart à Hydrotel. Graine fixée dans le banc depuis ; un verdict d'entraînement se lit en écart APPARIÉ dans un même run. Les 59 sous-bassins de la grappe (lancés avant la graine) restent valides bassin par bassin, leurs références absolues sont bruitées.

## R77 — Un seul bloc à gradient NaN mettait à zéro l'apprentissage du champ pour toute l'époque (2026-09-04)

**Statut : établi, corrigé.** Le trainer accumule les gradients de tous les blocs de 45 jours et ne fait qu'UN pas d'optimisation par époque. Avant ce pas, tout gradient contenant inf ou NaN était remplacé par zéro. Or le premier bloc de chaque époque rend un gradient NaN : c'est un KGE de 34 jours de janvier sous glace, sans historique (le KGE continu de R67 n'en a pas encore), sur un débit quasi constant, et le plancher absolu de 1e-8 sur les variances donnait des écarts-types de 1e-4 par lesquels `r` et `gamma` divisaient. Conséquence : **le gradient accumulé de tout le champ spatial était mis à zéro à chaque époque, et le champ n'apprenait rien**, silencieusement.

**Signature qui l'a trahi**, sur le sous-bassin gaspésien 021702 : validation identique à la quatrième décimale entre les époques 0 et 1 (KGE 0,8416, NSE 0,8345, gamma 0,882) alors que la perte d'entraînement bougeait (1,026 → 0,950), donc seulement par le prior sur des paramètres sans effet sur le débit. Avec `val_every = 1`, une validation identique au dix-millième sur deux états de paramètres différents n'est possible que si les paramètres qui comptent n'ont pas bougé.

**Portée, MESURÉE et plus étroite que ce que la première rédaction avançait.** Les journaux locaux de trois entraînements sur région entière (Montréal sur deux époques, deux fois ; Côte-Nord E sur une époque) ne contiennent AUCUN message « NaN/Inf gradients … zeroed ». Le poison frappe le banc de sous-bassin, où la station est UNIQUE et son premier bloc de janvier sans variance ; sur une région à vingt-huit stations, le KGE du premier bloc est moyenné sur des stations dont certaines varient, et reste fini. L'hypothèse que les anciens champions aient été privés d'apprentissage par ce mécanisme n'est donc PAS soutenue par les journaux disponibles ; elle reste à vérifier sur les journaux de grappe (`grep -c "NaN/Inf" ~/scratch/meandre/*/*.log`). Le correctif reste juste et nécessaire pour tout banc à peu de stations.

**Correctifs.** (1) Plancher RELATIF sur les écarts-types du KGE, 0,1 % du débit moyen : gradient fini sur une fenêtre constante, effet nul ailleurs. (2) Filet par bloc dans le trainer : copie des gradients avant chaque bloc, restauration si le bloc les rend non finis, bloc compté comme jeté. Un bloc perdu coûte 1,5 % de l'époque ; il n'en coûtait 100 %. Tests au vert. **Tous les entraînements des 3 et 4 septembre sur sous-bassin sont à refaire, ainsi que la tâche de grappe 2394135 (59 sous-bassins), lancée avec le défaut.**

## R77 bis — Sur la grappe, le poison a bien frappé des régions entières, et de façon ASYMÉTRIQUE entre les bras (2026-09-04)

**Statut : établi.** Comptage du message « NaN/Inf gradients … zeroed » dans les journaux de grappe (une occurrence = une époque dont le gradient du champ a été mis à zéro) :

| journal | époques perdues sur 30 |
|---|---|
| correctifs / gasp-corrigé | 5 |
| correctifs / gasp-ancien, outv (2), sagu (2) | 0 |
| kmusk / mont-témoin | 3 |
| kmusk / slso-témoin | 3 |
| kmusk / les dix autres | 0 |
| sousbassins / gasp-021702, mont-030424, slno-052233 | 2 sur 2 (toutes) |
| sousbassins / mont-030919 | 1 sur 2 |

**Ce que cela change.** La restriction de R77 (« les régions entières ne sont pas touchées ») valait pour trois journaux locaux ; elle ne vaut pas en général. Sur une région entière, le premier bloc a une frontière tirée au hasard à chaque époque (revue du 2026-07-01) : certaines époques tombent sur une fenêtre à variance quasi nulle, et le champ perd alors toute l'époque. Le taux est faible (0 à 17 %) mais **non nul et inégal entre les bras d'un même banc**. Sur le banc apparié des correctifs (R69), le bras corrigé de la Gaspésie a perdu 5 époques sur 30 et son témoin aucune : l'écart de −0,035 y est confondu par cinq époques de handicap, et R69 doit se lire avec cette réserve. Les bancs de sous-bassin à station unique perdent TOUTES leurs époques, comme prévu.

**Conséquence pratique.** Tout verdict d'entraînement antérieur au correctif de R77 doit être accompagné du comptage de ce message dans son journal ; un bras qui a perdu des époques n'est pas comparable à un bras qui n'en a pas perdu. Le filet par bloc rend le comptage inutile pour les runs à venir (un bloc jeté coûte 1,5 % d'une époque, pas 100 %). La tâche 2394135 est annulée et relancée en 2396210 avec le correctif.

**Complément R77, 12 h 04.** Sur le run relancé avec les deux correctifs, le premier bloc rend ENCORE un gradient non fini malgré le plancher relatif sur les écarts-types du KGE : le filet par bloc l'a jeté et a restauré les gradients (« bloc 0 (t=731) : gradient non fini, bloc JETE »), et l'époque continue. Le plancher n'a donc pas supprimé la cause, qui est ailleurs que dans `r` et `gamma` (candidats : le burn-in de 11 jours qui laisse 34 jours, la division par la moyenne dans `beta` ou `gamma`, ou un terme de la perte hors KGE). Le filet suffit à sauver l'époque (un bloc sur soixante-cinq perdu) ; la cause exacte reste à isoler hors ligne, sur ce seul bloc, en trois minutes.


## R78 — Le trainer ne fait qu'UN pas d'optimisation par époque, après cinq époques de réchauffement (2026-09-04)

**Statut : établi par lecture du code et par mesure ; conséquence en cours de test.** `_train_epoch` remet les gradients à zéro avant la boucle des blocs, accumule les gradients de tous les blocs (pondérés par leur durée), et appelle `optimizer.step()` UNE fois après la boucle : c'est une descente de gradient en plein lot, une époque = un pas d'Adam. Par défaut `warmup_epochs = 5`, donc les cinq premiers pas sont à taux presque nul. Trente époques valent donc vingt-cinq pas d'Adam utiles à 5e-4.

**Mesure.** Sur le sous-bassin gaspésien, le point de reprise écrit après l'époque 0 est IDENTIQUE, à 5 décimales de norme près, au modèle initialisé avec la même graine ; la validation est identique au dix-millième entre les époques 0 et 1 ; la perte d'entraînement ne bouge que par le tirage aléatoire des frontières de blocs. L'optimiseur tient pourtant bien les 96 227 paramètres, encodeur spatial compris (vérifié). Un essai de cinq époques, pensé comme test rapide, tenait tout entier dans le réchauffement et ne pouvait rien montrer ; les deux verdicts « le champ n'apprend pas » rendus ce matin sur ce banc en sont la conséquence, avant même R77.

**Conséquence probable, à confirmer.** Un champ spatial de 96 000 paramètres n'apprend pas en vingt-cinq pas. C'est l'explication la plus simple de « zéro époque ≈ trente époques » (OUTV 0,739 contre 0,781, loi des ancrages) et de « l'entraînement dégrade » : le modèle n'a presque jamais été entraîné ; ses performances viennent de l'initialisation de littérature et des ancrages. Les défauts R64, R67, R75 et R77 sont réels mais portaient sur un entraînement qui ne déplaçait de toute façon presque rien.

**Correctif, sous levier `MEANDRE_PAS_PAR_BLOC=1`** : un pas d'optimisation après chaque bloc (la troncature du gradient habituelle en séries temporelles), soit environ 65 pas par époque au lieu d'un, avec le filet de R77 par bloc ; le pas de fin d'époque est alors sauté. Le banc de sous-bassin pose aussi `warmup_epochs = 0`. Test en cours : 3 époques ≈ 195 pas, contre 30 pas pour toute une flotte antérieure.

**Complément R78, 13 h 55 : le gradient est cohérent, la dégradation vient du pas et de l'objectif mouvant.** Objection d'Essi : un entraînement ne peut pas aller contre sa perte. Vérifié sur le banc mini (384 nœuds, 60 jours, cible artificielle) : un pas d'Adam sur un bloc fait BAISSER la perte de ce même bloc à tout taux, 2,115 → 2,111 (1e-5), → 2,074 (1e-4), → 1,915 (5e-4). La machinerie de gradient est saine. La montée de la perte moyenne d'époque sur le sous-bassin (1,04 → 1,28) et l'effondrement de la validation (0,83 → 0,55, biais 1,22, gamma 0,63) après 65 pas viennent donc (1) de taux réglés pour UN pas par époque et appliqués soixante-cinq fois, groupes à ×10 et ×50 compris, et (2) d'un objectif non stationnaire d'un bloc à l'autre : les dix premiers blocs de chaque époque n'ont qu'un historique d'hiver, donc les dix premiers pas optimisent un KGE d'hiver seul. Deux essais en cours ou préparés : taux divisé par dix (lancé 13 h 41), et amorçage de l'historique par une passe sans gradient en début d'époque, pour que chaque bloc voie le KGE de toute la période.


## R79 — Le banc de sous-bassin optimisait le KGE du bloc seul : la branche groupée de la perte ignore l'historique (2026-09-04, 14 h 35)

**Statut : établi, corrigé.** `HydroLoss` a deux branches. La branche par station (`per_station=True`, celle que `slso.py` passe et que la version 1.0 a toujours utilisée) calcule le KGE sur l'historique détaché plus le bloc courant (R67). La branche groupée, prise par défaut, calcule NSE, KGE, PBIAS et NRMSE sur le bloc courant seul, sans historique. Le banc de sous-bassin ne passait pas l'option : tous ses entraînements jusqu'à 14 h 35 (R75, R77, R78, les essais rapide et socle) ont optimisé un KGE de quinze ou quarante-cinq jours, la faute même que R67 corrige, et l'amorçage de l'historique n'y servait à rien. Signature dans le journal : un Nash-Sutcliffe imprimé dans les composantes alors que son poids est nul (il n'est calculé que dans la branche groupée). Correctif : `per_station=True` dans les deux pertes du banc. La version 1.0 n'est pas concernée. Les verdicts d'entraînement du banc antérieurs à ce correctif sont caducs.

## R80 — Un bloc de quinze jours n'a aucune perte de débit, et le pas se fait alors sur le prior seul (2026-09-04, 15 h 20)

**Statut : établi par trace, corrigé.** La perte par station ne garde une station que si le bloc compte au moins trente observations valides. Avec des blocs de quinze jours et un rodage du quart, il reste douze jours vivants : aucune station gardée, perte de débit nulle, et l'avertissement « No stations have >= 30 valid observations » 23 fois par époque sur 24 blocs. Le trainer ajoute le prior de littérature à chaque bloc ; la perte du bloc a donc un gradient, celui du prior seul, et avec un pas par bloc (R78) l'optimiseur a fait 23 pas par époque vers les cibles de littérature sans qu'aucune observation ne pèse. Mesure sur gasp 021702, blocs de quinze jours, taux 2e-4, 144 pas : validation 2013 de 0,834 à 0,570 avec MOD16, à 0,672 sans MOD16, à 0,726 au KGE seul ; dans tous les cas beta chute (0,91 vers 0,67 à 0,78) et l'ET simulée monte (0,89 vers 1,03 mm par jour) sans qu'aucun terme ne la vise : c'est le prior qui tire. Trace directe : une seule ligne de débogage du terme KGE par époque, sur le bloc final de 51 jours ; le terme KGE valait 0,04 (51/366 fois 0,23) quand 1 moins KGE de l'année valait 0,25.

Deux correctifs. Dans la perte, le minimum de trente compte l'historique détaché plus le bloc, et exige au moins une observation vivante. Dans le trainer, un bloc dont la perte de données n'a pas de gradient ne fait pas de pas (`_sans_donnees`) : le gradient du prior seul est jeté et le bloc compté.

**Ce que ces essais disent quand même.** Le sous-bassin gasp 021702 a un bilan d'eau fermé au socle : précipitation CaSR 967 mm par an, débit observé 628, différence 340, ET simulée 325 sur la période d'évaluation. MOD16 y vaut 551 mm par an, 60 % de plus que le bilan ne permet. L'essai avec MOD16 a vidé la rivière (beta 0,67), mais il comparait l'ET en NIVEAU absolu (`et_mode="level"`, défaut de `HydroLoss`), alors que la recette du socle passe le terme en TENDANCE (`et_mode="anomaly"`, R24) : MOD16 y donne la forme de l'ET, jamais son volume, précisément parce que les sources ne sont pas cohérentes en niveau (rappel d'Essi, 2026-09-04, 15 h 30). Le banc laissait le défaut ; corrigé. La conclusion « MOD16 vide la rivière » ne vaut que pour un mode que la recette n'utilise pas, et l'écart de 60 % en niveau reste une mesure utile de cette incohérence, pas un défaut de la perte. Par ailleurs l'année d'entraînement 2012 (beta 1,153 au socle) et l'année de validation 2013 (beta 0,911) se contredisent de 24 % sur le volume : un entraînement sur une seule année est structurellement trompeur sur beta, limite du mode rapide.


## R80 bis — Sous une boucle juste, l'entraînement suit sa perte ; une seule année d'entraînement ne généralise pas sur le volume (2026-09-04, 15 h 40)

**Statut : établi sur un sous-bassin, à confirmer à l'échelle régionale (flotte de fin de semaine, bras A contre B).** Après R79 et R80, essai au KGE seul sur gasp 021702, un an d'entraînement (2012), blocs de quinze jours, 24 pas par époque, taux 2e-4 en cosinus, six époques, graine 1234. Chaque bloc voit l'année entière (344 à 354 jours d'historique détaché plus 12 à 22 jours vivants) et le terme de perte, 0,25 au départ, coïncide avec 1 moins le KGE de l'année mesuré hors ligne. Perte d'entraînement de 0,25 à 0,17 en 144 pas ; KGE 2012 de 0,751 à 0,788 au meilleur point (beta de 1,153 à 1,108) ; validation 2013 de 0,834 à 0,818 après une époque, puis 0,805, 0,807, 0,813, 0,814 : baisse puis remontée, sans dérive. Pointes 0,81 contre 0,83 au départ, platitude 5,8 % contre 6,3 % : pas de rabotage, pas d'aplatissement.

Lecture : l'optimiseur améliore bien son objectif, et l'écart de validation vient du conflit de volume entre 2012 (beta 1,15) et 2013 (beta 0,91), qu'une seule année d'entraînement ne peut pas arbitrer. Les six essais antérieurs de la journée qui montraient une validation en chute (0,57 à 0,73) mesuraient les défauts R79 et R80, pas le modèle. La question « l'entraînement améliore-t-il le modèle sur plusieurs années » reste ouverte et passe à la flotte.


## R81 — Le pilote lisait la profondeur des lacs comme leur surface (2026-09-05)

**Statut : établi, corrigé dans le pilote ; sept scripts d'évaluation portent encore la même lecture.** `etl_run.py` impose la loi de tarage d'Hydrotel à chaque lac (`ETL_LAKE_TRL=1`) en lisant `troncon.trl` ; il prenait les jetons (ptr+1, ptr+2, ptr+3) pour (surface en km², c, k), alors que depuis R60 le bloc d'un lac est (surface en m², profondeur en m, c, k). Vérifié sur OUTV : médiane du jeton ptr 631 712 m², médiane du jeton ptr+1 3,5 m (p90 17,8). Chaque lac recevait donc une surface de 3,5 km² en médiane au lieu de 0,63, et k_lake = c/A cinq à trente fois trop petit ; la ligne écrasait aussi les surfaces HydroLAKES posées correctement plus haut. Constante de temps des lacs à 0,1 m³/s : 12 à 17 jours en médiane au lieu de 0,5 à 1,3 jour, p90 37 à 110 jours, maximum de plusieurs années ; zéro lac au-dessus de 30 jours avec la vraie surface, contre 9 sur 63 en Gaspésie, 57 sur 348 au Saguenay, 108 sur 514 en Outaouais. Présent dans le modèle entraîné comme non entraîné ; ce n'est PAS la cause des plateaux d'été (R83), les 88 couples station-bras sans aucun lac en amont étant aussi plats que les autres, mais il aplatit les régions lacustres et rend caducs les bancs de routage qui l'ont utilisé. Lectures à aligner : eval_periodes.py, fidelite_hydrotel.py, hgm_ab.py, lacs_trl.py, banc_routage.py, banc_micro_routage.py, banc_modules.py.

## R82 — Le KGE à historique détaché avait un gradient cent fois trop faible (2026-09-05)

**Statut : établi par calcul, corrigé.** Une statistique de séquence sur N = N_hist + n jours a un gradient en 1/N par jour vivant ; la MSE du bloc l'a en 1/n. Le trainer multiplie ensuite la perte du bloc par n/N_train. Le KGE recevait donc un facteur n/N de trop par rapport aux termes locaux : 34/2934, soit 1,2 %, pour un bloc de 45 jours après un an d'historique, 0,5 % avec l'amorçage. Or gamma est le SEUL terme de toute la perte qui punit directement la platitude ; il était de fait inerte à chaque pas, et la valeur imprimée, juste, le cachait. Correctif dans `loss.py` : la sensibilité par jour vivant est remise à l'échelle N/n (valeur inchangée, gradient multiplié), pour le KGE, le NSE et le NRMSE quand l'historique est présent. Conséquence : aucun entraînement antérieur n'a jamais été tenu par gamma.

## R83 — Les plateaux d'été sont créés par l'apprentissage, portés par la troisième couche du sol, et la perte les tolère (2026-09-05)

**Statut : établi sur la flotte du 4 septembre (dumps 2022-2024, 28 runs, 92 stations) ; paramètres responsables à lire dans les points de reprise.**

Anatomie (agent 1). Deux régimes. Plateaux d'HIVER, communs aux deux bras et présents dans l'observé (sous glace) : 25 à 64 % des jours de décembre à mars contre 5 à 31 % observés, mêmes stations et mêmes dates dans A et B. Plateaux d'ÉTÉ, propres à certaines solutions : SLSO-B plat 31,5 % des jours de mai à novembre contre 2,0 % observé (rapport 16), SAGU-B 34,9 contre 5,2, GASP-A 24,6 contre 7,9 ; SLSO-A et SAGU-A presque normaux (6,8 et 16,7 %). Le plateau d'été n'est jamais constant : il dérive avec une constante de temps de 300 à 950 jours, monte un jour sur trois, se tient au dixième centile du débit d'été, à 0,9 fois la médiane annuelle et 1,4 à 2,6 fois le minimum d'hiver. Pendant un plateau, une crue observée de 45 à 88 % en un jour fait bouger le simulé de 0,2 à 1,6 % ; hors plateau le même modèle répond dans 58 à 81 % des cas. Sur SLSO-B, 21 % des crues d'été tombent dans ce silence, contre 0,2 % sur SLSO-A. Indépendant des lacs (corrélations 0,04 et 0,08 sur 184 couples), de l'aire et du nombre de tronçons. Régionalement synchrone (26 % des jours d'été avec plus de la moitié des stations plates en SLSO-B, 0 % en SLSO-A). Deux entraînements de la même recette produisent ou non le régime : c'est une solution atteinte par l'optimisation.

Mécanisme (agent 2). Sous le socle, le drainage de la troisième couche est linéaire, constante de temps 1/krec : 417 jours à la borne haute, 2083 à la référence. Le ruissellement hortonien journalier est nul dès que K_sat_1 dépasse 0,03 m/j (départ 0,3). Le seul chemin latéral rapide, l'hypodermique de la deuxième couche, vaut K_sat_2 fois l'humidité à la puissance 10,9 : il s'éteint dès que la couche sèche ou que K_sat_2 est appris bas. Une pluie de 30 mm percole en un jour jusqu'à la troisième couche, y comble le déficit de l'ETR (75 % prélevée là, racines 1,5 m contre 0,38 m pour les deux couches du dessus), et le débit ne voit que +2,5 % de drainage, lissé sous 1 % par la nappe et le noyau. Leviers appris : krec (niveau du plateau, 0,5 à 2,5 mm/j), K_sat_2 et porosités (hypodermique), K_c jusqu'à 1,5 fois une Linacre déjà calée, seuil de stress et alpha d'assèchement non borné. Le correctif de drainage non linéaire (R34, R37, `l3_drain_exp`) n'est pas dans le socle. Hors de cause : VSA et hortonien sous-journalier (absents), milieux humides (non appris), nappe (4 à 33 jours, transmet).

Perte (agent 4). Aucun terme actif ne punit un plateau d'été : la MSE, le log-MSE et le terme pics ont pour optimum, sous un décalage d'un jour, l'espérance conditionnelle lissée ; le terme pics ne voit que les jours observés au-dessus du 75e centile, donc jamais l'été ; le PBIAS par bloc de 34 jours vaut zéro pour un simulé plat à la moyenne du bloc ; le prior tire K_sat_1 vers 0,08 m/j et krec vers 2e-5 m/h, plus d'infiltration et plus de chemin lent ; les lacs n'ont ni prior ni ancre et un taux 50 fois plus grand. Le seul contrepoids, gamma, était inerte (R82). Reconstruction de la perte de la flotte : les deux termes GRACE pesaient environ 84 % du total dans les régions où ils sont actifs (leur composante est accumulée sans le poids de bloc, d'où des affichages à 6000 %), avec un écart RMS de 160 mm entre stockage simulé et GRACE, cible insatisfaisable (R47) ; c'est là que la validation s'est effondrée. Le garde-fou de divergence exige un saut à six fois l'EMA en une époque et ne voit pas une montée de 1,27 à 3,08 en cinq (dette 23).

Verdict de forme (auditeur externe, mêmes dumps) : quatorze régions sur quatorze REFUSÉES ; Gaspésie, meilleur KGE de la flotte (0,819), est une calamité masquée : son score vient du point de reprise de l'époque 0, l'entraînement s'étant effondré à l'époque 1, et ce point produit 115 jours plats d'affilée en hiver et 40 suites de plus de 30 jours sur 15 stations.


## R84 — La colonne n'est pas convergée : le plafond de 64 sous-pas fabrique les deux tiers du ruissellement de surface (2026-09-05)

**Statut : établi par mesure, non corrigé.** Le schéma de sol boucle en sous-pas de Courant et, quand le plafond `MEANDRE_NSUBSTEP` est atteint avant la fin de la journée, la branche de fermeture de masse verse en ruissellement de surface toute la pluie non traitée. Mesure sur huit nœuds gaspésiens ancrés, été 2013, colonne isolée, mêmes paramètres, seul le plafond change :

```
sous-pas   surface   hypodermique   nappe   débit
    64       65 %        21 %        14 %   1,48 mm/j
   512       11 %        73 %        16 %   1,62 mm/j
```

La recette du socle pose 64. La partition entre chemin rapide de surface et chemin hypodermique, qui gouverne la forme de l'hydrogramme, est donc à 54 points près un artefact numérique, et non la physique d'Hydrotel. Le débit total ne bouge que de 9 %, ce qui explique qu'aucun bilan de masse ni aucun KGE ne l'ait signalé. Conséquences à examiner : la fidélité au binaire C++ n'a jamais été vérifiée à ce plafond sur une région entière ; l'apprentissage de K_sat_1 interagit avec le plafond, puisqu'une conductivité plus forte raccourcit le pas de Courant et fait déborder la boucle plus tôt.

## R85 — Deux entraînements de la même recette diffèrent d'un facteur 40 sur la conductivité qui commande l'écoulement hypodermique (2026-09-05)

**Statut : établi.** Comparaison des champs de paramètres des deux points de reprise du Saint-Laurent sud de la flotte du 4 septembre, évalués sur les 2889 nœuds de la région : A (boucle corrigée, plateaux d'été sur 6,8 % des jours) contre B (ancienne boucle, plateaux sur 31,5 % des jours). Médianes par nœud :

```
paramètre        A          B         B/A
K_sat_2       10,8 m/j   0,275 m/j    0,025
theta_wp_1    0,0033     0,092        28
k_gw          0,0033 /j  0,0163 /j    4,9
theta_fc_1    0,062      0,273        4,4
krec          5,6e-6     1,65e-5      2,9
K_musk        6,6 h      16,8 h       2,5
porosity_1    0,200      0,495        2,5
porosity_3    0,200      0,502        2,5
K_c           1,489      1,18         0,79
```

Les vingt autres paramètres diffèrent de moins de 20 %. Deux lectures. Première : la solution à plateaux a exactement la signature que la relecture de la colonne prédisait, une conductivité de deuxième couche quarante fois plus faible, donc un écoulement hypodermique éteint, et un drainage de troisième couche trois fois plus rapide, donc un débit d'été porté par le seul réservoir lent. Seconde : la solution sans plateaux n'est pas pour autant physique, ses porosités sont collées à leur borne basse (0,200), son coefficient cultural à sa borne haute (1,489 pour une borne à 1,5) et son coefficient de Muskingum de forme à 0,449 pour une borne à 0,49. Ce n'est PAS de l'équifinalité au sens strict, correction du 2026-09-05 : les fiches d'exécution des deux points de reprise montrent que les bras diffèrent par deux réglages du protocole d'optimisation, `MEANDRE_PAS_PAR_BLOC` et `MEANDRE_HISTORIQUE_AMORCE`. Deux protocoles conduisent à deux mondes paramétriques, ce qui est plus grave qu'un tirage : le protocole d'optimisation choisit la physique. Aucune contrainte de la perte ni du prior ne les sépare, et le collage aux parois est mesuré des deux côtés : A a ses porosités à moins de 2 % de la borne basse sur 2889 nœuds sur 2889 et son coefficient cultural à plus de 98 % de la borne haute sur 2870 nœuds ; B a sa conductivité de deuxième couche à moins de 2 % de la borne basse sur 2170 nœuds.

**Le fait le plus grave n'est pas l'écart, c'est le classement.** Sur la période d'évaluation, le monde à plateaux l'emporte : KGE médian 0,665 contre 0,414. La métrique de sélection ne tolère pas les plateaux, elle les préfère. Nuance nécessaire : le monde sans plateaux est trop nerveux (rapport des coefficients de variation 1,575 contre 0,813 pour le monde plat), donc le score choisit le moins mauvais des deux, et non le plateau pour lui-même. Il reste qu'aucun terme de la perte ne mesure le chemin de l'eau, seul discriminant physique entre les deux.

**Ce que le banc de colonne ne reproduit pas.** Sur la colonne isolée, faire varier krec sur trois ordres de grandeur et diviser K_sat_2 par mille change la part de nappe de 0 à 67 % et la part de jours plats de 0 à 26 %, mais ne produit jamais de suite plate de plus de cinq jours : la production de surface répond à chaque pluie. Le plateau de trente à cent quarante jours de la flotte demande donc quelque chose de plus que ces deux paramètres, à chercher du côté du sol calé du socle, du réseau et des lacs, ou de l'interaction avec R84.


## R86 — Les plateaux d'été ont DEUX causes distinctes, et elles ne sont pas les mêmes selon le modèle (2026-09-05)

**Statut : établi sur le sous-bassin de la station 022704 du Saint-Laurent sud, 73 tronçons, 796 km², 4 lacs, été 2022-2024.** Le même sous-bassin est rejoué avec trois jeux de paramètres, et l'on mesure séparément la platitude de la PRODUCTION de la colonne, somme des trois chemins avant réseau et lacs, et celle du DÉBIT ROUTÉ à l'exutoire.

```
                production plate   débit routé plat   observé
zéro époque        2,7 % (2 j)       14,0 % (5 j)     1,7 % (1 j)
point A            4,1 % (3 j)        3,6 % (3 j)     1,7 % (1 j)
point B           17,5 % (7 j)       12,9 % (6 j)     1,7 % (1 j)
```

Deux causes indépendantes. Au départ, la colonne produit un signal qui varie normalement et c'est le ROUTAGE qui multiplie la platitude par cinq : lacs et Muskingum. Dans le point de reprise à plateaux, c'est la COLONNE elle-même qui est plate à la source, et le routage n'y ajoute rien. Le point de reprise sans plateaux ne souffre d'aucun des deux, mais pour une autre raison, il est trop nerveux (gamma 1,575 contre 0,813).

**Partition des chemins d'eau en été, même sous-bassin**, qui donne la signature physique de chaque solution :

```
                surface   hypodermique   nappe   KGE    gamma   débit d'été
zéro époque       11 %        63 %        26 %   0,199  0,423   13,7 m³/s
point A           83 %        15 %         3 %   0,051  1,575   10,9 m³/s
point B           22 %        24 %        54 %   0,414  0,813    7,8 m³/s
```

Trois régimes hydrologiques incompatibles, issus de la même recette, des mêmes données et de la même graine, séparés seulement par la boucle d'optimisation. Le point A, dont la conductivité de deuxième couche est quarante fois plus grande (R85), envoie 83 % du débit d'été par le ruissellement de surface : c'est le plafond de sous-pas de R84 qui le lui permet, une conductivité forte raccourcissant le pas de Courant jusqu'à faire déborder la boucle, qui verse alors la pluie non traitée en surface. Le point B, conductivité quarante fois plus faible, éteint l'hypodermique et fait porter 54 % du débit d'été par la nappe : c'est le régime de plateau. Aucun des trois ne ressemble à l'autre, et l'observé n'est reproduit par aucun (KGE 0,05 à 0,41 sur ce sous-bassin difficile).

**Rétractation partielle de R83.** L'affirmation « les plateaux d'été sont créés par l'apprentissage » venait du sous-bassin gaspésien 021702, où le modèle non entraîné est effectivement propre (8 à 11 jours de suite maximale). Elle ne vaut pas partout : sur 022704, le modèle non entraîné présente déjà 20,6 % de jours plats sur l'année et une suite de 43 jours, par le routage. L'apprentissage déplace la cause de la platitude du routage vers la colonne, il ne la crée pas de rien.


## R87 — Le plateau survit à la convergence numérique : R84 est réel mais n'est pas la cause (2026-09-05)

**Statut : établi, réfute l'hypothèse numérique.** Même point de reprise à plateaux, même sous-bassin (station 022704, 73 tronçons), mêmes paramètres appris, seul le plafond de sous-pas change. Été 2024 :

```
sous-pas   surface  hypodermique  nappe   production plate   débit plat
   64        13 %       17 %       70 %    22,3 % (7 j)       14,0 % (6 j)
  256        11 %       19 %       70 %    19,8 % (7 j)       14,9 % (6 j)
```

Et sur le modèle non entraîné du même sous-bassin, la partition ne bouge pas non plus (hypodermique 59 % dans les deux cas). Le plafond de sous-pas déforme la partition dans le régime de crue saturée (R84, mesuré sur la colonne isolée et confirmé indépendamment : ruissellement de 93 mm à 64 sous-pas contre 44 mm à 256 pour 120 mm de pluie), mais il ne fabrique pas les plateaux d'été et ne change pas le chemin de l'eau à l'échelle du sous-bassin. Refaire la flotte à 256 sous-pas ne réglera pas la platitude. R84 reste un défaut de fidélité à corriger pour la crue, pas une cause de la platitude.

**Ce qui reste, par élimination.** La platitude du point de reprise à plateaux est portée par ses paramètres appris : 70 % du débit d'été par la nappe contre 32 % au départ, l'écoulement hypodermique passant de 59 % à 17 %. Le déplacement du chemin de l'eau est l'objet même de l'apprentissage, et rien dans la perte ne le mesure.


## R88 — GRACE est innocentée : l'éteindre dégrade la Gaspésie (2026-09-05)

**Statut : établi, réfute l'hypothèse du 5 septembre au matin.** Paire de contrôle sur la Gaspésie, MOD16 gardée en tendance, les deux termes GRACE éteints, tout le reste identique au candidat de livraison. Tenu de côté 2022-2024, quinze stations :

```
                        KGE médian   platitude   été       suite plate   pointes
avec GRACE, bras B         0,810      25,0 %     10,1 %      74 j         0,95
sans GRACE, bras B         0,672      34,6 %     42,9 %      89 j         1,22
avec GRACE, bras A         0,771      22,6 %     15,8 %      78 j         0,88
sans GRACE, bras A         0,810      32,2 %     32,0 %      84 j         0,85
```

Éteindre GRACE coûte 0,14 de KGE au bras B et quadruple sa platitude d'été. La présomption tirée de la coïncidence géographique des effondrements du 4 septembre était fausse : GRACE tient la forme au lieu de la casser, et les journaux le disaient déjà, six des neuf effondrements ayant eu lieu dans des régions où GRACE est éteinte. La cause des effondrements est le protocole d'optimisation, pas la contrainte de stockage.

## R89 — Le KGE n'est pas nécessaire dans la perte, et il coûte des pointes (2026-09-05)

**Statut : établi sur un sous-bassin, à confirmer à l'échelle provinciale.** Neuf pertes comparées sur le sous-bassin gaspésien 021702, même graine, même protocole, six époques de vingt-quatre pas. Référence à zéro époque : KGE 0,834, gamma 0,917, pointes 0,83 de l'observé.

```
perte                                         KGE    gamma  pointes  suite plate
socle actuel (KGE 1, biais 0,5, MSE 0,1, ET) 0,813   0,876   0,76       8 j
socle + variations journalières              0,821   0,891   0,82       9 j
socle + variations + soutien d'étiage        0,821   0,887   0,82       7 j
sans KGE, avec les deux termes de forme      0,820   0,888   0,84       9 j
forme seule (biais, ET, variations, étiage)  0,834   0,928   0,91       5 j
KGE + variations seuls                       0,818   0,963   0,89       9 j
```

La perte sans KGE ni erreur quadratique est la seule qui ne dégrade pas le KGE en six époques, la seule dont les pointes dépassent celles du modèle de départ, et elle a la plus courte suite plate. Le socle actuel perd sept points de pointes. Le rôle du KGE dans la perte se réduit donc à ce que le biais de volume fait mieux et sans rabotage.

**Réserve.** Ce sous-bassin ne présente pas la maladie de platitude d'étiage : son simulé est moins plat que l'observé, 4 à 8 % contre 9,3 %. Le balayage mesure le rabotage, pas les plateaux.

## R90 — Le terme de variations en valeur absolue est aveugle à un étiage figé (2026-09-05)

**Statut : établi par construction et par mesure, corrigé.** L'écart-type des variations journalières est dominé par les crues. Cas de contrôle, trois cent soixante-cinq jours, cent jours d'étiage figés, crues intactes :

```
                              étiage figé   pointes rabotées de moitié
terme en valeur absolue          0,00000            0,268
terme en logarithme              0,00285            0,015
```

Le terme absolu ne voit rien quand l'étiage est gelé, ce qui est précisément le défaut visible sur les hydrogrammes du rapport, dont les pointes sont justes et les basses eaux plates. Le terme logarithmique donne le même poids à une variation de 5 % quel que soit le niveau du débit. Les deux sont complémentaires : l'absolu garde les pointes, le logarithmique garde les basses eaux. Levier `ETL_WDQ` et `ETL_WFDC` au pilote, options `--w-dq`, `--w-dq-log` et `--w-fdc` au banc.

## R91 — Les prélèvements et rejets de deux ingestions étaient empilés dans les bases (2026-09-08)

**Statut : établi, corrigé.** `BasinCache.import_withdrawals` insère sans jamais vider la table. Les deux ingestions successives de la donnée d'io-eau, celle de la copie locale du 15 avril et celle de la source du 10 juin, coexistaient donc, sur des tronçons différents. Signature décisive : les nœuds surnuméraires s'arrêtaient tous au 1er décembre 2024, date de fin exacte de la copie d'avril, tandis que les nœuds à jour allaient au 1er février 2026, date de fin de la source de juin. Aucun nœud de la source n'était absent des bases : c'était un surplus, pas un manque.

```
région   nœuds périmés   amplitude périmée   nœuds à jour   amplitude à jour
mont          33             39,0 m³/s           217           39,1 m³/s
outv          83              1,5 m³/s           215            8,1 m³/s
slso          27              1,2 m³/s           444            9,5 m³/s
```

En Montérégie, le flux anthropique vu à l'entraînement valait donc le double du flux réel, dont la moitié sur de mauvais tronçons. Le cas le plus visible est l'ancien appariement de l'émissaire de la rive sud, 25,2 m³/s déversés sur un cours d'eau au débit naturalisé de 0,34 m³/s, soit 7192 % : la correction de juin avait déplacé cet émissaire sur MONT00002 en AJOUTANT la ligne juste, sans retirer la fausse. Toutes les flottes entraînées avant le 8 septembre ont vu cette donnée corrompue, ainsi que les couches anthropiques de la carte.

**Correctif** : la table est vidée avant réimportation dans `ingest_withdrawals.py`. Réingestion des quinze régions faite le 8 septembre : zéro nœud orphelin restant, toutes les régions couvrent 2001-01-01 à 2026-02-01.

**Deux défauts distincts trouvés au passage.** La région de l'Outaouais moyen n'avait AUCUNE table de prélèvements : ses 2379 nœuds ont été entraînés en régime naturalisé alors que leurs débits observés sont influencés, et son signal anthropique cartographié était un zéro silencieux. Elle est ingérée, 98 sites et 46 nœuds. Et un site demeure douteux dans la source elle-même, à signaler à io-eau : X2003275, prélèvement de surface de 0,76 m³/s attribué à MONT00574 dont le débit naturalisé vaut 0,02 m³/s, soit 3858 %.

## R92 — La cause du gradient non fini : la perte posait des NaN dans le tenseur simulé (2026-09-08)

**Statut : établi par le mode anomalie de PyTorch, corrigé, test de non-régression écrit.** Le chantier ouvert depuis le 4 septembre est clos.

Pour ignorer les observations manquantes, la branche par station de `HydroLoss` recopiait le débit observé ET le débit simulé, puis posait `float("nan")` aux pas de temps invalides des DEUX tenseurs, avant de réduire par `torch.nanmean` et `torch.nansum`. Ces réductions ignorent les NaN à l'aller, ce qui explique que la perte, les débits simulés et les composantes affichées soient toujours restés finis. Leur rétropropagation, elle, rend un gradient NaN, et ce gradient remonte dans le tenseur simulé, donc dans tout le champ spatial. Trois termes étaient touchés : l'erreur quadratique, le biais de volume et l'erreur quadratique logarithmique.

**Diagnostic.** Le mode anomalie posé sur la seule passe arrière nommait l'opération, `PowBackward0`, sans sa ligne. Posé aussi sur la passe avant, il désigne `loss.py:843`, c'est-à-dire l'élévation au carré de l'écart de la MSE. Les quatre familles de puissances de la physique, conductivité de Campbell, succion du sol, vidange des lacs et drainage de la troisième couche, ont toutes un plancher sur leur base et sont hors de cause.

**Ampleur.** Sur la flotte du 6 septembre, comptée par le message du filet : 2000 blocs jetés sur le Saint-Laurent sud, 1760 en Gaspésie, 1380 sur la Côte-Nord est, 1060 en Montérégie et en Outaouais, sur environ 4400 pas d'optimisation possibles. Les régions aux séries d'observations complètes, Saguenay, Côte-Nord B et C, lac Abitibi et Vaudreuil, n'en ont jeté aucun. La quantité de blocs jetés suit donc les lacunes d'observation, ce qui est la signature attendue de ce défaut. Près de la moitié des pas d'optimisation ont été perdus dans les régions les plus touchées, sans que rien ne le signale hors du compteur du filet.

**Correctif.** Le masque devient multiplicatif : les observations manquantes sont remplacées par zéro, l'écart est multiplié par un masque binaire, et la somme est normalisée par le nombre de pas valides. La valeur rendue est identique à celle de l'ancienne moyenne sur les jours observés, seul le gradient change. Le tenseur simulé ne reçoit plus jamais de NaN. Test de non-régression dans `tests/test_training/test_gradient_nan_obs.py` : gradient fini pour les trois termes avec une lacune partielle et une station entièrement absente, et égalité des valeurs avec la moyenne sur les jours observés.

**Ce que ce défaut n'a PAS fait.** Il n'a pas faussé les valeurs de perte ni les scores, qui étaient calculés correctement. Il a supprimé des pas d'apprentissage. Le filet par bloc écrit le 4 septembre, qui jetait ces blocs au lieu de perdre l'époque entière, a donc protégé les entraînements sans jamais nommer leur cause.

## R93 — Le fichier de prélèvements ajoute trois fois plus d'eau qu'il n'en retire, et un tiers de cet ajout est reconstruit (2026-09-10)

**Statut : établi.**

**Constat.** Sur `io-eau-meandre.parquet`, en débit moyen par site sur 2001-2024, les prélèvements déclarés retirent 65,1 m³/s, les rejets déclarés en ajoutent 71,9, et des entrées reconstruites en ajoutent 53,9 de plus. Le solde est un ajout net de 60,7 m³/s au réseau hydrographique du Québec méridional.

**Les entrées reconstruites.** Elles se reconnaissent au suffixe `_synth` de leur identifiant. Le fichier en compte 1851, et elles sont TOUTES positives : aucune n'est un prélèvement. Chacune est placée sur le tronçon même du prélèvement qu'elle accompagne, ce qui est vrai des 1851 sans exception. Elles couvrent 94 pour cent du volume prélevé, soit 61,0 des 65,1 m³/s, avec un taux de retour médian de 0,88 et pondéré de 0,883. Un prélèvement ne retire donc à son tronçon que 12 pour cent de son volume.

**Conséquence sur les résultats de naturalisation.** C'est l'explication du contraste mesuré sur les caches `nb-<reg>-{avec,sans}.npz` : sur 14 061 tronçons dont le débit naturalisé dépasse 1 m³/s, 328 voient leur débit augmenté de plus de 1 pour cent et 34 seulement le voient diminué d'autant. L'effet simulé des prélèvements est faible parce que le fichier leur rend 88 pour cent de leur volume au même endroit, pas parce que le modèle l'amortit.

**Ce que ce constat RÉFUTE.** L'hypothèse selon laquelle le motif d'une nappe en déficit net et d'un réseau de surface en excédent net serait la signature observée d'une eau prise en profondeur et rendue en surface, dont l'exhaure minière. Ce motif est en grande partie CONSTRUIT : 1072 des 1851 couples convertissent un prélèvement souterrain en rejet de surface, et cette conversion est opérée par la règle de reconstruction, pas par une déclaration. L'exhaure reste une explication plausible d'une partie des rejets déclarés, mais ce motif ne l'établit pas. Une note écrite le 2026-09-09 dans la présentation affirmait le contraire et a été corrigée.

**Ouvert : le double comptage.** L'eau distribuée par un réseau d'aqueduc est prélevée, reçoit un retour reconstruit sur le tronçon du prélèvement, puis ressort à la station d'épuration qui déclare son propre rejet. Si les deux mécanismes portent sur le même mètre cube, il est rendu deux fois. Les couples surface vers surface pèsent 47,2 m³/s de retours reconstruits, à comparer aux 71,9 m³/s de rejets déclarés. Test : comparer le rejet déclaré d'une station d'épuration au retour reconstruit attribué aux prélèvements de son territoire desservi.

**Exposition des stations d'entraînement (2026-09-10).** Sur les 178 stations des quatorze régions, l'effet anthropique simulé sur leur propre tronçon vaut : 132 sous 0,5 pour cent en valeur absolue, 18 entre 0,5 et 1 pour cent, 15 entre 1 et 5, et 8 au-delà de 5. Cinq stations seulement subissent un effet négatif au-delà de 0,5 pour cent, aucune au-delà de 5. La dissymétrie se retrouve donc jusque dans les données d'entraînement : 41 stations reçoivent plus de 0,5 pour cent d'eau ajoutée contre 5 qui en perdent autant. Les huit stations les plus exposées sont 030304 (+13,1 %), 030340 (+10,9 %), 021916 (+8,7 %), 030345 (+7,6 %), 02E901, 030262 et 030299 (+6,1 %, même tronçon) et 046404 (+5,8 %). Détail dans `exposition-stations.csv`.

**Conséquence pour le calage.** Pour 132 stations sur 178, le terme anthropique est trop faible pour peser sur les paramètres appris. Pour les 25 dont l'effet dépasse 1 pour cent, le modèle se voit retirer de l'eau du débit observé avant comparaison ; si les retours reconstruits sont surestimés, il doit produire davantage d'écoulement naturel pour compenser, ce qui biaise ses paramètres vers un territoire plus humide. Ce biais est donc localisé, pas systémique.

**Vérifié aussi.** L'ingestion ne filtre pas ces lignes : `.runs/quebec/ingest_withdrawals.py` somme `net_withdrawal` par tronçon sans regarder l'identifiant. Les entrées reconstruites sont donc bien dans les bases de bassin et dans tous les entraînements.

## R94 — L'export des quantiles de la Gaspésie ne provient pas du modèle présenté (2026-09-10)

**Statut : établi. Le diagramme de Talagrand est retiré de la présentation.**

**La règle.** La tête de quantiles est apprise sur un modèle physique GELÉ, et sa médiane est le débit du modèle déterministe. Le contrôle qui en découle, posé par Essi, est que le débit médian de l'export probabiliste doit être identique à celui du modèle déterministe présenté, au millionième près, avant tout diagramme.

**Le contrôle échoue.** `quant-gasp.npz` comparé aux trois exports déterministes disponibles pour la même région, mêmes seize stations dans le même ordre, mêmes 1096 jours de 2022-2024, mêmes observations au bit près :

| export déterministe | écart relatif médian | écart relatif maximal |
| --- | --- | --- |
| q-gasp-A-v4.npz | 0,419 | 7,06 |
| q-gasp-B.npz | 0,458 | 21,98 |
| q-gasp-A.npz | 0,446 | 19,59 |

Aucune valeur journalière ne coïncide à 1e-6 près, à aucun niveau de débit : l'écart relatif médian vaut 0,39 sur le décile inférieur et 0,36 sur le décile supérieur. La corrélation des séries simulées va de 0,66 à 0,90 selon la station. Les KGE par station diffèrent nettement, de 0,507 contre 0,814 à la station 011509 et de 0,776 contre 0,595 à la station 022507.

**Le piège à éviter.** Les KGE MÉDIANS des deux exports coïncident à 0,715, et les débits moyens à 0,3 pour cent près. Un contrôle porté sur le score médian aurait donc conclu à tort que le socle était préservé. Seule la comparaison valeur par valeur détecte le défaut. C'est la raison d'être de la règle du millionième.

**Ce que valait le diagramme.** Sur 16 246 observations, la couverture de l'intervalle à 90 pour cent valait 89,6 pour cent, apparemment excellente, mais elle résultait de deux erreurs opposées : aucune observation ne descendait sous le cinquième centile prédit, contre 5 pour cent attendus, et 10,4 pour cent dépassaient le quatre-vingt-quinzième, contre 5 attendus. Ce diagnostic reste sans objet tant que le modèle sous-jacent n'est pas identifié.

**Correctif livré.** Le contrôle est écrit DANS la diapositive de `.reports/quebec/presentation.qmd` : chaque région dont l'export de quantiles s'écarte du déterministe est écartée, et si aucune ne passe, la diapositive imprime la raison du refus au lieu d'un diagramme. La présentation ne peut donc plus afficher un diagramme de Talagrand tracé sur un autre modèle.

**Cause identifiée le 2026-09-10.** Les neuf configurations de phase probabiliste, `.runs/quebec/config/<region>-quantile.toml`, portent toutes `warm_start_from = "checkpoints/best-<region>.pt"`. Aucune ne désigne le point de reprise de la flotte, `best-<region>-etl-fdsA-v4.pt`. La tête de quantiles a donc été apprise par-dessus un autre modèle physique que celui présenté, et le défaut est systématique, non propre à la Gaspésie. La configuration de la Gaspésie porte de surcroît deux commentaires contradictoires sur cette même ligne, l'un annonçant un départ à chaud et l'autre un départ à froid.

**Correctif à appliquer.** Faire pointer `warm_start_from` sur le point de reprise effectivement présenté, puis réexporter les quantiles. Le contrôle du millionième, désormais écrit dans la diapositive, dira si la correction tient. Tant qu'il n'est pas passé, aucun résultat probabiliste n'est présentable.


## R95 — Coût de la rétropropagation mesuré : 2,4 à 2,7 fois la passe avant (2026-09-10)

**Statut : établi.**

La présentation avançait « environ deux fois et demie le temps de la simulation » sans preuve. Mesure faite sur le lac Abitibi, 393 tronçons, recette du socle appliquée, routage par opérateur, carte NVIDIA RTX 2000 Ada. Deux temps sur le même nombre de pas de temps, appariés dans la même répétition, avec synchronisation de la carte avant chaque relevé et un tour de chauffe non compté : passe avant seule sous `torch.no_grad()`, puis passe avant avec graphe, perte quadratique et rétropropagation.

Sur 30 pas de temps et neuf répétitions, la passe avant seule prend 17,8 s en médiane et la passe complète 43,1 s, soit un rapport de 2,4 sur les médianes et de 2,7 en médiane des rapports appariés. Un contrôle sur 45 pas de temps, la taille de bloc réellement utilisée à l'entraînement, rend 2,42.

**Réserve.** Un autre processus occupait la carte pendant la mesure, ce qui disperse les temps absolus d'un facteur deux d'une répétition à l'autre. Le rapport, apparié dans chaque répétition, n'en souffre pas.

**Conclusion.** L'affirmation de la présentation est confirmée et y est maintenue.

## R96 — Le seuil de dé-crachinage de 0,3 mm/h n'est dérivé de rien (2026-09-10)

**Statut : ouvert. Question posée par Essi, vérification faite dans le dépôt.**

**Ce que le seuil fait.** `dedrizzle` met à zéro toute heure de précipitation d'intensité inférieure au seuil. Mesuré aujourd'hui sur une maille CaSR du Saint-Laurent sud-ouest, quatre années horaires : le seuil de 0,3 mm/h touche 85 pour cent des heures où il tombe quelque chose, et ces heures portent 24 pour cent du volume. Le journal enregistre l'effet correspondant sur le Saint-Laurent nord-ouest, la fraction de jours pluvieux passant de 62 à 40 pour cent.

**D'où vient la valeur.** De nulle part de traçable. C'est la valeur par défaut du paramètre `threshold_mm_h` de `meandre/data/forcing_correction.py`, reprise telle quelle par les deux constructeurs de forçage, où elle est exposée sous `DRIZZLE_H` sans jamais avoir été balayée. Aucun rapport ne consigne d'essai à une autre valeur, ni de critère qui aurait servi à la choisir.

**Ce que la littérature citée appuie, et ce qu'elle n'appuie pas.** La revue du prétraitement météorologique cite Lavers et ses collaborateurs, 2022, pour ERA5 et Lespinas, 2015, pour CaPA. Ces travaux établissent l'EXISTENCE d'un biais de bruine, précipitation trop fréquente et trop faible, jours humides surestimés. Aucun ne fixe la valeur du seuil qui le corrige. Le dé-crachinage est donc appuyé dans son principe et arbitraire dans son réglage.

**Le seul contrôle observationnel disponible ne le soutient pas.** La grille krigée des stations du ministère, 2889 nœuds sur 2000-2024, porte 61,4 pour cent de jours au-dessus de 0,1 mm par jour, soit la même fréquence que CaSR brut. Le dé-crachinage éloigne donc CaSR de cette référence au lieu de l'en rapprocher. Réserve importante : le krigeage étale la pluie sur tous les nœuds et gonfle de lui-même le compte des jours humides, si bien que cette grille est un mauvais juge de la FRÉQUENCE, même si elle reste un bon juge du VOLUME de bassin. La question reste donc ouverte, elle n'est pas tranchée contre le seuil.

**Pourquoi cela compte.** Les 24 pour cent de volume retirés sont rendus ensuite par le calage annuel sur le bilan de Budyko. Le seuil ne change donc pas le volume final, il change la DISTRIBUTION TEMPORELLE de la pluie, en concentrant la même eau sur moins d'heures. C'est exactement la grandeur qui gouverne le ruissellement hortonien et les pointes de crue. Un réglage non dérivé pilote une propriété de premier ordre.

**Le test qui trancherait.** Comparer la fréquence de jours pluvieux à des séries de stations PONCTUELLES, non krigées, aux nœuds qui les portent, et balayer le seuil pour trouver celui qui reproduit cette fréquence. À défaut, balayer le seuil et juger sur les signatures de crue, jamais sur le volume annuel, qui est insensible au seuil par construction.

## R97 — Ce que le calage du volume par le bilan suppose vraiment, et ce qu'il coûte (2026-09-10)

**Statut : établi. Objection soulevée par Essi, quantifiée.**

**Objection.** « On prend un débit médian aux stations, qui ne sont aucunement représentatives des débits dans le réseau, et on fait un bilan là-dessus ? »

**Première réponse : ce n'est pas un débit.** `rescale_forcing_budyko.py` divise le débit moyen de chaque station par son aire drainée et convertit en millimètres par année. La grandeur médiane est donc une LAME ÉCOULÉE, normalisée par l'aire, et non un débit. C'est ce qui rend comparables un bassin de 27 km² et un de 15 515 km² du Saguenay. La médiane porte sur des lames, pas sur des débits.

**Deuxième réponse : ce que la correction fixe.** Un seul scalaire par région, la précipitation annuelle moyenne. La grille CaSR est multipliée par ce rapport sans que sa structure spatiale ni sa chronologie ne changent. L'hypothèse n'est donc pas que les stations représentent le réseau tronçon par tronçon, mais que le bilan d'eau des bassins jaugés estime le bilan d'eau moyen de la région.

**Le choix de la médiane n'est pas le point faible.** Écart entre la médiane des lames et leur moyenne pondérée par l'aire drainée : 4,5 % en Gaspésie, 3,7 % au Saint-Laurent sud-ouest, moins de 3 % partout ailleurs, 0,1 % au Saint-Laurent nord-ouest.

**Les vrais points faibles, mesurés.**

Premier, le nombre de jauges. Six régions en portent deux ou moins. Toute la grille du bassin du lac Abitibi est recalée sur un seul bassin de 222 km², et celle de la Côte-Nord A sur un seul de 768 km².

Deuxième, le désaccord entre stations d'une même région. Étendue rapportée à la médiane, dans les six régions entraînées : 120 % au Saguenay, dont les lames vont de 162 à 976 mm par année, 96 % en Montérégie, 90 % au Saint-Laurent nord-ouest, 73 % en Outaouais, 61 % en Gaspésie, 60 % au Saint-Laurent sud-ouest. La médiane est un centre robuste, mais la dispersion autour d'elle est du même ordre que la grandeur estimée.

Troisième, la circularité anthropique. Le débit observé aux stations comprend les prélèvements et les rejets, donc un bilan d'eau NATUREL est fermé sur un débit GÉRÉ. Effet mesuré sur la lame médiane régionale, en corrigeant chaque station de l'effet anthropique simulé sur son propre tronçon : Montérégie −3,0 %, Saint-Laurent sud-ouest −1,3 %, bassin de l'Abitibi −1,2 %, Saint-Laurent nord-ouest −0,3 %, négligeable dans les dix autres régions. La précipitation corrigée est donc surestimée d'au plus 3 %, et seulement là où les rejets sont importants.

Quatrième, non mesuré ici mais structurel : la partie non jaugée d'une région, souvent plus au nord, reçoit la correction déduite de sa partie jaugée.

**Lien.** Cette correction est aussi ce qui PRESCRIT le niveau d'évapotranspiration du modèle, limite déjà consignée au point 25 de la liste des limites. Les deux se lisent ensemble : le bilan impose à la fois le volume de précipitation et, par conservation, le niveau d'évapotranspiration vers lequel l'entraînement converge.

## R98 — L'hiver observé sur lequel le modèle s'entraîne est lui-même une reconstruction, et elle est bien plus plate que les mesures (2026-09-10)

**Statut : établi.**

**Mesure.** Part des observations de décembre à mars, depuis 2001, portant le drapeau de reconstruction du Centre d'expertise hydrique, et platitude hivernale observée, définie comme la part des jours où le débit varie de moins de 1 pour cent d'un jour à l'autre, calculée sur tous les jours puis sur les seuls jours mesurés. Médiane par station.

| région | stations | reconstruit | platitude, tous jours | platitude, jours mesurés |
| --- | --- | --- | --- | --- |
| outv | 16 | 74,8 % | 14,6 % | 6,9 % |
| gasp | 18 | 84,1 % | 20,5 % | 6,6 % |
| sagu | 23 | 93,8 % | 36,4 % | 10,0 % |
| mont | 28 | 59,0 % | 4,5 % | 2,8 % |
| slno | 32 | 80,7 % | 15,7 % | 6,0 % |
| slso | 41 | 80,1 % | 7,3 % | 2,7 % |
| abit | 4 | 83,7 % | 38,2 % | 15,9 % |
| cnda | 1 | 94,3 % | 40,3 % | 10,6 % |
| cndb | 2 | 89,9 % | 23,2 % | 6,6 % |
| cndc | 2 | 48,8 % | 41,0 % | 26,1 % |
| cndd | 2 | 94,5 % | 51,5 % | 14,0 % |
| cnde | 2 | 92,4 % | 48,3 % | 12,3 % |
| labi | 1 | 92,6 % | 14,7 % | 7,4 % |

La base de l'Outaouais moyen ne porte pas le drapeau ; elle est hors tableau.

**Premier constat.** De 49 à 95 pour cent des observations hivernales ne sont pas des mesures. Sous couvert de glace, le débit est reconstruit. Cette reconstruction est deux à quatre fois plus plate que les jours réellement mesurés de la même station et de la même saison.

**Conséquence sur l'ENTRAÎNEMENT, et c'est la plus importante.** Les termes de débit de la fonction de perte traitent ces valeurs reconstruites exactement comme des mesures. Le modèle est donc entraîné, quatre mois par an et sur la grande majorité des jours de ces mois, à reproduire une courbe lissée par un procédé d'estimation. C'est un mécanisme candidat direct pour les plateaux d'hiver simulés, et il est extérieur au modèle.

**Conséquence sur le VERDICT DE FORME.** Le verdict compare la platitude simulée à la platitude observée, tous jours confondus. Cette référence étant gonflée par la reconstruction, le seuil est plus haut qu'il ne devrait, donc le verdict est plus INDULGENT qu'il n'y paraît. Les huit régions qui échouent échouent contre une référence déjà trop plate.

**Ce que ce constat ne dit pas.** Il ne dit pas que la reconstruction du Centre d'expertise hydrique est fausse : sous glace, un débit lisse peut être le bon. Il dit que le modèle apprend cette régularité comme si elle avait été mesurée, et qu'aucune des deux hypothèses n'a été testée.

**Test proposé, préalable à tout entraînement.** Une paire appariée sur un sous-bassin : perte de débit calculée sur tous les jours contre perte calculée en écartant les jours reconstruits, jugée sur la platitude hivernale simulée et sur le calendrier de la crue printanière. Le risque à mesurer est que l'exclusion laisse l'hiver presque sans contrainte, de 75 à 95 pour cent des observations disparaissant.

## R99 — Jugé sur des mesures réelles, le défaut de platitude hivernale disparaît dans sept régions sur huit (2026-09-10)

**Statut : établi. Il révise le verdict de forme et retire l'essentiel du grief des plateaux d'hiver.**

**Objection d'Essi qui a déclenché la mesure.** « Un masque dur ne risque-t-il pas de générer n'importe quoi en hiver, alors qu'on a au moins une approximation ? »

**Premier résultat : le masque dur est écarté.** Les jours reconstruits ne sont pas dispersés, ils forment des suites continues. Longueur médiane de 22 jours en Montérégie à 45 au Saguenay, neuvième décile de 45 à 62 jours, maximum de 122 jours, soit l'hiver entier. Le nombre médian de jours réellement mesurés par station et par hiver va de 2 au Saguenay et 6 en Gaspésie à 42 en Montérégie, et 5,3 pour cent des hivers du Saguenay n'en comptent aucun. Écarter les jours reconstruits des termes de débit laisserait donc l'hiver sans contrainte. L'objection est validée : la reconstruction porte une information de VOLUME qu'il ne faut pas jeter ; ce qu'elle ne porte pas, c'est une information de VARIABILITÉ.

**Second résultat, et c'est le principal.** Le verdict de forme comparait la platitude simulée sur tous les jours d'hiver à la platitude observée sur tous les jours d'hiver, référence gonflée par la reconstruction. Comparaison refaite à échantillon identique : uniquement les paires de jours calendaires CONSÉCUTIFS dont les deux observations portent le drapeau de mesure. Ce choix élimine l'artefact d'échantillonnage, une différence entre deux jours mesurés éloignés n'étant pas une variation d'un jour à l'autre.

| région | stations | paires | platitude simulée | platitude observée | rapport | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| outv | 12 | 86 | 11,1 % | 8,3 % | 1,34 | ok |
| gasp | 9 | 67 | 23,9 % | 7,5 % | 3,19 | REFUS |
| mont | 23 | 124 | 1,9 % | 1,9 % | 1,00 | ok |
| slno | 16 | 49 | 2,6 % | 6,5 % | 0,40 | ok |
| slso | 24 | 60 | 3,0 % | 2,2 % | 1,36 | ok |
| abit | 1 | 100 | 36,0 % | 36,0 % | 1,00 | ok |
| cndb | 1 | 49 | 8,2 % | 8,2 % | 1,00 | ok |
| cndc | 1 | 355 | 51,5 % | 31,5 % | 1,63 | ok |

Sur les huit régions comparables, une seule échoue, la Gaspésie. Le Saint-Laurent nord-ouest est même moins plat que ses observations. Sous l'ancienne comparaison, la Gaspésie, la Montérégie, le Saint-Laurent nord-ouest et le Saint-Laurent sud-ouest échouaient toutes les quatre.

**Ce que cela change.** Le grief des plateaux d'hiver visait en grande partie une comparaison contre une courbe lissée par un procédé d'estimation, et non un défaut du modèle. Un entraînement dédié à corriger l'hiver corrigerait donc surtout un artefact de mesure. La Gaspésie reste un cas réel à traiter.

**Ce que cela ne dit pas.** Rien sur l'été ni sur les longues suites plates hors hiver, que ce test ne couvre pas. Rien non plus sur les régions transférées à station unique, où le rapport porte sur une seule série. Et cela ne dit pas que le modèle est bon en hiver, seulement que sa platitude hivernale ne s'écarte pas de celle des mesures.

**Correctif à porter au verdict de forme.** Calculer la platitude observée et simulée sur les paires de jours consécutifs mesurés, et non sur tous les jours. Le verdict actuel, dans `.runs/quebec/etl_run.py`, compare des échantillons différents.

## R100 — La seconde règle du verdict de forme n'est PAS un artefact : les suites plates simulées sont réelles (2026-09-10)

**Statut : établi. Il corrige une conclusion trop favorable que j'avais tirée une heure plus tôt.**

Le verdict de forme refuse si l'une de DEUX règles se déclenche : la platitude simulée dépasse le double de l'observée, ou la plus longue suite de jours plats dépasse 30 jours. La révision de la comparaison à échantillon identique ne touche que la PREMIÈRE. J'avais écrit qu'une correction de l'hiver reviendrait surtout à corriger un artefact ; la mesure ci-dessous montre que c'est faux pour la seconde règle.

Plus longue suite de jours consécutifs où le débit varie de moins de 1 pour cent, médiane par station, sur la période d'évaluation, et part de jours reconstruits à l'intérieur de la suite simulée :

| région | suite simulée | suite observée | part reconstruite de la suite |
| --- | --- | --- | --- |
| outv | 18 j | 11 j | 0,0 % |
| gasp | 81 j | 25 j | 97,4 % |
| sagu | 116 j | 25 j | 98,1 % |
| mont | 20 j | 2 j | 95,6 % |
| slno | 46 j | 9 j | 97,3 % |
| slso | 46 j | 7 j | 0,0 % |
| abit | 21 j | 26 j | 40,9 % |
| cndb | 86 j | 25 j | 97,7 % |
| cndc | 68 j | 7 j | 0,0 % |

**Deux lectures, toutes deux défavorables au modèle.** D'abord, la suite simulée dépasse la suite observée dans huit régions sur neuf, et souvent d'un facteur trois à dix : le modèle est plus plat que la reconstruction elle-même, alors que celle-ci est déjà lissée. Ensuite, dans trois régions, l'Outaouais, le Saint-Laurent sud-ouest et la Côte-Nord C, la plus longue suite simulée tombe entièrement dans des périodes MESURÉES : 46 et 68 jours de débit constant y sont contredits par des mesures réelles.

**Ce que les deux constats donnent ensemble.** Le modèle reproduit correctement la petite variation d'un jour à l'autre là où l'on mesure, et il sait néanmoins tenir un débit constant pendant des semaines ailleurs. Le défaut n'est donc pas diffus, il est LOCALISÉ dans le temps. C'est une piste plus précise que « le modèle est plat » : il faut chercher ce qui gèle la production pendant des blocs de plusieurs semaines, pas ce qui amortit la variabilité en général.

**Correction de ma conclusion antérieure.** J'ai écrit qu'une v5 dédiée à l'hiver corrigerait surtout un artefact de comparaison. Cela vaut pour la première règle, pas pour la seconde. Le grief des longues suites plates tient.

---

## R101 — Le double comptage des rejets portait presque toute la dissymétrie du fichier de prélèvements (2026-09-15) — ÉTABLI

Essi a corrigé le 15 septembre un double comptage des rejets dans le dérivé d'io-eau. Ce que la correction déplace, mesuré sur la période 2001-2024 et par site.

| grandeur | avant | après |
|---|---:|---:|
| solde provincial | +62 m³/s | +9,0 m³/s |
| entrées reconstruites | 1 851 | 318 |
| part du volume prélevé couverte par un retour | 94 % | 2,7 % |
| taux de restitution médian | 0,88 | 0,16 |

**Ce qui est réfuté par cette correction.** La phrase qui disait qu'un prélèvement ne retire réellement que douze pour cent de son volume, le reste lui étant restitué sur le tronçon même, était un artefact du double comptage. Un prélèvement retire désormais l'essentiel de son volume, et la naturalisation repose sur des volumes déclarés des deux côtés du bilan plutôt que sur une règle de retour supposée.

**Ce qui reste, et devient l'explication entière de la dissymétrie.** Parmi les lignes déclarées, la surface est excédentaire de 15 m³/s et le souterrain déficitaire de 7 : de l'eau prise dans la nappe et rendue au cours d'eau. Le fichier porte l'origine de l'eau mais pas le secteur d'activité, donc l'attribution à un usage précis reste non vérifiable.

**Effet régional.** Au Saguenay, le terme net de surface change de signe, de +1,39 à -0,14 m³/s. En Gaspésie il passe de +2,90 à +1,65. Le terme souterrain ne bouge dans aucune région, ce qui confirme que le double comptage était entièrement dans les rejets de surface.

**Conséquence sur les résultats publiés.** Les quinze bases régionales sont réingérées. Les caches de naturalisation, donc la carte de l'effet simulé et les hydrogrammes gérés contre naturalisés, ont été refaits le 15 septembre. Les POIDS, eux, ont été ajustés avec les rejets doublés : l'écart de naturalisation reste valide puisque les deux passes partagent les mêmes poids, mais le calage a absorbé une part d'un flux fictif, et seul un recalage le corrigera. Toute conclusion tirée de la valeur ABSOLUE des paramètres de sol des champions actuels est donc à considérer comme caduque jusqu'à ce recalage.

---

## R102 — L'ancrage géométrique du routage corrige les pointes et dégrade tout le reste : les vingt-quatre heures étaient une béquille (2026-09-15) — ÉTABLI

Épreuve appariée sur Narval, tâche 3093071, six tâches terminées de 1 h 34 à 2 h 44, deux régions et trois bras, même graine, vingt époques. Témoin : bornes du temps de transfert 4, 48, 24 heures. Bornes : 0,2, 48, 2 sans ancre. Ancre : mêmes bornes avec K = L / c.

L'ancre s'est posée comme au contrôle : médiane 0,98 heure au Saguenay et 0,61 en Gaspésie, 348 et 63 lacs laissés libres, longueur de rivière médiane 5,7 et 3,6 kilomètres.

| région | bras | KGE médian | jours plats | été | suite plate | pointes sim/obs | q99 sim/obs |
|---|---|---:|---:|---:|---:|---:|---:|
| Saguenay | témoin | 0,778 | 28,6 % | 21,6 % | 96 j | 0,81 | 1,21 |
| Saguenay | bornes | 0,740 | 27,4 % | 17,5 % | 75 j | 1,08 | 1,17 |
| Saguenay | ancre | 0,701 | 46,1 % | 47,3 % | 116 j | 1,11 | 1,23 |
| Gaspésie | témoin | 0,810 | 25,6 % | 9,8 % | 74 j | 0,95 | 1,00 |
| Gaspésie | bornes | 0,558 | 28,4 % | 14,5 % | 93 j | 1,46 | 1,36 |
| Gaspésie | ancre | 0,595 | 17,6 % | 1,9 % | 79 j | 1,33 | 1,32 |

Observé : 14,1 et 12,1 pour cent de jours plats, 4,1 et 4,4 pour cent en été.

**Le mécanisme visé fonctionne.** Le rapport des pointes annuelles passe de 0,81 à 1,11 au Saguenay et de 0,95 à 1,33 en Gaspésie. Le rabotage est corrigé, et dépassé.

**Le modèle est pourtant plus mauvais partout.** Le KGE médian perd 0,077 au Saguenay et 0,215 en Gaspésie. Au Saguenay la platitude empire fortement, de 28,6 à 46,1 pour cent de jours plats et de 96 à 116 jours pour la plus longue suite, ce qui est l'inverse du but recherché.

**Erreur de méthode, celle-là même qu'une condition préalable du fichier de consignes interdit.** Le témoin doit porter le défaut à corriger. Or les pointes du témoin valaient 0,81 au Saguenay et 0,95 en Gaspésie, avec un quantile 99 à 1,00 : ces deux régions ne rabotaient presque pas, alors que la mesure régionale qui a motivé le chantier donnait 0,39 à 0,59. L'ancre ne pouvait donc que dépasser la cible.

**Ce que l'épreuve établit malgré cela, et c'est la conclusion utile.** Le temps de transfert de vingt-quatre heures n'était pas un défaut exploité par l'optimiseur, c'était une compensation. La mesure du 2026-09-14 donne une variabilité des variations d'un jour de 1,350 pour la production de la colonne contre 0,437 pour l'observé, le réseau en retirant 87 pour cent. Retirer l'atténuation du canal sans corriger la colonne expose le défaut au lieu de le guérir : les pointes montent au-dessus de un, le quantile 99 monte avec elles, la corrélation se casse et le score suit.

**Conséquence sur le chantier.** L'ancre n'est pas adoptée seule. La cible n'est pas le canal mais la réponse du versant, c'est-à-dire ce qui sépare la production de la colonne de l'entrée dans le canal. L'hydrogramme unitaire de versant, présent dans le dépôt et débranché, est le mécanisme prévu pour cela, et il est conçu pour aller avec un canal peu atténuant. La prochaine épreuve doit les juger ENSEMBLE, et sur une région dont le témoin porte réellement le défaut, ce que ni le Saguenay ni la Gaspésie ne font.

---

## R103 — Le correctif de la longueur des lacs, écrit le 3 septembre, n'a jamais atteint les données (2026-09-15) — ÉTABLI

Pour un tronçon de lac, la colonne de longueur de la table `edges` des bases régionales contient l'AIRE DU LAC en mètres carrés, verbatim : rapport à `lake_area_m2` de 1,0000 avec un écart-type nul sur les 348 arêtes lacustres du Saguenay. Un lac de 644 075 mètres carrés est donc lu comme un tronçon de 644 kilomètres.

Le code ne fait plus cette erreur. Le commit du 3 septembre, « le bloc métrique d'un lac est une surface, pas une longueur », pose la longueur d'un lac à la racine de sa surface, avec un commentaire qui annonce déjà que l'ancienne lecture donnait des centaines de kilomètres. Mais aucune base régionale n'a été reconstruite depuis, et `territorial-raw-QC.parquet`, qui porte les attributs provinciaux, date du 1er août.

**Ce que cela fausse aujourd'hui.** La distance à l'exutoire est cumulée à partir de cette colonne, et c'est l'un des seize canaux que le champ spatial consomme. Sa moyenne provinciale vaut 4 124 kilomètres et son maximum 17 218 dans l'Outaouais aval, alors que le plus long cours d'eau du Québec fait environ mille kilomètres. Recalculée en n'employant la colonne que pour les tronçons de rivière et la distance entre centroïdes corrigée de la sinuosité, 1,13, pour les lacs, elle donne une médiane de 252 kilomètres au Saguenay, 259 en Gaspésie et 394 dans l'Outaouais aval, avec des maximums de 756, 639 et 695. La corrélation de rang entre l'ancienne et la nouvelle vaut 0,876, 0,925 et 0,702 : sur l'Outaouais, l'ancienne mélange aussi l'ORDRE des tronçons, puisque chaque lac en amont ajoute sa surface au cumul. Le champ reçoit donc, sous l'étiquette d'une distance, une variable qui dit surtout quelle surface de lac se trouve en amont.

**Ce que cela n'a pas faussé.** L'ancrage géométrique du routage du 15 septembre écarte explicitement les lacs, après avoir mesuré que leur longueur médiane valait 461 kilomètres contre 5,7 pour un tronçon de rivière. Il était donc protégé par construction.

**Ampleur réelle de l'effet, mesurée et plus faible que je ne l'avais laissé entendre.** Le routage ne lit jamais la longueur d'arête : le temps de transfert Muskingum est prédit par le champ. Le délai entier en jours est bien calculé depuis la longueur, mais il n'est employé que par la voie d'attention sur le temps de parcours, désactivée partout sur la ligne québécoise. Reste la voie indirecte, par le canal d'attributs. Une fois les deux versions centrées et réduites, comme le champ les reçoit, elles sont corrélées en rang à 0,799 ; en substituant l'une à l'autre sur le point de reprise du Saguenay, l'écart relatif médian sur les quarante-deux paramètres prédits vaut 0,14 pour cent, et le temps de transfert Muskingum ne figure pas parmi les quatorze plus touchés. La seule exception notable est la neige : la température de fonte bouge de 17 pour cent et le facteur de fonte de 4,5, ce qui est cohérent puisqu'un tronçon éloigné de l'exutoire est aussi un tronçon d'amont, donc d'altitude.

**Le correctif reste à appliquer** parce qu'un canal faux reste faux et qu'il coûte quelques minutes, non parce qu'il tiendrait une part importante de l'erreur du modèle.

**Correctif à appliquer, mais pas à chaud.** Deux gestes chirurgicaux qui préservent l'ordre des nœuds, donc la validité des points de reprise : remettre dans `edges` la longueur du lac à la racine de sa surface, que `troncon.trl` permet de recalculer ; puis recalculer `dist_to_outlet_km` dans les attributs provinciaux. Une reconstruction complète des bases par `build_regions.py` corrigerait aussi, mais elle risque de déplacer la numérotation des nœuds et invaliderait tout ce qui existe.

**Leçon de méthode.** Un correctif de lecture ne vaut rien tant que les caches qu'il alimente n'ont pas été refaits. Il faudrait que la construction d'un cache inscrive l'empreinte du code qui l'a produit, comme le fait déjà `HydroModel.save` pour les points de reprise.

---

## R104 — Le biais de volume du modèle est organisé par le matériau parental, que le modèle ne reçoit pas (2026-09-15) — ÉTABLI

Croisement de la couverture pédologique de l'IRDA, 680 feuillets au 1:20 000, avec les unités hydrologiques de PHYSITEL, puis agrégation au tronçon par la même moyenne pondérée par l'aire que les attributs existants. Sept régions, 20 087 tronçons, dont 6 264 couverts à plus de la moitié de leur surface. Cumul amont sur le bassin de chaque station, puis comparaison au biais de volume mesuré sur 2022-2024. Aucun entraînement.

Sur les 69 stations dont le bassin est couvert à plus de 50 pour cent, aire médiane 367 km², corrélations de rang avec le rapport des moyennes simulée sur observée :

| attribut | rho brut | rho à aire contrôlée | p |
|---|---:|---:|---:|
| matériau till | -0,532 | -0,524 | < 0,00001 |
| matériau sable | +0,500 | +0,493 | 0,00002 |
| matériau argile | +0,394 | +0,388 | 0,00098 |
| matériau organique | +0,291 | +0,280 | 0,02 |
| classe de drainage | de -0,09 à +0,19 | — | > 0,4 |
| affleurement rocheux | +0,032 | +0,040 | 0,74 |

**Ampleur.** Entre le quartile pauvre en till, moins de 32 pour cent de la surface, et le quartile riche, plus de 75 pour cent, le rapport des moyennes passe de 1,112 à 0,868 et le KGE médian de 0,727 à 0,578. Pour le sable, de 0,941 à 1,226 entre les quartiles extrêmes. Le modèle assèche donc les bassins de till d'environ treize pour cent et noie ceux de sable d'environ vingt-trois.

**Ce n'est pas un effet de taille.** Contrôler par le logarithme de l'aire drainée ne déplace les corrélations que de 0,01. Sur les 29 stations de moins de 300 km² prises seules, l'effet est au moins aussi fort : till -0,478 avec p = 0,009, sable +0,597 avec p = 0,0006.

**Pourquoi le modèle ne peut pas le voir.** Il reçoit trois pourcentages granulométriques, sable, limon et argile. Un till et un sable délavé peuvent partager la même granulométrie et se comporter à l'inverse, le till étant compact, souvent surmonté d'une couche perméable, et reposant sur un substrat peu conducteur. Le matériau parental est une classification génétique qui porte cette différence ; la granulométrie ne la porte pas.

**Mécanisme plausible, non vérifié.** Sur un till classé comme loam, le champ prédit une conductivité modérée, l'eau percole vers la réserve profonde et, la vidange étant trop lente, elle ne ressort pas dans la période : déficit de volume. Sur un sable, la conductivité prédite est forte mais le substrat imperméable réel produit un écoulement hypodermique rapide que le modèle envoie en profondeur, d'où l'excès inverse. À éprouver.

**La classe de drainage, elle, n'explique rien du biais de volume.** C'était l'attribut que j'attendais ; il ne sort pas. Cela n'élimine pas son intérêt comme contrainte sur les paramètres, mais son apport comme entrée explicative du biais est nul sur cet échantillon.

**Limites.** 69 stations, sept régions, et uniquement là où la couverture pédologique dépasse la moitié du bassin, donc surtout les basses-terres agricoles. La relation est une corrélation : les bassins de till diffèrent aussi par leur position sur le Bouclier, même si le contrôle par l'aire ne change rien.

**Complément au constat précédent : le matériau n'est pas déductible de ce que le modèle possède (2026-09-15).** Prédiction du matériau dominant à partir des seize attributs actuels, sur 5 658 tronçons couverts à plus de 70 pour cent, validation croisée par BLOCS spatiaux d'un demi-degré pour que le voisin d'un tronçon retiré ne reste pas dans l'apprentissage : 78,0 pour cent d'exactitude contre 67,4 pour la classe majoritaire, soit un gain de 10,6 points. Le matériau est donc en grande partie une information neuve, et les trois pourcentages granulométriques ne le reconstituent pas.

**Conséquence, qui corrige une affirmation trop rapide.** Une contrainte auxiliaire ne corrige les poids que si la correction s'exprime avec des attributs disponibles partout. Comme le matériau ne se déduit pas des attributs actuels, une contrainte fondée sur lui ne pourrait pas être appliquée hors couverture : le champ n'en retiendrait qu'une correction moyenne. Le raisonnement qui vaut pour MODIS et GRACE, dont la couverture est complète, ne se transpose donc pas tel quel.

**Deux issues.** Accepter que le gain porte sur les 11 pour cent cartographiés, où se trouvent les usages, les prélèvements, la plupart des jauges et les petits bassins de la plaine. Ou chercher la même variable à couverture provinciale complète, c'est-à-dire une cartographie des dépôts de surface, plus grossière que le 1:20 000 de l'IRDA mais entière. Pour un modèle provincial, la seconde vaut mieux, et elle est à vérifier avant d'engager le chantier.

---

## R105 — Le rabotage des pointes n'existe que dans les basses-terres ; ailleurs le modèle est trop nerveux (2026-09-15) — ÉTABLI

Mesure sur les caches par station de la période d'évaluation 2022-2024, refaits le 15 septembre avec les champions déployés, quatorze régions.

| région | stations | pointes sim/obs | nervosité | excès de platitude | KGE |
|---|---:|---:|---:|---:|---:|
| Montérégie | 23 | 0,76 | 0,64 | +7,4 pt | 0,601 |
| Saint-Laurent sud-ouest | 29 | 0,78 | 0,57 | +4,4 pt | 0,595 |
| Saint-Laurent nord-ouest | 26 | 0,88 | 0,65 | +5,4 pt | 0,565 |
| Gaspésie | 15 | 1,03 | 0,94 | -0,5 pt | 0,632 |
| Côte-Nord C | 2 | 1,23 | 2,12 | -2,9 pt | 0,634 |
| Côte-Nord B | 2 | 1,25 | 1,49 | -3,8 pt | 0,662 |
| Saguenay | 19 | 1,26 | 0,91 | -0,9 pt | 0,636 |
| Outaouais aval | 16 | 1,30 | 1,40 | +1,4 pt | 0,527 |
| Côte-Nord A | 1 | 1,40 | 2,09 | -8,1 pt | 0,339 |
| Labrador | 1 | 1,51 | 1,10 | -6,0 pt | 0,765 |
| Abitibi | 3 | 1,55 | 2,08 | -11,6 pt | 0,592 |
| Outaouais moyen | 3 | 1,62 | 1,45 | -11,4 pt | 0,424 |

**Aucune région n'a de pointes sous 0,75.** Le déficit de 0,39 à 0,59 qui a motivé le chantier du routage ne décrit aucun déploiement régional courant.

**Deux populations opposées.** Les trois régions des basses-terres du Saint-Laurent portent le sur-lissage : pointes de 0,76 à 0,88, nervosité de 0,57 à 0,65, excès de platitude de 4,4 à 7,4 points. Ce sont les seules dont la nervosité tombe sous 0,70. Partout ailleurs le modèle est trop nerveux : pointes de 1,03 à 1,62, nervosité jusqu'à 2,12, et une platitude INFÉRIEURE à l'observée, jusqu'à 11,6 points de moins.

**Conséquences pour le chantier du routage.** L'épreuve appariée de l'ancre a porté sur le Saguenay et la Gaspésie, qui appartiennent à la seconde population : raccourcir leur temps de transfert ne pouvait qu'aggraver une nervosité déjà excessive. Une épreuve valide doit se faire sur la Montérégie ou le Saint-Laurent sud-ouest. Et un réglage uniforme du routage est faux en principe, puisqu'il faudrait accélérer les basses-terres et ralentir le Bouclier : c'est un argument POUR une ancre géométrique, qui varie avec la longueur et la pente, à condition de la juger région par région et jamais sur une moyenne provinciale.

**Réserve.** Ces chiffres viennent des champions déployés ; les bras de Narval employaient la recette du socle à vingt époques, dont les témoins donnaient 0,81 et 0,95 pour le Saguenay et la Gaspésie contre 1,26 et 1,03 ici. La recette déplace donc le diagnostic, et cette mesure doit être refaite sous la recette qu'on entend éprouver.

**CORRECTION du constat précédent, le même jour.** La partition en deux populations est en grande partie un artefact de DÉPLOIEMENT et non une propriété du territoire. Six régions tournent avec leur propre point de reprise, six autres avec le champion de la Gaspésie appliqué ailleurs.

| déploiement | régions | pointes médianes | nervosité médiane | KGE médian |
|---|---:|---:|---:|---:|
| point de reprise propre | 6 | 0,96 | 0,78 | 0,598 |
| champion de la Gaspésie transféré | 6 | 1,46 | 1,78 | 0,613 |

Un modèle calé en Gaspésie et posé sur l'Abitibi produit donc un débit deux fois plus nerveux que l'observé. Cela ne renseigne pas sur l'Abitibi, cela mesure le coût d'un transfert.

**Ce qui survit.** Parmi les six régions calées chez elles, les trois des basses-terres restent les plus lisses, 0,57, 0,64 et 0,65, contre 0,91, 0,94 et 1,40 pour les trois autres. Trois observations sur six ne fondent pas un chantier.

**Et la nervosité ne se prédit pas par le territoire.** Sur les 143 stations, elle forme un continuum, médiane 0,86 et du premier au neuvième décile 0,39 à 1,89, non deux amas. Le meilleur attribut explicatif est l'ordre de Strahler à -0,467, puis la fraction urbaine à -0,315. Une prédiction par l'ensemble des attributs du bassin amont donne un R² de +0,08 en retirant des stations au hasard mais de -0,27 en retirant des RÉGIONS entières, donc pire que de prédire la moyenne générale. Rien ne généralise.

**Conséquence de méthode, qui vient d'Essi.** Un ancrage réglé par région ajusterait un artefact de production ; la région est un artefact et non une entité hydrologique. Le diagnostic du routage ne peut pas se trancher sur des caches qui mélangent six modèles calés et un modèle transféré six fois. Il faut un modèle unique sur tout le domaine, une validation croisée par stations et par régions retirées, et le travail porte alors sur la fonction de perte.

---

## R106 — Le terme des variations d'un jour est le seul qui ne préfère jamais une série lissée (2026-09-15) — ÉTABLI

Banc de fonction de perte, 77 stations complètes, neuf régions. On déforme l'hydrogramme observé comme le modèle se trompe et on demande à chaque terme s'il préfère l'original. La référence n'est pas la vérité mais la série NETTE DÉCALÉE D'UN JOUR, parce que c'est le choix réel du modèle : rester net et en retard, ou lisser pour réduire l'écart quotidien.

**La recette du socle récompense le lissage.** Face à un lissage sur sept jours, son total est meilleur de 8,0 pour cent pour la version lissée, et elle la préfère sur 64 pour cent des stations. Elle préfère aussi des pics rabotés sur 23 pour cent des stations. Chacun de ses cinq termes est trompé par le lissage de sept jours : biais de volume -11,0 pour cent, écart quadratique -16,9, écart quadratique logarithmique -15,1, pics -21,3. Seul le KGE résiste, +46,0.

**Le terme des variations d'un jour ne l'est jamais.** Il préfère la série nette dans cent pour cent des cas, pour les six déformations. Et il est quasi insensible à un décalage pur : sa valeur sur la série nette décalée vaut 8·10⁻¹⁰, contre des ordres de grandeur au-dessus dès qu'on lisse. Il pénalise donc le défaut qu'on veut combattre et ignore l'erreur de calendrier que le modèle ne peut pas éviter.

**Ce terme existe dans la perte, sous le nom `w_dq`, et n'est pas dans la recette du socle,** dont les poids sont KGE 1,0, biais de volume 0,5, écart quadratique 0,1, écart quadratique logarithmique 0,3 et pics 0,5.

**Réserve sur l'ampleur.** Le banc déforme l'observation elle-même, de sorte que la série de référence est la vérité décalée et que le terme y vaut presque zéro par construction. Sur une simulation réelle, les deux séries diffèrent partout et les rapports seraient bien plus modestes. La conclusion qualitative tient, l'ampleur chiffrée non : le poids devra être trouvé, pas déduit du banc.

**Ce que cela ouvre.** Une épreuve appariée à deux bras, données identiques et une seule différence, le terme ajouté ou non. C'est la question que la journée a isolée : la perte récompense-t-elle encore le lissage une fois qu'on lui adjoint le seul terme qui ne s'y laisse pas prendre.

**Poids proposé pour ce terme, et ce qu'on attend de lui (2026-09-15).** Deux réglages indépendants convergent. Par la part du total, sur 129 stations et des simulations réelles, il faut 0,32 pour que le terme pèse cinq pour cent de la perte, 0,68 pour dix et 1,52 pour vingt. Par le renversement du verdict du banc, il faut 0,27 en médiane pour que la perte cesse de préférer le lissage sur sept jours, et 0,98 pour corriger quatre-vingt-dix pour cent des stations concernées ; pour le rabotage des pics, 1,22 en médiane et 2,69 pour quatre-vingt-dix pour cent.

**Poids retenu : 1,0.** Il pèse alors environ treize pour cent du total sur une simulation réelle. ATTENDU, énoncé avant l'épreuve : il supprime la préférence pour le lissage sur environ quatre-vingt-dix pour cent des stations, et sur un peu moins de la moitié pour le rabotage des pics.

**Pourquoi ne pas monter à 2,69.** Le terme pèserait trente pour cent du total, ce qui est trop pour un terme unique à côté de cinq autres.

**CORRECTION d'un argument que j'avais avancé.** J'avais écarté 2,69 en disant qu'un terme sur les variations pousserait à la nervosité et aggraverait les huit régions déjà trop nerveuses. C'est faux. Le commentaire du pilote, écrit le 2026-09-05, énonce que ce terme est le SEUL de la perte qui punisse SYMÉTRIQUEMENT le plateau et la nervosité, là où gamma compare des écarts-types de niveau qu'un plateau à la bonne moyenne satisfait. Il compare des écarts-types de VARIATIONS, donc un excès comme un défaut lui coûtent.

**Attendu renforcé, et plus facile à réfuter.** Le terme doit améliorer les DEUX populations mesurées le même jour : les trois régions des basses-terres dont la nervosité vaut 0,57 à 0,65, et les régions dont elle dépasse 1, jusqu'à 2,12. Si le bras des variations n'améliore que l'une des deux, la propriété de symétrie annoncée par le code est fausse et il faudra le dire.

---

## R107 — Les plateaux d'hiver ne sont pas récompensés par la perte : c'est la physique du gel qui est absente (2026-09-15) — ÉTABLI

Deux mesures indépendantes se rejoignent, et elles déplacent la recherche.

**La perte rejette déjà un hiver figé, vigoureusement.** Banc de déformation sur 77 stations : face à la série nette décalée d'un jour, un hiver remplacé par une récession exponentielle est noté 49,9 pour cent PLUS CHER par la recette, et elle ne le préfère sur AUCUNE station. Le terme des variations d'un jour le rejette aussi, sur cent pour cent des stations. Les plateaux ne sont donc pas une affaire d'incitation : aucun réglage de poids ne les fera disparaître, puisque la perte les combat déjà.

**Or le modèle en produit de 74 à 116 jours.** Les six bras de l'épreuve de routage du même jour reçoivent tous le verdict de forme « FORME REFUSÉE (plateaux) », avec des platitudes de 17,6 à 46,1 pour cent contre 12 à 14 pour cent dans l'observé. Si la perte les combat et qu'ils persistent, c'est que le modèle ne SAIT PAS produire un hiver mouvant.

**La physique du gel est absente ou indifférenciée.** Le champion de la Montérégie porte 37 sorties de champ sur 42. Les cinq manquantes sont `krec`, `diff_gel`, `fs_neige` et les deux écarts de canopée : elles sont remplies de zéros au chargement, donc figées au milieu de leurs bornes et identiques sur tous les tronçons. Deux d'entre elles, la diffusivité thermique apparente et l'amortissement par le couvert nival, sont précisément les paramètres du gel. Le seul paramètre de gel présent, `frost_alpha`, a une norme de ligne de 0,0562 pour une médiane de 0,0592, donc dans le dernier décile : le champ ne le différencie pas non plus.

**La fonte, elle, est bien apprise,** et n'est donc pas en cause : le facteur de fonte a une norme de 0,419 et la température de fonte 0,448, soit sept fois la médiane. Les paramètres les mieux appris sont les porosités, 0,68, les épaisseurs de couche, 0,67, et les conductivités, 0,61.

**Conséquence, et attendu énoncé d'avance pour l'épreuve de la nuit du 15 au 16 septembre.** Les deux bras portent des étiquettes neuves, donc aucun point de reprise ne leur correspond et ils partent à FROID, avec les quarante-deux sorties. La physique du gel y sera entraînée pour la première fois. Si les plateaux viennent bien de là, les deux bras doivent montrer des suites plates nettement plus courtes que les 74 à 116 jours mesurés le matin, et cela INDÉPENDAMMENT du terme des variations, qui ne les vise pas.

---

## R108 — Le terme des variations d'un jour règle les plateaux d'hiver et relève le KGE, au prix des pointes là où elles étaient bonnes (2026-09-15) — ÉTABLI

Épreuve appariée sur Narval, tâche 3117503, vingt-huit tâches, quatorze régions et deux bras, données corrigées et recette identique : seule différence, `w_dq` à 0 ou à 1,0. Dix-neuf tâches terminées, neuf régions avec leurs deux bras. Les neuf autres ont échoué sur un défaut du code, traité plus bas.

| région | pointes témoin → variations | suite plate (j) | jours plats (%) | KGE |
|---|---|---|---|---|
| Montérégie | 0,74 → 0,98 | 52 → 22 | 39,8 → 7,0 | 0,510 → 0,638 |
| Saint-Laurent nord-ouest | 0,70 → 0,87 | 59 → 35 | 22,9 → 16,7 | 0,652 → 0,677 |
| Côte-Nord C | 0,80 → 1,09 | 62 → 21 | 33,2 → 17,7 | 0,683 → 0,709 |
| Abitibi | 1,13 → 1,13 | 52 → 5 | 19,5 → 12,0 | 0,817 → 0,862 |
| Outaouais aval | 1,04 → 0,81 | 57 → 20 | 31,0 → 14,8 | 0,670 → 0,664 |
| Gaspésie | 0,95 → 0,95 | 89 → 88 | 26,1 → 29,4 | 0,725 → 0,781 |
| Saguenay | 0,86 → 0,72 | 98 → 101 | 27,6 → 32,2 | 0,769 → 0,781 |
| Saint-Laurent sud-ouest | 0,89 → 0,87 | 46 → 49 | 24,7 → 27,4 | 0,654 → 0,643 |
| Outaouais moyen | 0,86 → 0,77 | 79 → 69 | 17,5 → 21,7 | 0,831 → 0,841 |

**Les plateaux reculent.** La plus longue suite plate passe de 60 jours en médiane à 35. Cinq régions gagnent de 24 à 47 jours ; quatre ne bougent pas.

**Le KGE monte dans sept régions sur neuf,** de +0,010 à +0,128, avec deux reculs de moins de 0,012.

**Les pointes sont le point faible, et l'effet n'est pas symétrique.** Le terme les relève franchement là où elles manquaient, de 0,70 à 0,87, de 0,74 à 0,98 et de 0,80 à 1,09, mais il les abaisse là où elles étaient correctes, de 1,04 à 0,81 dans l'Outaouais aval et de 0,86 à 0,72 au Saguenay. La symétrie annoncée par le commentaire du pilote ne se vérifie donc pas sur les pointes annuelles, même si elle se vérifie sur la platitude.

**ATTENDU RÉFUTÉ.** J'avais prédit, avant l'épreuve, que les suites plates raccourciraient dans les DEUX bras, le départ à froid entraînant la physique du gel pour la première fois. C'est faux : sur les deux régions comparables à l'épreuve du matin, le témoin donne 98 jours au Saguenay contre 96 et 89 en Gaspésie contre 74. Le témoin ne s'améliore pas. L'absence de `diff_gel` et `fs_neige` dans les anciens points de reprise n'explique donc pas les plateaux, et cette piste est éliminée.

**Ce que l'épreuve établit.** Les plateaux d'hiver sont une affaire de fonction de perte et non de physique manquante, et le terme des variations d'un jour les traite. La Montérégie, région la plus faible du domaine, gagne 0,128 de KGE, vingt-quatre centièmes sur ses pointes et trente-trois points de platitude.

**Piste ouverte.** Un poids inférieur à 1,0 conserverait vraisemblablement le gain sur les plateaux en réduisant la perte sur les pointes des régions déjà nerveuses. À éprouver sur l'Outaouais aval et le Saguenay, qui sont les deux qui reculent.

**Le défaut qui a coûté neuf tâches.** `loss.py` lève une erreur sur `L_dq` quand un bloc d'entraînement ne garde aucune station atteignant trente observations valides. Le garde-fou existe dans le dépôt depuis le 9 septembre, mais le paquet expédié la veille ne contenait que le pilote et les scripts, pas le module. Même classe d'incident que la longueur des lacs le matin même : un correctif écrit ici et jamais parvenu là-bas. Le paquet de reprise contient désormais les 223 fichiers du module et des scripts, et le correctif a été éprouvé sur le cas exact avant la resoumission.

---

## R109 — Les modèles de référence du 15 septembre et leur première tête de quantiles n'ont fait qu'un pas d'optimisation par époque (2026-09-16) — ÉTABLI

`reference_variations.sbatch` et `quantile_ref.sbatch` ne posaient pas `MEANDRE_PAS_PAR_BLOC=1`. La recette embarquée dans les points de reprise le confirme : aucune variable de ce nom n'y figure. Le pilote fait donc un pas d'Adam par époque, comme décrit en R78. Les trente époques du modèle déterministe valent trente pas, et les dix de la tête de quantiles en valent dix.

Signature sur la tête de la Montérégie : les biais de la dernière couche valent -0,996 et 1,004 pour une initialisation à -1 et 1. Le quantile à 5 % vaut 0,87 fois la médiane et celui à 95 % 1,14 fois. Sur 2022-2024, la couverture de l'intervalle annoncé à 90 % vaut 0,37 toutes régions confondues.

Tâche 3170422, même socle gelé, avec un pas par bloc et un taux de 1e-2 : couverture de 0,853 à 90 % et de 0,420 à 50 % sur 2022-2024, 144 stations. Le diagramme de Talagrand est plat de la deuxième à la dix-neuvième classe, entre 3,4 et 5,3 %. Sa première classe vaut 5,8 % et sa dernière 9,7 %. Le Saguenay reste à 0,58 et l'Abitibi à 0,66.

Conséquence. R108 compare deux bras qui ne diffèrent que par le terme des variations : la comparaison reste appariée. Mais les deux modèles sont restés proches de leur initialisation, et tout énoncé sur un paramètre appris par ces points de reprise est à suspendre. Le temps de transfert de 4,4 heures en Montérégie en est un exemple : il reflète le départ, pas un choix de l'optimiseur. Le réentraînement avec un pas par bloc est prévu dans la prochaine ronde de modélisation, après le livrable du 2026-09-30.

---

## R110 — Le croisement de l'IRDA surestimait la couverture, et l'information du matériau parental est plus modeste qu'annoncé (2026-09-16) — ÉTABLI

`croiser_irda.py` prenait l'aire d'une unité hydrologique dans un dictionnaire construit ligne à ligne. Or 2 282 des 8 623 polygones de SLSO, et 1 396 des 5 564 de la Montérégie, portent un identifiant répété : une unité peut compter jusqu'à neuf morceaux. Le dictionnaire gardait l'aire du dernier morceau, souvent 0,01 km², alors que les classes de sol étaient sommées sur tous. La couverture montait ainsi jusqu'à 630 sur un tronçon. Corrigé par la somme des aires par unité ; la couverture plafonne à 1 sur les quinze régions. Les chiffres de R104 reposaient sur la table fautive.

Question mal posée, retirée. Une première version jugeait l'IRDA sur sa capacité à prédire le biais de volume et le KGE, par des corrélations univariées. Ce n'est pas le critère : une donnée d'entrée du champ se juge sur ce qu'elle ajoute à la séparation des nœuds, donc à l'identifiabilité, et non sur un score.

Propriétés numériques, depuis le géopaquet 2026_01 (`irda_proprietes.py`). Chaque composante de polygone reçoit une texture de surface en ilr, prise dans la BDHP (3 824 composantes) ou au centre de gravité de sa classe granulométrique de famille (746). S'y ajoutent le rang de drainage, le logarithme de la perméabilité au centre de sa classe de Wischmeier et Smith, le code de structure, le groupe hydrologique, et les parts de contact lithique mince et de sol organique. Moyennes pondérées par le pourcentage des composantes, puis par l'aire.

Redondance avec les attributs actuels (`test_irda_redondance.py`), compositions en ilr, gradient boosté en validation croisée par blocs d'un degré, 6 700 à 7 100 tronçons couverts à plus de moitié. R² hors bloc : 0,14 et 0,13 pour les deux coordonnées de texture, 0,37 pour le drainage, 0,12 pour la perméabilité, 0,09 pour la structure, 0,19 pour le groupe hydrologique, 0,18 pour le contact lithique, 0,14 pour les sols organiques. L'IRDA n'est donc pas redondante avec l'entrée actuelle. Parmi le quart des paires de tronçons voisins les plus semblables en attributs actuels, 40 % s'écartent de plus d'un demi-écart-type en propriétés IRDA. Le rang effectif de l'entrée passe de 9,3 à 11,1.

Redondance interne. Les huit propriétés n'ont qu'un rang effectif de 3,8. La structure suit la texture (r de -0,78 et -0,75) et le groupe hydrologique suit la perméabilité (r de -0,80). La BDHP dérive les deux de la texture et de la perméabilité : elles sont à retirer. Restent six variables : deux coordonnées de texture, drainage, perméabilité, contact lithique, sols organiques.

Intégration au champ, mesurée sur le modèle de la Montérégie sans simulation. Le chargeur rembourre les nouvelles colonnes par des poids aléatoires à petite échelle : les paramètres bougent de 0,02 % en médiane et de 0,6 % au pire. Des poids nuls laissent le modèle identique à 1e-6 près. Le gradient sur les nouveaux poids vaut le quart de celui des entrées existantes.

---

## R111 — SIIGSOL et l'indice d'humidité topographique sont en grande partie redondants avec les attributs du champ (2026-09-17) — ÉTABLI

Ingestion standard (`meandre/data/auxiliary`, `ingerer_auxiliaire.py`) à partir des `source.toml` de `sources/siigsol` et `sources/humidite-lidar`. SIIGSOL : texture en ilr et logarithme de la matière organique, six profondeurs, moyennés par aire sur les 28 035 tronçons, couverture médiane de 0,90 à 0,99 hors LABI et VAUD, qui débordent en Ontario. Indice d'humidité : feuillets lus à distance sur l'aperçu à 8 m, histogramme par unité hydrologique, 20 904 tronçons couverts à plus de moitié ; six feuillets de l'index ne sont pas publiés, et aucun LiDAR ne couvre CNDD ni CNDE.

Redondance, gradient boosté en validation croisée par blocs d'un degré contre les 15 attributs actuels. SIIGSOL : R² hors bloc de 0,52 et 0,75 pour les deux coordonnées de texture en surface, 0,78 pour la matière organique en surface, 0,55 à 0,75 en profondeur. Les 18 variables n'ont qu'un rang effectif de 2,15 : les profondeurs se répètent. La corrélation avec la texture de PHYSITEL n'est que de 0,31 et 0,54, mais les autres attributs, altitude, pente et occupation, la reconstituent. Indice d'humidité : R² de 0,63 à 0,76 pour ses sept statistiques, rang effectif de 1,78. Deux dimensions s'en dégagent, le quantile 10 et le quantile 90, corrélés à 0,09.

Contraste avec l'IRDA, dont les propriétés n'étaient prédites qu'à 0,09 à 0,37. Les deux produits ministériels sont construits à partir de covariables de télédétection et de relief proches de l'entrée du champ ; l'IRDA vient de levés de terrain. À retenir, si on les ajoute : une coordonnée de texture en surface et une en profondeur pour SIIGSOL, les quantiles 10 et 90 de l'indice d'humidité.

---

## R112 — Les niveaux du réseau de suivi des eaux souterraines couvrent nos régions et portent une information que le débit n'a pas (2026-09-17) — ÉTABLI

Base de diffusion du réseau, 41 millions de mesures aux six heures de 1968 à 2026, réduites à une moyenne journalière par puits (`rsesq_niveaux.py`). Sur 297 puits, 186 tombent à moins de 3 km d'un tronçon. Sur 2022-2024, 171 puits ont 1 096 jours de mesures, soit la période entière ; sur 2001-2024, 184 puits en ont 5 000 en médiane, soit quatorze ans. Répartition : 53 en Saint-Laurent sud-ouest, 46 en Montérégie, 27 en Outaouais aval, 20 en Gaspésie. Le réseau est décrit par aquifère, 163 au roc et 134 en mort-terrain, par confinement, 131 libres et 130 captifs, et 50 puits sont déclarés influencés.

Information propre, mesurée sans simulation. Pour les 93 paires puits-station distantes de moins de 25 km, l'anomalie mensuelle du niveau, signe inversé puisque la mesure est une profondeur, est corrélée à l'anomalie mensuelle du débit observé à 0,51 en médiane, de -0,11 à 0,81. Le débit n'explique donc qu'un quart de la variance du niveau ; les trois autres quarts sont une information que l'entraînement sur le débit seul ne voit pas. Quarante-six pour cent des paires restent sous 0,5.

Le registre l'autorise déjà comme contrainte de tendance : ce sont des mesures directes de hauteur d'eau, sans assimilation de nos débits. La valeur absolue, elle, est une profondeur sous le repère du tubage et ne se compare pas au stockage simulé ; seules les variations le peuvent.

Reste à mesurer, et cela demande une simulation : le stockage souterrain simulé suit-il ces variations, et une contrainte sur ce point resserre-t-elle les paramètres de l'aquifère entre deux graines. Le pilote ne sort pas encore le stockage souterrain journalier aux nœuds des puits.

---

## R113 — Le manteau simulé est juste en Outaouais, trop épais en Gaspésie et trop mince au Saguenay (2026-09-18) — ÉTABLI

Comparaison sans simulation, sur la climatologie mensuelle de l'équivalent en eau des caches du 15 septembre et les relevés CanSWE des bases régionales, 2001-2024. Sites retenus par représentativité : à moins de 15 km du nœud et à moins de 150 m d'écart d'altitude. Rapport du manteau simulé au manteau mesuré, médianes mensuelles :

| région | sites | janvier | février | mars | avril |
|---|---:|---:|---:|---:|---:|
| Outaouais aval | 73 | 1,11 | 1,12 | 1,03 | 1,04 |
| Gaspésie | 24 | 1,29 | 1,64 | 1,65 | 1,56 |
| Saguenay | 34 | 0,78 | 0,76 | 0,76 | 0,65 |
| Saint-Laurent nord-ouest | 30 | 0,92 | 0,91 | 0,87 | 0,76 |
| Côte-Nord B | 33 | 1,23 | 1,24 | 1,22 | 1,32 |
| Côte-Nord A | 20 | 0,98 | 0,97 | 0,93 | 1,03 |
| Abitibi | 4 | 0,85 | 0,87 | 0,85 | 0,68 |

Remarque d'Essi, vérifiée le jour même : la mesure est d'autant moins sûre que les sites sont rares. La dispersion entre sites est en effet considérable au sein d'une même région ; en Outaouais, sur 73 sites, le rapport par site va de 0,65 au premier décile à 2,42 au neuvième. En tirant n sites au hasard parmi ces 73, l'intervalle à 90 % de la médiane vaut 0,76 à 2,00 pour n = 4, 0,95 à 1,49 pour n = 20 et 1,07 à 1,34 pour n = 73. Les valeurs de l'Abitibi, du Labrador et de la Côte-Nord C, tirées de trois à cinq sites, ne distinguent donc pas un défaut du modèle d'un tirage de sites, et celle de la Côte-Nord B reste dans l'intervalle attendu.

Trois écarts survivent à ce test : la Gaspésie à 1,55 sur 24 sites, au-dessus de l'intervalle ; le Saguenay à 0,84 sur 34 sites et le Saint-Laurent nord-ouest à 0,91 sur 30 sites, en dessous. Ailleurs, la médiane par site tient entre 0,84 et 1,29, soit dans le bruit de la mesure. Réserve : la dispersion de l'Outaouais sert de référence pour toutes les régions ; une région plus montagneuse en aurait davantage, ce qui rendrait le test encore plus indulgent.

L'écart gaspésien persiste donc, à 1,6 fois la mesure contre le facteur deux relevé en août. Le déficit du Saguenay et du Saint-Laurent nord-ouest s'aggrave d'avril en avril, ce qui touche directement la crue printanière. L'Outaouais, où la contrainte de neige serait la mieux fondée avec ses 73 sites retenus, est précisément la région où le manteau est déjà juste.

Conséquence pour le chantier de la neige mesurée au sol : la contrainte a un sens là où le manteau est faux, donc en Gaspésie, au Saguenay et au Saint-Laurent nord-ouest, et non sur la région la mieux couverte. Le couple couverture-défaut doit guider le choix de la région d'épreuve. Ces rapports portent sur les modèles du 15 septembre, entraînés en trente pas d'optimisation.

---

## R114 — Le réservoir souterrain du modèle ne respire pas : 2 mm de battement saisonnier contre 0,67 m mesuré, et en opposition de phase (2026-09-18) — ÉTABLI

Évaluation seule des modèles du 15 septembre, sept régions, avec le stock souterrain journalier écrit aux nœuds portant un puits du réseau de suivi (`ETL_DUMP_NAPPE`, `comparer_nappe.py`). Cent dix-huit puits comparés, 169 mois chacun en médiane. Le stock simulé est en millimètres, le niveau mesuré en mètres sous le repère du tubage : la comparaison porte sur la dynamique.

| mesure | résultat |
|---|---|
| corrélation des anomalies mensuelles | -0,30 en médiane, aucun puits au-dessus de 0,5 |
| corrélation du cycle saisonnier | -0,55 |
| mois le plus haut, mesuré | avril (53 puits), mai (52) |
| mois le plus haut, simulé | septembre (40), août (35) |
| amplitude saisonnière mesurée | 0,67 m de battement |
| amplitude saisonnière simulée | 2 mm de stock |

Un battement de 0,67 m correspond, pour une porosité de drainage plausible de 2 à 20 pour cent, à 13 à 130 mm de variation de stock. Le modèle en fait varier deux : son aquifère se vide presque aussi vite qu'il se remplit et ne stocke donc rien. Le peu qu'il varie culmine en fin d'été, quand la nappe réelle est au plus bas, d'où l'opposition de phase. La pente de régression du stock sur le niveau rend une porosité de drainage négative sur la totalité des puits, ce qui signale la même chose autrement.

Ce constat nomme un fait déjà rencontré sans être expliqué : la nappe affamée du modèle retenu, et l'aquifère restituant qui ne paie jamais. Le temps de séjour du réservoir souterrain, et non son coefficient de recharge seul, devient le paramètre à revoir.

Réserves. Ces modèles n'ont fait que trente pas d'optimisation. Le stock est pris au nœud le plus proche du puits, sans égard à la profondeur captée ni au confinement. La porosité de drainage n'est utilisée que pour juger l'ordre de grandeur.

**Cause, lue dans les paramètres sans simulation (même jour).** Le coefficient de vidange `k_gw` vaut 0,07 à 0,13 par jour selon la région, soit un temps de séjour médian de 7 à 16 jours, identique aux nœuds des puits. Avec une recharge annuelle de 60 à 130 mm, un réservoir qui se vide en dix jours contient en moyenne 3 mm et suit la recharge avec dix jours de retard : l'amplitude de 2 mm et le maximum d'été en découlent tous deux. Ce temps de séjour vient de deux choix de conception. Le champ provincial de `k_gw` est ajusté sur les récessions de débit de 127 stations, qui mesurent la vidange rapide du sol et du versant, non celle de la nappe. Et le prior du champ interdit les temps de séjour longs, son commentaire disant empêcher une dérive vers 0,005 par jour, soit 200 jours, précisément l'ordre de grandeur d'une nappe réelle. Le débit seul ne pouvait pas révéler cette contrainte ; les niveaux mesurés la révèlent d'emblée.

---

## R115 — La géologie du socle de SIGÉOM entre par le cadre commun, et sa lithologie n'est pas prédictible par les attributs actuels (2026-09-18) — ÉTABLI

Géopaquet provincial de Géologie Québec, 576 Mo, licence CC-BY 4.0, ingéré par le type `vecteur_classes` : parts de classes par tronçon, par intersection avec les unités hydrologiques, couverture à 1 sur les quinze régions hors Labrador et Vaudreuil, qui débordent de la carte. Deux couches. La province géologique, 63 divisions ramenées à six classes : Plate-forme du Saint-Laurent, Appalaches, Grenville, Supérieur, Churchill, autre bouclier. La lithologie de la carte généralisée, 1 395 descriptions ramenées par mots-clés à huit classes hydrogéologiques : carbonate, clastique fin, clastique grossier, volcanique, intrusif felsique, intrusif mafique, gneiss, formation de fer ; 99,3 % de la surface est classée. Vérification sur la Montérégie : 42 % de plate-forme et 58 % d'Appalaches, 53 % de carbonates et 37 % de clastiques fins.

Redondance, gradient boosté en validation croisée par blocs d'un degré contre les quinze attributs actuels. La province se prédit à 0,51 à 0,66 de R² hors bloc : la position et l'altitude la portent déjà en grande partie. La lithologie ne se prédit pas : R² de -0,03 à 0,23 selon la classe, 0,23 pour les carbonates, 0,18 pour le gneiss, 0,09 pour les intrusifs felsiques. C'est le même profil que l'IRDA, et le contraire de SIIGSOL et de l'indice d'humidité. Les 163 puits du réseau captant dans le roc n'ont, sans cette couche, aucun descripteur de leur aquifère dans l'entrée du champ.

---

## R116 — La couche 3 du sol est saturée en permanence : le drainage profond est un robinet sans état, et l'accélérer aplatit l'hydrogramme sans rien perdre (2026-09-18) — ÉTABLI

Le drainage profond s'écrit ici comme le vidage linéaire de la couche 3, débit proportionnel à la teneur en eau, avec une constante de récession calée sur les récessions de débit. Multiplier cette constante par cinq, à poids gelés sur quatre régions, porte la recharge de 55 à 612 mm par an et renverse la corrélation des anomalies de nappe de -0,25 à +0,25, le maximum simulé passant de septembre à juin. Le débit, lui, s'effondre : KGE médian sur 2022-2024 de 0,654 à -0,180 en Outaouais, 0,006 en Gaspésie, 0,121 au Saint-Laurent nord-ouest.

L'effondrement n'est pas une perte d'eau. Aux quarante tronçons les plus débitants de l'Outaouais, le débit moyen simulé MONTE de 1131 à 1431 m³/s, soit 27 pour cent de plus, parce que l'évapotranspiration tombe de 547 à 459 mm par an : vider la couche 3 cinq fois plus vite assèche la zone racinaire, dont la teneur en eau moyenne passe de 0,517 à 0,276, et le sol ne peut plus alimenter la demande. Ce qui disparaît, c'est la variabilité. L'écart-type du débit tombe à 0,32 de sa valeur et le cycle mensuel, qui allait de 0,55 à 1,81 fois la moyenne, devient plat entre 0,95 et 1,04. Le réservoir souterrain reçoit 611 mm par an pour un stock moyen de 18 mm, soit onze jours de séjour : il ne retient rien et convertit la recharge en débit constant.

**Le défaut de structure.** Dans le modèle retenu, le rapport de la teneur en eau à la porosité dans la couche 3 vaut 1,000 en médiane et 0,952 au dixième centile : la couche est saturée neuf jours sur dix. Une couche saturée n'a plus d'état, et toute loi qui dépend de sa teneur en eau est débranchée. C'est l'explication du résultat nul de la variante à drainage non linéaire essayée le même jour : sa courbure s'applique à une variable qui ne bouge pas. La couche 3 ne fonctionne pas comme un réservoir mais comme un bouchon, l'excès remontant par la cascade de saturation vers la couche 2, d'où il repart en écoulement hypodermique.

**Le drainage libre n'est pas le remède.** Sous gradient unitaire, le flux au bas de la colonne vaudrait la conductivité hydraulique de Campbell à la teneur en eau du moment. Appliquée aux séries enregistrées avec la conductivité à saturation du champ, 0,38 m par jour en médiane, elle donne 300 000 mm par an, cinq cents fois la précipitation, et ce pour tout exposant entre 7 et 15. Ce qui limite la percolation au bas d'un sol québécois n'est pas la conductivité du sol mais celle du substratum sous lui. La constante de récession joue ce rôle par accident, sans dépendre de la géologie. Le taux de percolation profonde est une propriété du dépôt et du socle, que la couche de géologie ingérée le même jour décrit désormais par tronçon.

La non-linéarité place tout de même son maximum en avril, pour tous les exposants essayés : le mécanisme est juste, l'échelle est fausse. Le remède doit être un couple, aucun de ses deux termes ne suffisant seul. Un taux de percolation d'ordre de grandeur celui du substratum, portant la courbure de Campbell pour que la couche 3 se désature et respire, calé sur une recharge de 100 à 250 mm par an. Et un temps de séjour de l'aquifère qui se compte en mois, faute de quoi la recharge ressort aussitôt en débit constant.

Réserves. Passes avant seules, à poids gelés, sans réentraînement : le modèle n'a pas eu l'occasion de compenser ailleurs. Les quarante tronçons les plus débitants portent un maximum simulé de juin-juillet dans le témoin, décalage qui reste à expliquer et qui pointe vers les retards des pseudo-lacs plutôt que vers la colonne.

---

## R117 — En Outaouais, le temps de parcours du routage est collé à sa borne supérieure et retarde la crue de deux mois sur les grands tronçons (2026-09-18) — ÉTABLI

Constat trouvé en vérifiant la réserve laissée par l'épreuve du drainage profond. Dans le modèle retenu en Outaouais, le cycle mensuel du débit simulé culmine en avril sur les tronçons de taille médiane, où il vaut 3,7 fois la moyenne annuelle, et en juin sur le décile le plus débitant. Deux mois séparent la crue de tête de bassin de celle du cours principal.

Le paramètre de routage l'explique. Le temps de parcours de Muskingum vaut 46,2 heures en médiane sur les 3 412 tronçons de l'Outaouais, et 43 pour cent d'entre eux sont à moins d'une heure de la borne supérieure du domaine autorisé, fixée à 48 heures. Deux jours de parcours par tronçon, cumulés le long du réseau, donnent l'ordre de grandeur du décalage observé. Aucune autre région ne présente ce comportement : la médiane vaut 14,5 heures en Abitibi, 22,4 en Gaspésie, 25,9 sur la Côte-Nord centrale, 28,0 au Saint-Laurent nord-ouest et 33,0 au Saguenay, et aucun tronçon n'y touche la borne.

Un temps de parcours de 48 heures sur un tronçon correspond à une vitesse de quelques centimètres par seconde, valeur d'un plan d'eau et non d'une rivière. L'hypothèse à éprouver est celle des pseudo-lacs déjà identifiés comme réservoirs actifs artificiels, denses sur ce territoire. Le fait est établi ; sa cause ne l'est pas.

Ce constat ne concerne pas la nappe et n'entre pas dans le chantier de la recharge. Il est consigné parce qu'il porte sur la région servant de témoin à ce chantier, et qu'il déplace la crue simulée de deux mois là où la nappe mesurée culmine en avril.

---

## R118 — Aucun réglage du drainage profond ne déplace la recharge vers le printemps ; la porte de gel est innocentée (2026-09-18) — ÉTABLI

Six variantes du drainage profond et du temps de séjour de l'aquifère, passes avant à poids gelés sur l'Outaouais, la Gaspésie, le Saint-Laurent nord-ouest et le Saguenay, jugées sur les puits du réseau de suivi et sur le débit tenu de côté de 2022 à 2024.

| Variante | Recharge mm/an | Mois de recharge maximale | Écart-type du débit | KGE | Corrélation des anomalies | Amplitude simulée |
| --- | --- | --- | --- | --- | --- | --- |
| Témoin | 100 | août | 1,000 | 0,65 à 0,71 | -0,36 | 2 mm |
| Taux x2, séjour long | 292 | juillet | 0,816 | 0,584 | -0,20 | 15 mm |
| Taux x3, séjour long | 458 | juillet | 0,589 | 0,247 | -0,15 | 16 mm |
| Taux x5 | 649 | juin | 0,280 | -0,087 | +0,23 | 6 mm |
| Taux x5, séjour long | 649 | juin | 0,275 | -0,102 | +0,03 | 21 mm |
| Taux x3, courbure de Campbell | 326 | juin | 0,714 | 0,577 | +0,01 | 24 mm |

Trois enseignements. Le maximum de recharge passe d'août à juillet puis à juin, jamais à avril ni à mai : le taux de drainage n'a pas de prise sur la saison. Le volume, lui, quitte le domaine plausible dès la première dose, la recharge du témoin valant déjà 100 mm par an. Et le taux le plus fort essayé, qui seul rend la corrélation des anomalies franchement positive, est la borne supérieure du domaine autorisé : une dose quatre fois plus grande donne des résultats identiques au chiffre près, la valeur étant écrêtée.

La courbure de Campbell mérite d'être retenue pour elle-même. Au même taux, elle rend le KGE de 0,247 à 0,577 et l'écart-type du débit de 0,473 à 0,714 par rapport au témoin, parce que le drainage se coupe quand la couche se vide au lieu de couler en continu. Elle donne aussi la meilleure amplitude de nappe de la série, 24 mm, dans la plage plausible de 13 à 130 mm. Elle ne déplace pas la phase.

Un réservoir linéaire ne peut pas donner à la fois l'amplitude et la phase, son amplitude et son retard étant réglés par le même coefficient. La mesure le montre directement : à vidange rapide, le taux x5 donne une corrélation des anomalies de +0,23 pour 6 mm d'amplitude ; en allongeant le séjour, l'amplitude monte à 21 mm mais la corrélation retombe à +0,03 et la corrélation saisonnière à -0,47. La nappe mesurée a les deux, 0,93 m de battement et un maximum juste après la fonte, ce qui exige une recharge en impulsion de printemps et non un filet annuel.

**Porte de gel innocentée.** La couche 3 du témoin atteint sa teneur en eau maximale en avril, mois où la recharge atteint son minimum annuel. Seul un facteur multiplicatif peut produire cette inversion, et la porte qui divise le drainage profond par deux sur sol gelé en était le candidat. Elle a été rendue jugeable et retirée : la recharge annuelle passe de 55 à 59 mm sur l'Outaouais, celle d'avril de 0,09 à 0,10 mm par jour, le maximum reste en août, et le débit ne bouge pas, le KGE valant 0,652 contre 0,654 en Outaouais, 0,784 contre 0,775 en Gaspésie et 0,677 contre 0,711 au Saint-Laurent nord-ouest. La porte n'est pas la cause.

Le candidat restant est numérique. La boucle de sous-pas du sol s'arrête à un plafond d'itérations, et le bloc qui referme le bilan fait ruisseler la pluie du temps non traité et prélève son évapotranspiration, sans jamais accumuler son drainage profond ni son écoulement hypodermique. La documentation de ce bloc situe la fuite qu'il corrige à 210 mm par an sur l'Outaouais, concentrée en mars-avril et en octobre-décembre, nulle l'été, et l'audit de fermeture situe 86 pour cent du déficit de débit en avril. Aux mois où la nappe se recharge, l'eau qui aurait dû percoler serait donc comptée en ruissellement de surface. La part de journée non traitée est désormais exposée comme diagnostic, et sa mesure par mois tranchera.

---

## R119 — Au plafond de sous-pas employé, le ruissellement de surface de la colonne est un artefact de discrétisation (2026-09-18) — ÉTABLI

Mesure autonome sur la colonne seule, en double précision, sol de limon silteux uniforme, pluie de 40 mm par jour pendant trois jours, sans gel. Production cumulée, en millimètres, selon le plafond d'itérations de la boucle de sous-pas.

| Plafond | Surface | Hypodermique | Drainage profond | Total |
| --- | --- | --- | --- | --- |
| 24 | 96,81 | 0,18 | 0,00 | 96,99 |
| 64 | 73,19 | 0,25 | 0,00 | 73,45 |
| 128 | 46,53 | 0,29 | 0,01 | 46,82 |
| 256 | 4,44 | 0,29 | 0,01 | 4,75 |
| 512 | 0,00 | 0,29 | 0,01 | 0,30 |
| 1152 | 0,00 | 0,29 | 0,01 | 0,30 |

Le sol est à 80 pour cent de sa teneur en eau à saturation, situation courante et non extrême. La solution convergée ne produit aucun ruissellement de surface : les 120 mm de pluie s'infiltrent. Le plafond employé en production en produit 73. La totalité de ce ruissellement vient de la troncature de la boucle. Sur un sol à 95 pour cent de saturation, la moitié du ruissellement reste artificielle, 93,33 mm contre 44,37 convergés, et l'écoulement hypodermique est divisé par sept, 0,30 contre 2,08.

La cause est arithmétique. Le sous-pas est plafonné à une heure dès qu'il y a infiltration, donc une journée de pluie exige au moins vingt-quatre itérations, et chaque subdivision imposée par la condition de Courant multiplie ce nombre. Un plafond de 48 ou de 64 ne laisse presque aucune marge. Le bloc qui referme le bilan de masse verse alors en ruissellement de surface la pluie du temps non traité, ce qui ferme le bilan sans rien dire du chemin de l'eau.

Portée. Le défaut agit là où le sol est humide et la pluie forte, c'est-à-dire à la fonte et aux crues. Il porte la même signature que trois constats antérieurs restés sans cause commune : la recharge simulée minimale en avril, le déficit de débit dont l'audit de fermeture situe 86 pour cent en avril, et les bassins jugés trop réactifs. Il pourrait aussi expliquer une partie des compensations obtenues par calage sur la conductivité de la première couche.

Corroboration déjà écrite dans le code, sans avoir été rassemblée. Sur la Gaspésie, dont la conductivité à saturation vaut 4,6 fois celle de l'Outaouais, le plafond mord bien davantage, et les teneurs en eau des deux premières couches y valent 0,63 et 0,55 de celles d'Hydrotel, contre 0,96 et 0,95 en Outaouais. Le sol gaspésien du modèle est sec d'un tiers parce que son eau part en ruissellement au lieu de s'infiltrer.

Coût, mesuré sur 3 412 tronçons sans compilation, pour une journée saturée sous 40 mm de pluie. Avec sortie anticipée de la boucle : 541 ms par jour simulé à 64 sous-pas, 2 954 à 512. En mode statique : 855 ms à 64, 5 407 à 512. Sur une journée ordinaire, sol à 80 pour cent de saturation sous 20 mm, le plafond ne change rien, 300 ms qu'il vaille 64 ou 512, la boucle sortant dès le temps épuisé. Le plafond relevé ne se paie donc que les jours où il sert. La production n'en profite pas : la compilation du sol impose le mode statique, donc les 64 sous-pas s'exécutent tous chaque jour, et au-delà de 64 la compilation se désactive d'elle-même, le graphe devenant trop profond. Le réglage actuel paie le coût plein tous les jours et tronque les jours qui comptent.

Réserves. Mesure sur la colonne isolée en mode statique, à pluie constante et sol uniforme. En production la boucle sort dès que le temps du jour est épuisé, si bien que le plafond ne mord que les jours humides ou intenses. L'effet à l'échelle d'une région, son coût en temps de calcul et son effet sur le KGE restent à mesurer.

Le diagnostic qui manquait est désormais exposé : la part de journée que la boucle ne traite pas sort avec les séries journalières.

---

## R120 — La saison de la recharge simulée est le négatif de la troncature de la boucle de sous-pas (2026-09-18) — ÉTABLI

Mesure directe, aux conditions exactes de production, sur quatre territoires, 2000 à 2024. La part de chaque journée que la boucle de sous-pas du sol ne traite pas est désormais exposée avec les séries journalières. Le KGE des quatre passes reproduit celui du témoin au millième près, le diagnostic n'ayant rien changé au calcul.

| Territoire | Part non traitée, annuelle | Avril | Août | Couples jour-tronçon à plus de la moitié | Corrélation avec la recharge mensuelle |
| --- | --- | --- | --- | --- | --- |
| Saguenay | 0,29 | 0,34 | 0,15 | 0,37 | -0,94 |
| Outaouais | 0,32 | 0,62 | 0,08 | 0,38 | -0,94 |
| Gaspésie | 0,36 | 0,49 | 0,11 | 0,40 | -0,99 |
| Saint-Laurent nord-ouest | 0,40 | 0,58 | 0,16 | 0,50 | -0,98 |

Entre 29 et 40 pour cent de chaque journée simulée n'est pas traitée, et plus de la moitié de la journée sur 37 à 50 pour cent des couples jour-tronçon. Le maximum tombe au printemps et le minimum en été. Le cycle mensuel de la recharge est l'image inversée de ce cycle, à une corrélation de -0,94 à -0,99 selon le territoire.

Conséquence. Ce que le modèle appelle sa saison de recharge n'est pas une propriété de la physique de sa colonne : c'est la trace des mois où son schéma numérique échoue. En Outaouais, la recharge d'avril vaut 0,09 mm par jour, minimum annuel, alors que 62 pour cent de la journée d'avril n'est pas traitée et que la teneur en eau de la troisième couche y atteint son maximum. L'eau de la fonte qui devrait percoler est versée en ruissellement de surface par le bloc de fermeture du bilan.

Ce constat explique d'un seul mécanisme la nappe qui ne respire pas, le déficit de débit dont 86 pour cent tombe en avril, et l'échec de toutes les variantes du drainage profond essayées le même jour : elles réglaient une loi que la troncature rend inopérante quatre jours sur dix.

Réserves. Le lien avec l'assèchement des couches de surface, net sur trois territoires, ne tient pas sur quatre : le Saguenay tronque le moins et n'a pas la première couche la plus humide. Texture et climat diffèrent aussi d'un territoire à l'autre. Ce qui reste à mesurer est l'effet d'un plafond suffisant sur le KGE, sur la recharge et sur le temps de calcul à l'échelle d'une région.

---

## R121 — Ce n'est pas le temps de séjour de l'aquifère qui interdit le battement mesuré, c'est la forme de la recharge (2026-09-18) — ÉTABLI

Banc autonome sur une nappe libre fictive, sans réseau ni routage : recharge imposée de 150 mm par an, porosité de drainage 0,05, lit du cours d'eau à 8 m sous le sol. Cible, mesurée sur les puits appariés du réseau de suivi : 0,93 m de battement, maximum une trentaine de jours après celui de la recharge. La formulation porte la profondeur de la surface libre en variable d'état, une loi stock-débit en carré de la charge selon Dupuit-Boussinesq, et une extraction depuis la zone saturée en rampe linéaire jusqu'à une profondeur d'extinction.

Le schéma converge au pas journalier : le battement vaut 1,5984 m avec un seul sous-pas et 1,5916 m avec soixante-quatre, quatre dixièmes de pour cent d'écart. Contrairement à la colonne de sol, cette pièce ne coûte rien en temps de calcul. Le cas linéaire sous recharge constante reproduit sa solution analytique à 2 pour 10^13.

**Recharge sinusoïdale, réservoir linéaire.** Le battement vaut 0,162 m pour un temps de réponse de 10 jours, 0,623 m pour 50 jours, 0,917 m pour 200 jours et 0,944 m pour 400 jours, avec des retards de 9, 41, 74 et 82 jours. La simulation suit la théorie à un pour cent. Atteindre la cible exige donc un retard de 74 à 82 jours, près de trois fois celui qui est mesuré : aucun réglage ne s'en sort, ce qui confirme par le calcul l'impossibilité constatée le même jour sur les variantes.

**Même réservoir, recharge en impulsion de fonte.** Le battement vaut 1,019 m pour un temps de réponse de 25 jours, avec 12 jours de retard, et 1,352 m pour 50 jours avec 16 jours. Le verrou saute. Un réservoir chargé par une impulsion culmine à la fin de l'impulsion quel que soit son temps de vidange, alors qu'un forçage sinusoïdal impose le déphasage de son filtre.

C'est le renversement du diagnostic. Le temps de séjour de l'aquifère n'est pas le paramètre fautif : quelques dizaines de jours suffisent, valeur compatible avec les récessions de débit observées et donc avec le champ actuel. Ce qui manque est une recharge en impulsion, que la colonne ne produit pas, puisqu'elle livre un filet quasi constant maximal en août. Le conflit entre le débit et la nappe n'est pas une incompatibilité de paramètres.

**Apport de la loi non linéaire.** À dix jours de temps de réponse, la loi en carré de la charge donne 1,054 m de battement contre 0,576 m pour la loi linéaire, à retard égal. Elle permet donc un grand battement tout en gardant une réponse de débit rapide, ce qui préserve les récessions.

**Apport de l'extraction depuis la zone saturée.** Sans elle, le minimum de la nappe tombe en mars, le réservoir se vidant jusqu'à la fonte suivante. Modulée par la demande atmosphérique et avec une profondeur d'extinction supérieure à celle où la nappe s'établit, 6,2 m ici, elle place le creux en octobre à 4 mm par jour de demande maximale et en septembre à 8 mm, là où les puits le mesurent. Appliquée à taux constant sur l'année elle ne déplace que la moyenne : la modulation saisonnière est ce qui compte.

Réserves. Cas fictif à paramètres uniformes, recharge imposée analytiquement. Le battement obtenu, de 1,6 à 2,2 m, dépasse la cible de 0,93 m, ce qui signifie seulement que les paramètres restent à ajuster ; le point est que la cible est désormais à l'intérieur de l'ensemble atteignable alors qu'elle en était dehors. Rien n'est encore branché sur une recharge simulée ni sur un puits réel.

---

## R122 — Une nappe libre branchée sur la recharge actuelle redresse le cycle, mais seule une recharge en impulsion le cale (2026-09-18) — ÉTABLI, RÉVISÉ LE JOUR MÊME

Nappe libre branchée sur la recharge simulée aux nœuds portant un puits, 62 puits sur quatre territoires, paramètres posés à des valeurs plausibles et non ajustés : temps de réponse de 100 jours, porosité de drainage 0,05, lit du cours d'eau à 8 m, extraction maximale de 4 mm par jour jusqu'à 9 m de profondeur, modulée par la demande atmosphérique.

| Forçage | Corrélation des anomalies | Corrélation du cycle saisonnier | Mois le plus haut simulé | Porosité de drainage impliquée |
| --- | --- | --- | --- | --- |
| Aquifère actuel | -0,36 | -0,50 | août | négative |
| Recharge simulée telle quelle | -0,23 | +0,45 | avril | 0,026 |
| Même recharge, corrigée de la troncature | -0,22 | +0,50 | avril | 0,030 |
| Impulsion de fonte, même total annuel | +0,08 | +0,74 | mai | 0,056 |
| Mesuré | | | mai | |

Avec exactement la recharge qui donne aujourd'hui un aquifère plat culminant en août, le module renverse le cycle saisonnier de -0,50 à +0,45 et place le maximum en avril. Le mécanisme responsable est l'extraction depuis la zone saturée quand la nappe est à portée des racines : elle creuse l'étiage d'été et place donc le maximum au sortir de la recharge hivernale. Le modèle actuel ne peut pas la représenter, son flux souterrain ne pouvant pas s'inverser.

Cela corrige le constat du même jour qui faisait de la forme de la recharge le verrou. L'impulsion de fonte ajoute par-dessus, portant le cycle de 0,45 à 0,74 et calant le mois exactement, mais elle n'est pas la condition nécessaire. La correction de la troncature n'apporte que 0,05 de corrélation saisonnière, confirmant que le défaut numérique pèse sur le volume de la recharge de printemps plus que sur la dynamique de la nappe.

La porosité de drainage impliquée vaut 0,026 à 0,056 selon le forçage, dans la plage du roc fracturé et du till, ce qui est cohérent avec des puits dont la majorité captent dans le socle. Elle est obtenue sans aucun ajustement. L'aquifère actuel, lui, rendait une porosité négative sur la totalité des puits : le module est donc commensurable aux mesures, donc identifiable, ce que le précédent n'était pas.

**Révision le jour même, après application du filtre de recevabilité.** Les chiffres ci-dessus mélangeaient 82 puits en nappe libre, 81 captifs et 20 semi-captifs. Un puits captif mesure une charge transmise et non le remplissage d'un réservoir : sa dynamique n'a pas à ressembler à celle d'une nappe libre. Sur les 27 puits en nappe libre et non influencés par un pompage, la corrélation du cycle saisonnier vaut +0,12 avec la recharge telle que la colonne la produit, +0,26 une fois celle-ci corrigée de la troncature, et +0,72 avec l'impulsion de fonte. La porosité de drainage impliquée vaut 0,034 à 0,070, toujours plausible.

La conclusion change donc de sens. L'extraction estivale depuis la zone saturée reste indispensable, mais elle ne suffit pas : sur les puits recevables, la forme de la recharge fait passer le cycle de 0,12 à 0,72. Les deux mécanismes sont nécessaires, et la colonne doit produire une impulsion de printemps. L'énoncé initial, qui tenait l'impulsion pour accessoire, était un artefact des puits captifs.

Réserves. Vingt-sept puits seulement après filtrage. La corrélation des anomalies mensuelles reste mauvaise, +0,07 au mieux : le cycle moyen est juste, la variation d'une année à l'autre ne l'est pas. Aucun paramètre n'est calé et le module tourne hors de la colonne, sans rétroaction sur le débit.

---

## R123 — Les puits identifient la présence du mécanisme souterrain, pas la valeur de ses paramètres (2026-09-18) — ÉTABLI

Balayage de chaque paramètre de la nappe libre sur 27 puits en nappe libre non influencés, quatre territoires, recharge simulée telle que la colonne la produit.

| Paramètre | Plage balayée | Corrélation du cycle saisonnier obtenue | Corrélation des anomalies |
| --- | --- | --- | --- |
| Extraction depuis la zone saturée | absente, puis 2 à 16 mm/j | -0,47 puis +0,02 à +0,12 | -0,35 puis -0,22 à -0,32 |
| Profondeur d'extinction | 3 et 6 m, puis 9 à 30 m | -0,47 puis +0,12 à +0,22 | -0,35 puis -0,14 à -0,32 |
| Temps de réponse | 20 à 400 j | +0,06 à +0,19 | -0,32 partout |
| Exposant de la loi stock-débit | 1 à 5 | +0,06 à +0,18 | -0,31 à -0,32 |

Les deux premières lignes sont des effets de seuil et non des sensibilités continues. Que l'extraction existe, et que sa profondeur d'extinction atteigne la nappe, fait basculer la corrélation saisonnière de -0,47 à +0,12 et le mois du maximum d'octobre à avril. Entre 2 et 8 mm par jour la mesure ne distingue presque rien. Le temps de réponse et l'exposant ne déplacent la corrélation que de 0,12 sur toute leur plage plausible.

Les anomalies mensuelles ne voient AUCUN paramètre de l'aquifère : elles restent à -0,32 quel que soit le temps de réponse, l'exposant ou l'intensité de l'extraction. Seule la profondeur d'extinction les bouge, de -0,32 à -0,14. L'échec interannuel ne vient donc pas de l'aquifère et aucun réglage de l'aquifère ne le corrigera : il vient de la recharge.

**Identifiabilité spatiale de la porosité de drainage : non démontrée.** La valeur ajustée par puits vaut 0,070 en médiane sur les 18 puits en dépôts granulaires et 0,034 sur les 9 puits dans le roc. Les médianes vont dans le sens attendu et chacune tombe dans sa plage physique, mais les distributions se recouvrent, le test de rang donne p = 0,46, et la nature de l'aquifère n'explique que 4 pour cent de la variance du logarithme des valeurs ajustées. Laisser un paramètre libre par puits reviendrait donc, en l'état, à laisser chaque puits absorber son erreur locale sans qu'aucune covariable ne le prédise.

Conséquence pour le terme de perte. Il doit d'abord porter sur ce que les puits identifient réellement, la présence du mécanisme et la phase, en mode anomalies et sans paramètre libre supplémentaire. L'identifiabilité spatiale de la porosité est une question à reposer quand la recharge sera correcte, puisqu'elle est aujourd'hui mesurée à travers deux défauts connus, la troncature de la boucle de sous-pas et l'absence d'impulsion de printemps.

Réserves. Vingt-sept puits dont neuf dans le roc, échantillon faible pour séparer deux distributions larges. La porosité ajustée absorbe aussi l'erreur sur l'amplitude de la recharge, qui varie d'un tronçon à l'autre. La nature binaire roc contre dépôts est une covariable grossière devant la lithologie du SIGÉOM et les dépôts de surface, disponibles par tronçon et non encore essayés ici.

---

## R124 — La dose de percolation supprime aussi la troncature : les variantes du 2026-09-18 étaient confondues (2026-09-19) — ÉTABLI

Le diagnostic de temps non traité, appliqué aux mêmes variantes de taux de percolation, montre que relever le taux SUPPRIME le défaut numérique. Sur l'Outaouais, la part de journée non traitée passe de 0,32 au témoin à 0,06 au taux doublé et à 0,00 au taux quintuplé ; en avril, de 0,62 à 0,17 puis à 0,00. Sur le Saint-Laurent nord-ouest, de 0,40 à 0,28 puis 0,05. La raison est que la conductivité de Campbell chute bien plus vite que le stock quand le sol s'assèche, si bien que la condition de Courant se relâche et que la boucle a le temps de finir sa journée.

Chaque dose de taux corrigeait donc simultanément deux choses, et une part du déplacement de phase attribué la veille à la loi de drainage venait du schéma. Le déconfondage se lit en comparant les deux passes à troncature nulle. Avec la couche 3 saturée à 0,521, la recharge vaut 80 mm par an, plate, maximum en février. Avec la couche 3 désaturée à 0,276, elle vaut 612 mm par an, maximum en mai. C'est donc la désaturation de la couche 3 qui déplace la recharge vers le printemps, et non le taux en lui-même.

**Front des compromis, Outaouais, avec la courbure de Campbell.**

| Configuration | Recharge | Mois max | Couche 3 | Non traité | Écart-type du débit | KGE |
| --- | --- | --- | --- | --- | --- | --- |
| Témoin | 55 mm/an | août | 0,517 | 0,32 | 699 | 0,654 |
| Taux x2, exposant 11 | 270 | juin | 0,504 | 0,12 | 535 | 0,598 |
| Taux x5, loi linéaire | 612 | mai | 0,276 | 0,00 | 225 | -0,180 |
| Taux x5, exposant 11 | 437 | mai | 0,475 | 0,02 | 414 | 0,304 |
| Taux x5, exposant 15 | 411 | mai | 0,485 | 0,03 | 473 | 0,421 |

Aucun point ne satisfait les quatre critères ensemble. Le maximum de mai exige un taux fort, qui porte le volume à 411 mm par an contre les 100 à 250 plausibles et fait tomber le KGE à 0,42. La courbure adoucit la facture sans l'annuler : à taux égal elle rend l'écart-type du débit de 225 à 473 et le KGE de -0,18 à 0,42. Au Saint-Laurent nord-ouest, le taux doublé avec exposant 11 donne 200 mm par an, volume plausible, pour un KGE de 0,655 contre 0,677, mais le maximum reste en juillet.

Ces essais sont tous faits SANS couplage entre la nappe et la colonne. Or la couche 3 ne peut s'y désaturer qu'en permanence, sous l'effet d'un taux élevé, alors qu'une vraie nappe la ressuie par saison sous l'effet du rabattement estival. C'est ce que le couplage livré le même jour doit permettre.

---

## R125 — La couche 3 retient en permanence 289 à 465 mm d'eau gravitaire, et c'est la cause de la recharge inerte (2026-09-19) — ÉTABLI

Teneur en eau simulée comparée à la capacité au champ du champ spatial, moyennes 2000-2024, quatre territoires. L'eau gravitaire est la lame retenue au-dessus de la capacité au champ, celle que la gravité draine par définition.

| Territoire | Couche 1 | Couche 2 | Couche 3 |
| --- | --- | --- | --- |
| Outaouais | +2 mm | -1 mm | +465 mm |
| Saint-Laurent nord-ouest | -7 mm | -32 mm | +325 mm |
| Gaspésie | -6 mm | -11 mm | +342 mm |
| Saguenay | -1 mm | -17 mm | +289 mm |

Les deux couches supérieures se tiennent à leur capacité au champ à quelques millimètres près, comportement attendu d'un sol qui se remplit, se ressuie, puis s'assèche sous l'effet des racines. La troisième est engorgée sur les quatre territoires, de 289 à 465 mm, soit cinq à six fois la recharge annuelle simulée. Sur l'Outaouais elle se tient à 0,517 de teneur en eau quand sa capacité au champ vaut 0,345 et sa porosité 0,519 : elle est à saturation toute l'année.

La cause est structurelle et non numérique. Les deux premières couches ont un drainage latéral et l'essentiel des racines ; la troisième n'a que la percolation vers l'aquifère, dont la constante de temps est ajustée sur les récessions de DÉBIT et non sur la physique du sol. Elle ne peut donc pas se ressuyer.

Deux conséquences en chaîne, qui expliquent tout le chantier. La couche n'a aucune capacité disponible à la fonte, si bien que la crue de printemps repart en surface au lieu de s'infiltrer. Et sa teneur en eau ne variant pas de plus de trois pour cent sur l'année, la recharge qui lui est proportionnelle ne varie pas non plus, d'où une recharge plate dont le maximum tombe en août.

Remède livré le même jour, sous `ETL_L3_TAU` : drainage de l'eau au-dessus de la capacité au champ avec une constante de temps de quelques jours, et rien en dessous. C'est la formulation standard, que Raven porte sous plusieurs noms dans sa famille de percolation. La recharge devient alors égale à ce qui percole réellement depuis la couche sus-jacente, donc pilotée par le climat et non par une constante de calage. Effet sur le débit et sur le volume de recharge à mesurer.

---

## R126 — La nappe libre ne coûte rien au débit, et le couplage par étranglement ne peut rien guérir (2026-09-19) — ÉTABLI

Nappe libre branchée dans la colonne, passes avant à poids gelés sur l'Outaouais et le Saint-Laurent nord-ouest, profondeur de la surface libre en variable d'état, loi en carré de la charge, extraction depuis la zone saturée pilotée par l'évapotranspiration potentielle de la colonne.

| Variante | Eau gravitaire L3 | Recharge | Mois max | Écart-type du débit | KGE | Battement | Mois haut |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Témoin | 382 mm | 76 mm/an | août | 1,000 | 0,665 | — | — |
| Nappe libre | 382 | 76 | août | 1,019 | 0,681 | 0,083 m | juin |
| Nappe et taux x2 courbé | 361 | 235 | juin | 0,849 | 0,648 | 0,163 m | août |
| Nappe et extraction forte | 361 | 235 | juin | 0,859 | 0,639 | 0,243 m | mars |
| Taux x2 courbé seul | 361 | 235 | juin | 0,836 | 0,627 | — | — |

La nappe libre seule est un gain net : KGE de 0,665 à 0,681 et écart-type du débit de deux pour cent. Elle récupère aussi une part de ce que la correction du drainage coûte au débit, 0,648 contre 0,627 sans elle. L'extraction forte porte le battement à 0,243 m et ramène le mois le plus haut d'août à mars, donc du côté du printemps, tout en restant loin des 0,93 m mesurés et d'un maximum en mai.

**Couplage par étranglement : sans effet, et la raison est structurelle.** Le facteur de gradient qui éteint le drainage quand la surface libre approche la base du sol s'active bien : avec le lit du cours d'eau posé à 4 m, la nappe s'établit à 3,31 m contre une base de sol à 3,17 m, et le facteur vaut 0,14. Mais la teneur en eau de la couche 3, la recharge et l'eau gravitaire restent identiques au millimètre près, 0,517, 55 mm par an et 457 mm. Un étranglement ne peut pas guérir un système déjà étranglé : le drainage profond est trop lent et non trop rapide. La pièce reste utile comme garde-fou une fois le drainage gravitaire actif, pas comme remède.

Effet mécanique seul du balayage de géométrie : une nappe plus proche de la surface bat davantage, 0,127 m à 4 m de profondeur de lit contre 0,048 m à 8 m, la loi en carré de la charge y répondant plus lentement.

**Identifiabilité croisée constatée.** L'évapotranspiration totale passe de 459 mm par an au témoin à 476 avec la nappe et 517 avec l'extraction forte, contre 400 à 500 mm par an pour l'évapotranspiration boréale réelle. L'intensité de l'extraction est donc contrainte par l'évapotranspiration satellitaire, alors que les puits ne la distinguent presque pas entre 2 et 8 mm par jour. Deux observations indépendantes tiennent la même pièce par deux bouts.

---

## R127 — Drainer l'eau gravitaire de la couche 3 fait tomber les trois défauts ensemble, au prix du volume (2026-09-19) — ÉTABLI

Loi de drainage de l'eau au-dessus de la capacité au champ, constante de deux jours, passe avant à poids gelés sur l'Outaouais. La loi est convergée dès trente-deux sous-pas à cette constante, vérification faite avant lecture.

| Outaouais | Témoin | Drainage gravitaire |
| --- | --- | --- |
| Teneur en eau de la couche 3 | 0,507 à 0,523 | 0,338 à 0,352 |
| Eau gravitaire immobilisée | +457 mm | -7 mm |
| Recharge, cycle mensuel | 0,09 à 0,21 mm/j | 0,35 à 5,51 mm/j |
| Amplitude saisonnière de la recharge | 2,28 | 15,9 |
| Mois de recharge maximale | août | avril |
| Part de journée non traitée | 0,32 | 0,00 |
| Recharge annuelle | 55 mm | 537 mm |
| Écart-type du débit | 699 | 742 |
| Débit moyen | 1131 m³/s | 1268 |
| KGE médian tenu de côté | 0,654 | 0,464 |

Les trois défauts du chantier tombent d'un seul coup. La couche descend exactement à sa capacité au champ et y reste, donc l'eau gravitaire disparaît. La recharge devient une impulsion de printemps, maximum en avril, d'amplitude saisonnière seize contre deux. Et la troncature de la boucle de sous-pas s'annule d'elle-même, le sol n'étant plus saturé : c'était bien la saturation permanente qui resserrait la condition de Courant.

Le prix est un volume de 537 mm par an, soit à peu près tout l'écoulement du bassin, et 0,19 de KGE. La cause est structurelle : toute la percolation descend désormais, alors que la couche saturée la refoulait vers l'écoulement hypodermique. Dans un sol réel, l'eau qui traverse le profil rencontre un dépôt et un socle bien moins perméables, et l'essentiel repart latéralement au-dessus de cette interface, ce qui EST l'écoulement hypodermique. La couche 3 du modèle se draine tout droit, sans ce plafond.

Remède livré le même jour sous `ETL_L3_KSUB` : la percolation est plafonnée par la conductivité du substratum, l'excès restant dans la couche et repartant latéralement par la cascade de saturation déjà présente. Ce plafond est une propriété du dépôt et du socle, donc prédictible par la géologie ingérée la veille, et non un paramètre de calage. Valeurs en cours d'épreuve.
