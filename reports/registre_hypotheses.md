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
7. **`pgrep` et `ps` du shell POSIX ne voient PAS les processus Windows** : cause de 4 incidents de contention en 3 jours, dont une nuit perdue et trois copies d'une même file prêtes à démarrer ensemble. Toute vérification de processus doit passer par PowerShell ou `tasklist`.
