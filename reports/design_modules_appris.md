# Design : modules de processus appris (plan d'intégration validé avec Essi 2026-07-21)

Principe : chaque processus mal contraint devient un petit module appris supervisé par SON observable ; la colonne Hydrotel clonée reste le squelette de conservation (extraction par couche, scaling factors, bilan). Aucun coefficient emprunté au calage Hydrotel (les ancrages sont gelés, cf. design_et_appris.md). Critère de sélection d'un candidat : (1) observable indépendant de la sortie du processus disponible à l'échelle des 15 régions, (2) goulot prouvé.

Discipline (leçon des pilotes conjoints, 2×20 h brûlées) : chaque phase passe par diagnostic gratuit sur l'existant, banc hors-ligne quand l'observable existe, pilote court 1 région, élargissement seulement ensuite. Critères de succès pré-enregistrés avant chaque run.

## Phase 0 : données barrages (indépendante, coût une soirée)

Répertoire des barrages MELCCFP (Données Québec, ~6000 ouvrages : coordonnées, hauteur, capacité de retenue, usage, gestionnaire) ingéré dans les 15 bases DuckDB, mappé au nœud le plus proche ; complément GRanD/GDW pour les grands réservoirs si utile. La tuyauterie DamData (lâchers imposés par nœud) existe déjà dans meandre/routing/dam.py, jamais alimentée.
Livrable gratuit : régression de l'écart méandre vs ensemble Hydrotel par station contre la capacité de retenue amont = quantifier la part de la régulation dans le déficit de l'est AVANT d'écrire un module.

## Phase 1 : ET appris (banc GAGNÉ 2026-07-21, intégration en cours)

Banc hors-ligne : R² 0.91 vs 0.82/0.77 pour McGuinness/Linacre calés au mieux, biais -1.5 % vs -12 %, gagnant dans chaque région tenue et chaque période. MLP suffit (mémoire inutile à 8 j). Vérif bilan : MOD16 cohérent ±20-30 % avec P-Q sur bassins naturels (biais chaud au sud +17..31 %, froid en taïga -20 % ; écarts CND/outm 041902 = régulation et dérivations, pas MOD16) → w_et reste contrainte douce.
Intégration : etp_channel (7e canal de forçage précalculé, module gelé, K_c bypassé). Run gasp-etl 12 epochs en cours ; repères : v4 0.489 (recette égale), v7 0.577 (ancrages). Si ≥ v7 : mono SAGU et MONT, puis pièce 1 du re-pilote conjoint.

## Phase 2 : fonte apprise (candidat n°1 suivant)

Goulot prouvé : le timing du freshet a tué les pilotes conjoints ; les ancrages fonte sont gelés et toxiques hors du sud (LABI 0.126). Observables : MOD10A1 couvert nival (loader import_modis_snow existe, présence en base à vérifier) = DATE de disparition par nœud ; relevés nivométriques MELCCFP = amplitude SWE.
Banc hors-ligne clone d'et_bench : leave-region-out, baselines degré-jour calé + ETI, critère pré-enregistré (battre les deux sur date de disparition ET SWE). Intégration comme modulation des taux de fonte, pilote 1 région.

## Phase 3 : régulation apprise (supervision indirecte, falsification petite échelle d'abord)

MLP d'exutoire pour les lacs À BARRAGE seulement : Q_out = f(stockage, Q_in, doy, attributs de l'ouvrage). Les lacs naturels gardent k/beta NeRF. Banc : SAGU (gamma résiduel 1.16-1.22 connu), critère = gamma vers ~1 sur les stations sous influence. Niveaux de réservoirs publics (CEHQ) en supervision directe si disponibles.
Limite assumée : les grands complexes (Manic, Lac-Saint-Jean) répondent à la demande électrique, pas à l'hydrologie ; pour eux, DamData avec lâchers observés imposés quand ils existent, le MLP pour la myriade de petits ouvrages.

## Phase 4 : re-pilote conjoint tout appris

3 régions, zéro ancrage, ET + fonte (+ régulation si prête) apprises, z_n coupés ou régularisés (pénalité ||z_n||², latent_lr_mult réduit — cf. discussion effet aléatoire 2026-07-20). Même critère pré-enregistré : aucune région ne régresse vs son mono held-out, au moins une gagne. MONT gagnait déjà (+0.01-0.05) avec le compromis non appris ; c'est le vrai test de la promesse une-recette-régionale-par-les-données.

## Coûts

Phase 0 une soirée ; bancs 1-2 : ~1 h GPU chacun, pilotes 3 h ; phase 3 la plus incertaine ; phase 4 ~8 h. Contraste voulu avec les pilotes à 20 h de l'ancienne méthode.
