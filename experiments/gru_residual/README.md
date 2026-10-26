# EN COURS, NE PAS SUPPRIMER

Expérience ouverte le 2026-10-22 à la demande d'Essi, pendant le nettoyage de la branche `nettoyage`. Ce dossier n'est rattaché à aucun pilote et rien dans le paquet ne l'importe : il ne doit être ni archivé, ni déplacé, ni supprimé par le nettoyage.

## Question

Un réseau récurrent placé en aval de la physique (correction du débit, éventuellement spatialisée par le champ spatial) peut-il modéliser ce qui échappe à la physique, et ce gain tient-il hors de la période d'entraînement et aux stations jamais vues ?

## Ce qui est déjà établi

Journal d'expériences, campagnes SLSO de l'été : un GRU minimal sur physique gelée, avec une correction multiplicative du débit bornée à ±30 %, fait passer le KGE médian 2022-2024 de 0,645 à 0,651 sur 24 stations ; le même nourri d'indicateurs d'état (sol, neige, gel) perd 0,015. Ces verdicts sont antérieurs au schéma semi-implicite du sol, au drainage de la couche 3, à la nappe libre et à l'évaporation de l'eau libre, et ne disaient rien des stations non jaugées.

## Plan

1. Plafond sans entraînement : part de l'écart entre débit observé et simulé qu'un modèle simple (ridge, arbres) prédit à partir du forçage et du débit simulé, en validation croisée par année et par station.
2. Structure de l'écart : saison, régime, station ; un écart systématique désigne un mécanisme physique à corriger, pas un réseau à entraîner.
3. Correction par tronçon sur l'apport latéral, poids partagés, conditionnée par le champ spatial, bornée, volume mesuré.
4. Entraînement sur physique gelée, puis conjoint en surveillant la dérive des paramètres physiques.
5. Généralisation aux stations retirées de l'entraînement, sur 2022-2024.

Seuil d'arrêt posé avant l'étape 1 : si la part prévisible hors échantillon, aux stations retirées, est inférieure à 0,05 de R² sur le logarithme du rapport observé sur simulé, l'expérience s'arrête.

## Suite

Le plan d'expériences qui en découle (ML à l'intérieur de la physique, candidates et épreuves) est dans `experiments/plan_ml/PLAN.md`, avec trois annexes (inventaire du code, littérature, observations).

## Fichiers

- `plafond.py` : étapes 1 et 2, sur les sorties du modèle provincial gelé (`D:/meandre-data/quebec/modele-provincial-2026-10/q-quebec-proba.npz`), aucune simulation.
- `RESULTATS.md` : verdicts au fil de l'eau.
