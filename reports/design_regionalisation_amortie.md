# Design : régionalisation amortie (modèle d'expérience)

Date : 2026-07-31. Statut : plan à valider avant implémentation.

## Problème que ça résout

Le déploiement actuel transfère un modèle entraîné sur UNE région (Gaspésie) au reste de la province. Mesuré le 2026-07-31 : hors de son domaine, le champ spatial appris rend des paramètres quasi constants (K_sat médian 0.0362-0.0369 de la Gaspésie à la Montérégie, k_gw 0.0451-0.0475). Le transfert n'apporte donc pas de régionalisation ; il apporte un jeu de paramètres global. Cause identifiée : les attributs territoriaux sont z-scorés PAR RÉGION, donc le nœud médian de chaque région arrive au réseau avec des attributs ~0 et reçoit les mêmes paramètres. L'entraînement conjoint, qui aurait pu exposer le modèle à plusieurs climats, échoue pour une raison indépendante (tout gradient supplémentaire dégrade le test hors-régime, doctrine du champion gelé).

La régionalisation amortie contourne les deux obstacles : au lieu d'entraîner UN modèle hydrologique sur plusieurs régions, on entraîne un PETIT MODÈLE DE RÉGRESSION sur les résultats de nos calibrations régionales déjà faites, et on s'en sert pour prédire les paramètres de départ partout.

## Données d'entraînement (déjà disponibles)

Une observation = un tronçon d'une région calibrée.
- Entrées (X) : attributs territoriaux BRUTS du tronçon (pente, élévation, textures sable/limon/argile, couverts forêt/agricole/urbain/humide/eau, aire drainée, ordre de Strahler, profondeur au socle, fraction lacustre), plus coordonnées, plus des covariables climatiques du bassin (précipitation et température moyennes du forçage, indice d'aridité). Extraction brute : `load_hydrotel(project_dir, normalise=False)`, déjà utilisée le 2026-07-30.
- Cibles (y) : paramètres calibrés du tronçon, extraits des checkpoints régionaux par `spatial_encoder.forward` (K_sat 1-3, porosités, Z2, Z3, C_f, T_melt, K_c, k_gw, krec, K_musk, x_musk).
- Sources actuelles : champions GASP (best-gasp-etl-ds), SAGU, MONT, SLSO historique, plus les runs v4/v7 régionaux archivés. Ordre de grandeur : ~10 000 tronçons, mais seulement 3-4 climats indépendants — limite à assumer.

## Modèle

XGBoost (ou GP à noyau anisotrope) par paramètre cible, avec deux exigences :
1. **Validation leave-one-region-out obligatoire.** Avec 3-4 régions, la seule métrique honnête est : entraîner sur les régions A,B,C et prédire D. Le R² intra-région serait trompeur (les tronçons voisins sont corrélés).
2. **Sorties bornées** aux plages physiques du modèle (mêmes bornes que le NeRF), et cibles en log pour les paramètres log-normaux (K_sat, k_gw, krec).

XGBoost pour la robustesse aux petits jeux de données et l'importance de variables directement lisible (quel attribut explique quel paramètre : c'est un résultat scientifique en soi). GP en variante quand on veut l'incertitude prédictive, indispensable pour les bassins non jaugés.

## Intégration

Le modèle d'expérience produit une carte de paramètres de départ pour toute région, jaugée ou non. Deux usages :
- **Initialisation** : remplace `init_from_literature()` par `init_from_regionalisation()`, en gardant les priors mesurés (récessions, bilan) qui restent prioritaires là où ils existent.
- **Prior d'entraînement** : la cible du terme de régularisation devient la prédiction régionalisée par tronçon, au lieu d'une constante de littérature.

Aucune modification du modèle hydrologique ni de la doctrine du champion gelé : c'est une couche amont qui fournit un meilleur point de départ.

## Boucle de capitalisation

Chaque nouvelle région calibrée est ajoutée au jeu d'entraînement et le modèle d'expérience est réajusté (quelques secondes). Sa qualité se suit par la validation leave-one-region-out : si l'erreur de prédiction des paramètres décroît en ajoutant des régions, la régionalisation apprend réellement ; si elle stagne, les attributs disponibles ne suffisent pas à expliquer les différences régionales, ce qui est un résultat négatif publiable et oriente vers l'acquisition d'autres descripteurs (géologie, drainage agricole, densité de milieux humides).

## Critère de succès pré-enregistré

Sur une région tenue : la carte de paramètres prédite doit donner, en inférence pure (sans entraînement local), un KGE held-out supérieur à celui du champion transféré ET à celui de la littérature corrigée par les mesures. Les trois configurations sont comparables en quelques minutes chacune, sur le même forçage et les mêmes adaptateurs mesurés.

## Étapes

1. Extraire les attributs bruts des 15 régions (script existant, une heure).
2. Constituer le jeu (X,y) depuis les checkpoints régionaux disponibles.
3. Ajuster XGBoost par paramètre, validation leave-one-region-out, rapport d'importance des variables.
4. Câbler `init_from_regionalisation()` et mesurer le critère de succès sur une région tenue.
5. Si le critère est atteint, la carte provinciale repart de cette initialisation ; sinon, documenter l'échec (attributs insuffisants) et conserver littérature + mesures.
