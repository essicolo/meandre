---
name: redaction
description: Rédiger ou réviser un document destiné à un lecteur humain (rapport, qmd, présentation, article). OBLIGATOIRE avant d'écrire une section, et impose une passe de révision phrase par phrase avant de livrer.
---

# Rédaction des livrables de méandre

Deux échecs fondent ce skill. Le 2026-08-31, un rapport livré illisible : vocabulaire interne du projet, nombres sans grandeur ni unité, et une définition dimensionnellement fausse (« la pression est le volume prélevé rapporté au débit », un volume sur un débit donne un temps). Essi a dû réécrire chaque phrase. Une liste de règles ne suffit pas : la première version de ce skill existait déjà quand ces phrases ont été écrites. Ce qui fonctionne : imiter un étalon de voix, et réviser mécaniquement.

## 1. L'étalon de voix est la prose d'Essi

Avant d'écrire, lire les paragraphes qu'Essi a déjà écrits dans le document, et les imiter. À défaut, imiter l'exemple ci-dessous, tiré du rapport sur les prélèvements (sa prose, verbatim) :

> La naturalisation des débits, définie ici par le retrait des prélèvements et rejets, ne peut être rigoureusement effectuée que si le modèle permet d'effectuer des études de sensibilité. Le corollaire d'une telle approche est que l'entraînement du modèle impute les erreurs de prédiction aux bons paramètres. En effet, la modélisation hydrologique est un problème inverse cherchant à reconstruire les paramètres du modèle à l'aide de débits observés.

Ce qui caractérise cette voix : le terme est défini au moment où il apparaît (« définie ici par ») ; la motivation précède la description ; les enchaînements sont logiques et explicites (corollaire, en effet) ; aucune emphase, aucun chiffre nu, aucun sigle non défini ; le registre est celui d'un hydrologue qui écrit pour des collègues, pas d'un journal de développement.

## 2. Exemples de transformation (fautes réelles, corrections réelles)

Écrit : « La pression est le volume prélevé rapporté au débit du tronçon : elle mesure la sollicitation. »
Pourquoi c'est faux : pression a un sens physique (force par surface) et la définition est dimensionnellement fausse.
Corrigé : « La fraction du débit prélevée est le débit prélevé sur un tronçon, en mètres cubes par seconde, rapporté au débit naturalisé du même tronçon. Elle est sans dimension et s'exprime en pourcentage. »

Écrit : « Le seuil pluie-neige hérité du calage, moins 2,2168 degrés, comptait comme pluie tout ce qui tombe à moins 2 degrés. »
Pourquoi c'est faux : jargon interne (hérité du calage), fausse précision (quatre décimales sans sens physique), unité absente.
Corrigé : « Le partage entre pluie et neige était fixé à -2,2 °C : toute précipitation tombant à une température supérieure était traitée comme de la pluie, ce qui est physiquement invraisemblable. »

Écrit : « 0,44 de médiane sur le tenu de côté. »
Corrigé : « un KGE médian de 0,44 sur la période d'évaluation 2022-2024, jamais utilisée pour l'entraînement. »

Écrit : « les plateformes s'effondrent » (pour des régions).
Pourquoi c'est faux : plateforme désigne une version calibrée d'Hydrotel (LN24HA), jamais une région. Et « s'effondrent » est de l'emphase.
Corrigé : « le KGE médian diminue de 0,3 dans quatre régions ».

## 3. Les tests mécaniques, appliqués phrase par phrase avant de livrer

La passe de révision est OBLIGATOIRE et se fait sur le document entier, une phrase à la fois :

1. Test du lecteur externe : un hydrologue qui n'a jamais lu le registre ni les conversations comprend-il cette phrase seule ? Sinon, réécrire.
2. Test dimensionnel : chaque définition (« X est Y rapporté à Z ») doit donner les bonnes unités. Un volume sur un débit est un temps, pas une fraction. Si les dimensions ne tombent pas juste, la phrase est fausse, pas maladroite.
3. Test des termes physiques : pression, charge, tension, capacité, conductivité, flux, intensité, énergie ne s'emploient que dans leur sens physique. Sinon, nommer la grandeur par sa construction.
4. Test du nombre : grandeur, unité en clair, période ou population. Arrondir à la précision qui a un sens physique.
5. Test du vocabulaire interne : tenu de côté, champion, gen1, recette, plateforme (pour une région), ancrage, forçage -hyb, kge_med, R19, dette 6 sont interdits. Traduire : période d'évaluation indépendante 2022-2024 ; modèle retenu ; configuration retenue ; version calibrée d'Hydrotel ; paramètre repris du calage d'Hydrotel ; données météorologiques d'entrée en précisant la correction ; KGE médian par station ; énoncer le fait au lieu du numéro.
6. Typographie d'Essi : pas de tirets cadratins, pas de gras ni d'italique dans ma prose, pas de guillemets décoratifs, pas de hard-wrap. Sa prose à lui est intouchable.

## 4. Ce que le document n'est pas

Pas un journal : aucune narration du travail (« nous avons ensuite découvert »), aucune histoire de bogue, aucune chronologie de diagnostic. L'état, pas la fabrication. Les limites s'énoncent factuellement, au même niveau que les résultats, jamais en excuse ni en aveu.

## 5. Structure par défaut d'un rapport de modélisation

Ce que le lecteur peut conclure, avec les conditions de validité. Comment c'est mesuré : données, méthode, périodes, ce qui est indépendant de quoi. Ce que ça ne dit pas. Annexes techniques. Chaque figure porte une légende autoportante : grandeur, unité, période, population, lecture des couleurs.
