---
name: redaction
description: Rédiger ou réviser un document destiné à un lecteur humain (rapport, qmd, présentation, article). OBLIGATOIRE avant d'écrire une section, et impose une passe de révision phrase par phrase avant de livrer.
---

# Rédaction des livrables de méandre

Deux échecs fondent ce skill. Le 2026-08-31, un rapport livré illisible : vocabulaire interne du projet, nombres sans grandeur ni unité, et une définition dimensionnellement fausse (« la pression est le volume prélevé rapporté au débit », un volume sur un débit donne un temps). Essi a dû réécrire chaque phrase. Une liste de règles ne suffit pas : la première version de ce skill existait déjà quand ces phrases ont été écrites. Ce qui fonctionne : viser un registre défini indépendamment des personnes, et réviser mécaniquement.

## 1. Le registre visé

Le standard n'est ni la prose de l'assistant ni celle d'Essi : c'est le registre des publications scientifiques en hydrologie de langue française (revues comme LHB Hydroscience, rapports de l'INRS ou de l'organisme d'expertise hydrique). Ses traits opérationnels :

- Chaque terme technique est défini à sa première occurrence, dans la phrase même où il apparaît.
- La motivation précède la description : on dit pourquoi une méthode est employée avant de dire comment.
- Les enchaînements logiques sont explicites (donc, en effet, en revanche, par conséquent), jamais implicites ni décoratifs.
- Aucune emphase (considérable, spectaculaire, s'effondre) : l'ampleur est portée par le chiffre, pas par l'adjectif.
- La voix passive ou impersonnelle domine ; le document décrit un état de connaissance, pas les auteurs au travail.
- Un paragraphe porte une idée, annoncée par sa première phrase.

Les paragraphes déjà écrits par Essi dans un document fixent le PLAN et l'ANGLE, jamais le standard de langue : ce sont des brouillons de travail, ils peuvent porter des coquilles, et ils lui appartiennent. On écrit autour, dans le registre ci-dessus, sans les modifier.

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
6. Typographie d'Essi : pas de tirets cadratins, pas de gras ni d'italique dans ma prose, pas de guillemets décoratifs, pas de hard-wrap. La prose d'Essi est intouchable.

## 3 bis. Les homonymes entre communautés

Un meme terme peut designer deux techniques sans rapport selon la communaute. Cas paye le 2026-08-31 : interpolation optimale designe le schema d'assimilation de Gandin en meteorologie (celui de CaPA), et une technique de redistribution des erreurs sur les debits en hydrologie operationnelle quebecoise. Regle : quand un terme technique vient d'une autre communaute que celle du lecteur, l'attribuer a sa source dans la phrase meme, et desamorcer explicitement l'homonyme si la communaute du lecteur en possede un.

## 4. Ce que le document n'est pas

Pas un journal : aucune narration du travail (« nous avons ensuite découvert »), aucune histoire de bogue, aucune chronologie de diagnostic. L'état, pas la fabrication. Les limites s'énoncent factuellement, au même niveau que les résultats, jamais en excuse ni en aveu.

## 5. Structure par défaut d'un rapport de modélisation

Ce que le lecteur peut conclure, avec les conditions de validité. Comment c'est mesuré : données, méthode, périodes, ce qui est indépendant de quoi. Ce que ça ne dit pas. Annexes techniques. Chaque figure porte une légende autoportante : grandeur, unité, période, population, lecture des couleurs.
