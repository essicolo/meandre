---
name: redaction
description: Rédiger ou réviser un document destiné à un lecteur humain (rapport, qmd, présentation, article), prose, figures et code inclus. OBLIGATOIRE avant d'écrire une section ou une figure, et impose de lancer verifier.py puis une relecture phrase par phrase avant de livrer.
---

# Rédaction des livrables de méandre

## Pourquoi ce skill existe

Trois livraisons ont échoué de la même manière. Le 2026-08-31, un rapport illisible : vocabulaire interne, nombres sans grandeur, une définition dimensionnellement fausse. Le 2026-09-14, le rapport provincial et sa présentation, relus avec attention, portaient encore 294 constats mécaniques. Parmi eux, 130 phrases de plus de 30 mots et 34 lignes de code mort commenté. Aussi 21 blocs de commentaires narratifs, des titres en forme de question, une section de 1 400 mots, des faits énoncés deux fois. Une liste de règles ne suffit pas : la première version de ce skill existait quand ces documents ont été écrits. Ce qui tient, c'est un registre défini indépendamment des personnes, une vérification mécanique, puis une relecture phrase par phrase.

## Quand l'appliquer

À chaque écriture ou révision d'un fichier qmd, md, tex ou de diapositives, y compris pour une seule figure ou un seul bloc de code. Le code d'un document est lu par le lecteur : il obéit aux mêmes exigences que la prose.

## Le registre visé

Le standard est celui des publications scientifiques en hydrologie de langue française (LHB Hydroscience, rapports de l'INRS, de l'organisme d'expertise hydrique). Chaque terme technique est défini à sa première occurrence, dans la phrase même. La motivation précède la description. Les enchaînements logiques sont explicites, jamais décoratifs. Aucune emphase : l'ampleur est portée par le chiffre. La voix impersonnelle domine, le document décrit un état de connaissance, pas les auteurs au travail. Un paragraphe porte une idée, annoncée par sa première phrase.

Les paragraphes déjà écrits par Essi fixent le PLAN et l'ANGLE, jamais le standard de langue. On écrit autour, sans les modifier.

## Les quatre familles de slop, et leur test

### 1. Prose : la longueur et le vide

Une phrase porte une affirmation, en 25 mots ou moins, 30 au plus. Une section fait 600 mots au plus ; au-delà, elle porte deux sujets et se scinde, ou elle répète et se coupe. Un fait est énoncé une fois, à l'endroit où le lecteur en a besoin ; s'il revient, c'est que le plan est faux. Les reformulations (c'est-à-dire, autrement dit) signalent une première formulation ratée : on garde la bonne et on supprime l'autre. Les qualificatifs de valeur (vraie épreuve, honnête, considérable, fortement) sortent, y compris des titres.

Test : le paragraphe coupé de moitié perd-il un fait ? Sinon, couper.

Écrit : « ### Pourquoi c'est la vraie épreuve », 211 mots. Corrigé : « ### Les stations transférées n'ont servi à aucun entraînement », 90 mots qui disent quelles stations, quelle période, quel résultat.

Écrit : « l'écart-type du logarithme de leur rapport [...] vaut 0,57 en médiane par station ». Quatre paragraphes plus loin : « L'erreur d'amplitude mesurée sur la période d'évaluation vaut 0,57 en médiane par station ». Corrigé : la seconde occurrence disparaît, le raisonnement renvoie au chiffre déjà donné.

### 2. Figures : lisibles seules

Un titre est un énoncé, jamais une question : la figure répond, elle n'interroge pas. La légende dit ce qui est tracé, grandeur, unité, période, population, lecture des couleurs ; la conclusion est dans le texte. Aucun gras ni italique dans une légende. Une figure porte un message ; si sa légende a besoin de deux phrases de mise en garde, la figure est fausse ou le message est double. Test : un hydrologue qui ne voit que la figure et sa légende peut-il dire en une phrase ce qu'elle montre ?

Écrit : « ## L'enveloppe tient-elle là où rien ne la calibre ? ». Corrigé : « ## Couverture de l'enveloppe sur les stations transférées », et la légende donne la couverture obtenue contre la couverture nominale.

Écrit, en légende : « La densité varie d'un facteur cent entre régions ; **cette contrainte est prête mais de poids nul dans les modèles présentés**. » Corrigé : le gras disparaît ; la phrase sur le poids nul va dans le texte, la légende ne garde que ce qui est tracé.

### 3. Code des documents : lisible comme la prose

Un nom de fonction dit ce qu'elle produit : `figure_png`, `carte_html`, jamais `montrer`, `carte`, `faire`. Un commentaire dit ce que fait la ligne quand ce n'est pas évident, en une ligne. L'histoire du bogue, l'ancienne version et la raison du choix vont dans le journal d'expériences, pas dans le document. Aucune majuscule d'insistance. Aucun code commenté : git garde l'historique. Un bloc de code dans un document ne fait que ce qui produit la figure ou le tableau ; les calculs lourds vont dans un module.

Écrit : onze lignes de commentaire commençant par « BANC DE LA TETE, sur des debits REELS deja simules [...] LE MODELE PROBABILISTE EST UN GENERATEUR D'ENSEMBLE ». Corrigé : « # Loi lognormale ajustée sur les sept quantiles prédits : la fonction quantile interpolée serait anguleuse. »

Écrit : trente-trois lignes `#"param_theta_fc_2",  # capacite au champ, couche 2`. Corrigé : supprimées.

### 4. Vocabulaire : externe, physique, dimensionnel

Le vocabulaire interne est interdit et se traduit. Tenu de côté devient période d'évaluation 2022-2024, jamais utilisée pour l'entraînement. Champion devient modèle retenu, recette devient configuration. Plateforme, pour une région, devient version calibrée d'Hydrotel. Ancrage devient paramètre repris du calage d'Hydrotel. Kge_med devient KGE médian par station. Un numéro d'hypothèse devient le fait qu'il désigne. Les mots pression, charge, tension, capacité, conductivité, flux, intensité, énergie ne s'emploient que dans leur sens physique. Toute définition « X est Y rapporté à Z » doit donner les bonnes unités. Un volume sur un débit est un temps, pas une fraction. Si les dimensions ne tombent pas juste, la phrase est fausse. Chaque nombre porte sa grandeur, son unité en clair, sa période ou sa population, à la précision qui a un sens physique.

Un terme venu d'une autre communauté est attribué à sa source dans la phrase même, et l'homonyme de la communauté du lecteur est désamorcé. Interpolation optimale est le schéma d'assimilation de Gandin en météorologie, et une redistribution des erreurs sur les débits en hydrologie opérationnelle québécoise.

Écrit : « La pression est le volume prélevé rapporté au débit du tronçon. » Corrigé : « La fraction du débit prélevée est le débit prélevé sur un tronçon, en mètres cubes par seconde, rapporté au débit naturalisé du même tronçon ; elle est sans dimension. »

Écrit : « Le seuil pluie-neige hérité du calage, moins 2,2168 degrés ». Corrigé : « Le partage entre pluie et neige était fixé à -2,2 °C. »

## Typographie

Pas de tirets cadratins ni demi-cadratins, pas de gras ni d'italique dans la prose, pas de guillemets décoratifs, pas de retour à la ligne forcé dans un paragraphe. Jamais le mot lame : écoulement.

## Ce que le document n'est pas

Pas un journal : aucune narration du travail, aucune histoire de bogue, aucune chronologie de diagnostic, ni dans la prose ni dans les commentaires. L'état, pas la fabrication. Les limites s'énoncent au même niveau que les résultats, jamais en excuse.

## Structure par défaut d'un rapport de modélisation

Ce que le lecteur peut conclure, avec les conditions de validité. Comment c'est mesuré : données, méthode, périodes, ce qui est indépendant de quoi. Ce que ça ne dit pas. Annexes techniques.

## La passe de livraison, obligatoire

1. Lancer `python .claude/skills/redaction/verifier.py <fichiers>`. Le script compte et localise, dans la prose : phrases de plus de 30 mots, sections de plus de 600 mots, segments répétés, remplissage, emphase, vocabulaire interne, titres en question, gras, italique, tirets. Dans le code : noms de fonctions vagues, blocs de commentaires, code commenté, majuscules d'insistance.
2. Traiter chaque constat : corriger, ou l'assumer explicitement avec sa raison. Un constat qu'on ne sait pas justifier est une correction.
3. Relire le document entier, une phrase à la fois, avec le test du lecteur externe. Un hydrologue qui n'a lu ni le registre ni les conversations comprend-il cette phrase seule ? Puis le test dimensionnel sur chaque définition, et le test de la figure seule sur chaque figure.
4. Relancer le script. Livrer avec le nombre de constats restants et leur justification, jamais en silence.
