# Résultats, au fil de l'eau

EN COURS, NE PAS SUPPRIMER. Voir README.md pour la question et le plan.

Matériau commun : sorties journalières du modèle provincial gelé (`modele-provincial-2026-10.toml`, entraîné sur 2001-2018) aux 144 stations portant au moins 365 jours mesurés entre 2019 et 2024, soit 283 488 jours-station. Les six années sont hors entraînement. Aucune simulation n'a été lancée. Données préparées par `donnees.py` dans `D:/meandre-data/experiments/gru_residual`.

## Étape 1. Plafond de ce qu'une correction en aval peut apprendre (2026-10-22, `plafond.py`)

Cible : écart journalier entre débit observé et simulé, divisé par le débit moyen simulé de la station (connu sans station). La cible en logarithme du rapport, celle du GRU de l'été, a été essayée d'abord : prévisible (R² jusqu'à 0,31) mais sa correction dégrade le KGE et le volume, parce qu'elle pèse peu les crues et que le retour depuis le logarithme biaise la moyenne. Elle est écartée pour le chiffrage.

Validation B, la seule qui compte pour une carte : apprentissage 2019-2021 sur 80 % des stations, évaluation 2022-2024 aux 20 % retirés, cinq plis. KGE médian par station sans correction : 0,696. Étiage : médiane sur les stations de l'écart absolu du logarithme du rapport simulé sur observé des minimums estivaux de la moyenne sur sept jours ; 0,255 sans correction, soit un écart typique de 29 %.

| Correction | R² de l'écart | KGE médian | Stations améliorées | Écart de volume médian | Erreur d'étiage |
|---|---|---|---|---|---|
| aucune | 0 | 0,696 | | 0,086 | 0,255 |
| arbres, débit simulé et jour de l'année | 0,17 | 0,672 | 37 % | 0,092 | 0,248 |
| arbres, plus forçage et manteau de degrés-jours | 0,27 | 0,714 | 54 % | 0,094 | 0,271 |
| arbres, plus attributs et champ spatial | 0,31 | 0,712 | 55 % | 0,111 | 0,309 |
| régression ridge, mêmes prédicteurs | 0,11 à 0,12 | 0,670 à 0,677 | 35 à 38 % | 0,14 à 0,15 | 0,37 à 0,55 |
| témoin : cycle mensuel moyen commun à toutes les stations | 0 | 0,689 | 39 % | 0,108 | 0,483 |

Validation A, stations jaugées : on retire une année, on apprend sur les cinq autres, toutes stations vues. Meilleur cas, arbres avec forçage et neige : KGE médian 0,711 à 0,757, 65 % des stations améliorées, étiage 0,255 à 0,275. Témoin sans réseau, moyenne de l'écart par station et par mois apprise sur 2019-2021 et posée sur 2022-2024 : KGE médian 0,696 à 0,718 (en logarithme : 0,721, avec volume et étiage améliorés, 0,086 à 0,066 et 0,255 à 0,240).

Lecture. Le seuil d'arrêt posé d'avance (R² de 0,05) est franchi : un quart de l'écart est prévisible aux stations jamais vues. Mais ce quart rapporte peu : +0,018 de KGE médian aux stations retirées, moins que le simple recalage du retard d'un jour (étape 2), avec un étiage dégradé. Les attributs statiques et le champ spatial n'ajoutent rien qui se transfère. Aux stations jaugées, une moyenne mensuelle par station fait presque aussi bien que les arbres.

## Étape 2. Structure de l'écart (`plafond.py`, `evenements.py`, `decalage.py`)

Décomposition de la variance du logarithme du rapport observé sur simulé : biais constant par station 7 %, cycle mensuel moyen par station 15 %, reste 78 %. Le reste a une autocorrélation de 0,87 à un jour, 0,37 à sept jours, 0,08 à trente jours : ce sont des erreurs d'événement, qui durent quelques jours.

Écart moyen selon le régime : le modèle surestime les très bas débits (quintile inférieur, −0,14 en logarithme, environ 13 %) et sous-estime les hauts (+0,13 à +0,14). Selon le mois : sous-estimation en juillet (+0,12), novembre (+0,12), surestimation en mai (−0,08).

Où est l'erreur quadratique : 70 % sur les 10 % de jours aux plus forts débits, 53 % sur les jours de montée qui ne sont que 30 % des jours, 0,5 % sur les 30 % de jours aux plus faibles débits.

Décalage dans le temps : le débit simulé est en retard d'au moins un jour sur 79 stations des 144 (corrélation croisée maximale). Recaler chaque station de son meilleur décalage fait passer le KGE médian de 0,712 à 0,739, davantage que toute correction apprise transférable. La variation journalière moyenne relative vaut 0,16 au simulé contre 0,22 à l'observé : l'hydrogramme simulé est trop amorti.

Origine du retard, par le délai de réponse à la pluie de bassin (décalage qui maximise la corrélation entre la pluie du jour et la hausse du débit) :

| Lacs en série vers la station | Stations | Retard du simulé (corrélation croisée, jours) | Délai de réponse simulé moins observé (jours) |
|---|---|---|---|
| 0 | 59 | 0,59 | 0,07 |
| 1 | 27 | 0,63 | 0,04 |
| 2 | 27 | 0,44 | 0,07 |
| 3 à 4 | 16 | 1,38 | 0,75 |
| 5 et plus | 15 | 0,67 | 1,07 |

Deux mécanismes distincts. Avec trois lacs ou plus en série, la réponse simulée démarre trop tard, de 0,75 à 1 jour : c'est cohérent avec le routage `operator-lagged`, où chaque lac relâche selon son stockage de la veille, ce qui cumule un pas par lac traversé. Sans lac, la réponse démarre au bon jour mais la crue est étalée : forme de la réponse rapide, pas calendrier.

## Étape 3, version hors ligne (`gru_horsligne.py`)

GRU de 32 unités en aval du modèle gelé, aux stations, même validation B, arrêt précoce sur 2021, 2022-2024 jamais vu. Graine 0 :

| Prédicteurs, sortie | R² de l'écart | KGE médian | Stations améliorées | Écart de volume | Erreur d'étiage |
|---|---|---|---|---|---|
| aucun correcteur | 0 | 0,696 | | 0,086 | 0,255 |
| minimal, additive | 0,13 | 0,694 | 41 % | 0,091 | 0,265 |
| minimal, bornée à ±30 % | 0,10 | 0,663 | 30 % | 0,149 | 0,255 |
| forçage et neige, additive | 0,27 | 0,725 | 57 % | 0,096 | 0,397 |
| forçage et neige, bornée | 0,23 | 0,704 | 48 % | 0,104 | 0,282 |
| tout et statiques, additive | 0,22 | 0,694 | 55 % | 0,134 | 0,678 |
| tout et statiques, bornée | 0,20 | 0,707 | 49 % | 0,135 | 0,287 |

Trois graines (0, 1, 2), moyenne et étendue :

| Prédicteurs, sortie | R² de l'écart | KGE médian | Stations améliorées | Écart de volume | Erreur d'étiage |
|---|---|---|---|---|---|
| aucun correcteur | 0 | 0,696 | | 0,086 | 0,255 |
| minimal, additive | 0,12 | 0,691 (0,689 à 0,694) | 42 % | 0,090 | 0,288 |
| minimal, bornée | 0,12 | 0,676 (0,663 à 0,693) | 34 % | 0,142 | 0,284 |
| forçage et neige, additive | 0,26 | 0,729 (0,725 à 0,735) | 57 % | 0,092 | 0,444 (0,37 à 0,56) |
| forçage et neige, bornée | 0,23 | 0,713 (0,704 à 0,724) | 50 % | 0,103 | 0,280 |
| tout et statiques, additive | 0,21 | 0,691 (0,683 à 0,696) | 50 % | 0,132 | 0,99 (0,68 à 1,53) |
| tout et statiques, bornée | 0,21 | 0,713 (0,707 à 0,718) | 51 % | 0,123 | 0,270 |

Lecture. Le GRU fait comme les arbres, pas mieux : la mémoire récurrente n'ajoute rien à des cumuls de forçage sur 1 à 90 jours. Le meilleur correcteur transférable gagne +0,033 de KGE médian aux stations jamais vues, au prix d'un étiage dégradé de 29 % à 45 à 75 % d'erreur typique ; la version bornée garde l'étiage presque intact (0,280) mais ne gagne que +0,017, et dégrade le volume. Les attributs statiques et le champ spatial, qui devraient porter la spatialisation, dégradent la version additive : le correcteur apprend les stations d'apprentissage. Le GRU minimal de l'été (débit simulé et jour de l'année) ne gagne plus rien sur ce modèle (0,676 à 0,691 contre 0,696).

## Verdict provisoire (2026-10-22)

Un correcteur récurrent en aval du modèle physique actuel n'est pas justifié pour la carte. Gain transférable aux stations non jaugées : +0,017 à +0,033 de KGE médian, toujours au prix de l'étiage ou du volume, et inférieur ou égal au gain du seul recalage du retard (+0,027). Aux stations jaugées, une moyenne mensuelle de l'écart par station fait presque aussi bien qu'un réseau.

Ce que l'écart désigne en revanche, c'est de la physique : le retard cumulé par les lacs en série du routage `operator-lagged` (approximation documentée dans `meandre/routing/operator_routing.py` comme « à valider numériquement contre le balayage »), et un hydrogramme trop amorti aux bassins sans lac. Les deux se testent en passe avant, sans entraînement : comparer `routing_mode = "operator"` et `"operator-lagged"` à poids gelés sur un territoire aux lacs nombreux (Outaouais : 8 stations à 3 lacs en série ou plus, retard interne moyen 1,5 jour ; Abitibi, Outaouais supérieur).

Les étapes 3 à 5 du plan (correcteur dans le pilote, entraînement conjoint, stations retirées) ne sont pas lancées : rien dans ces mesures ne laisse attendre qu'elles changent la décision.
