# Le forçage CaSR : correction de volume et température

Note interne, 2026-09-15. Trois mesures faites sans simulation, à la suite d'une remarque d'un collègue selon laquelle CaSR n'aurait pas véritablement de défaut de volume d'eau mais plutôt un défaut de calendrier de fonte, peut-être porté par la température. Ces analyses ne figurent pas dans la présentation, sur décision d'Essi : elle y annonce seulement qu'une correction de volume est employée aujourd'hui et qu'elle est remise en question.

Les caches sont produits par `.runs/quebec/cache_volume_casr.py`, `.runs/quebec/biais_temperature_casr.py` et `.runs/quebec/cible_budyko_anthropique.py`, et déposés dans `.reports/quebec/caches/`.

## La correction de volume est petite là où elle est mesurable

La correction multiplie tout le canal de précipitation par une constante par région, celle qui ferme le bilan de Budyko-Fu : précipitation corrigée égale écoulement observé aux stations plus évapotranspiration de Fu calculée sur l'évapotranspiration potentielle d'Oudin. Le facteur appliqué, mesuré en comparant la moyenne du canal avant et après recalage sur 2000-2024, vaut 0,969 en médiane provinciale.

| région | jauges | CaSR brut (mm/an) | cible (mm/an) | facteur | écart |
|---|---:|---:|---:|---:|---:|
| Labrador | 1 | 1147 | 983 | 0.857 | -14.3 % |
| Côte-Nord A | 1 | 1147 | 1010 | 0.881 | -11.9 % |
| Côte-Nord C | 2 | 1153 | 1050 | 0.911 | -8.9 % |
| Côte-Nord B | 2 | 1342 | 1249 | 0.930 | -7.0 % |
| Côte-Nord D | 2 | 1159 | 1079 | 0.931 | -6.9 % |
| Côte-Nord E | 2 | 1265 | 1179 | 0.932 | -6.8 % |
| Saguenay | 24 | 1128 | 1093 | 0.969 | -3.1 % |
| Abitibi | 4 | 982 | 963 | 0.980 | -2.0 % |
| Gaspésie | 18 | 1101 | 1091 | 0.991 | -0.9 % |
| Saint-Laurent nord-ouest | 32 | 1122 | 1134 | 1.010 | +1.0 % |
| Outaouais moyen | 4 | 949 | 966 | 1.018 | +1.8 % |
| Outaouais aval | 16 | 1065 | 1112 | 1.044 | +4.4 % |
| Montérégie | 25 | 982 | 1072 | 1.092 | +9.2 % |

Cinq régions portent seize stations hydrométriques ou plus. Quatre d'entre elles tiennent à moins de cinq pour cent de l'unité, la Gaspésie à moins d'un pour cent. Sur ces territoires, la précipitation de CaSR ferme déjà le bilan d'eau et la correction ne fait presque rien.

Toutes les réductions dépassant six pour cent sont sur la Côte-Nord et au Labrador, où la cible est construite sur une ou deux stations pour des bassins de plusieurs milliers de kilomètres carrés. Au Labrador, la réduction de 14,3 pour cent repose sur une seule station. Ce que la correction attribue à une erreur du produit météorologique peut donc aussi bien être l'incertitude de sa propre référence, et la méthode ne permet pas de trancher.

La Montérégie reste inexpliquée : vingt-cinq stations et une augmentation de 9,2 pour cent.

## L'hypothèse d'une cible gonflée par les rejets est écartée

Une station située en aval de rejets mesure de l'eau qui n'est pas tombée en pluie sur son bassin, ce qui élèverait la cible et ferait ajouter de la précipitation pour expliquer de l'eau d'origine anthropique. La Montérégie reçoit un apport net de 10,09 mètres cubes par seconde pour un débit médian de station de 4,9, ce qui rendait l'hypothèse plausible.

La mesure l'écarte. En cumulant le bilan anthropique net de tous les tronçons du bassin de chaque station et en le retirant du débit observé, l'écoulement spécifique médian se déplace de 0,1 pour cent en Montérégie, de 3,0 pour cent au plus ailleurs, et de moins de 1,2 pour cent dans treize régions sur quatorze. La médiane sur les stations est insensible parce que les gros apports portent sur quelques tronçons seulement, tandis que la médiane est fixée par des bassins de quelques centaines de kilomètres carrés.

## La température ne porte pas le calendrier de la fonte

Comparaison du forçage à la station météorologique d'Environnement et Changement climatique Canada la plus proche, à moins de quinze kilomètres d'un tronçon, sur 170 stations et les années 2016-2019, en degrés Celsius.

| mois | minimum | maximum | moyenne | degrés-jours simulés | observés |
|---|---:|---:|---:|---:|---:|
| janvier | +1.27 | -0.44 | +0.42 | 0.13 | 0.11 |
| février | +1.39 | -0.43 | +0.48 | 0.26 | 0.24 |
| mars | +1.23 | -0.41 | +0.41 | 0.66 | 0.62 |
| avril | +1.03 | -0.25 | +0.39 | 3.90 | 3.64 |
| mai | +1.14 | -0.21 | +0.46 | 11.65 | 11.19 |
| juin | +1.03 | -0.43 | +0.30 | 16.32 | 16.02 |
| juillet | +1.08 | -0.63 | +0.23 | 20.04 | 19.82 |
| août | +1.11 | -0.80 | +0.16 | 18.79 | 18.63 |
| septembre | +1.25 | -0.64 | +0.30 | 14.70 | 14.39 |
| octobre | +1.00 | -0.53 | +0.24 | 7.88 | 7.65 |
| novembre | +0.97 | -0.33 | +0.32 | 1.75 | 1.60 |
| décembre | +1.13 | -0.48 | +0.32 | 0.23 | 0.21 |

La date à laquelle le cumul de degrés-jours depuis le premier janvier atteint cinquante, qui est l'indicateur direct du déclenchement de la fonte, arrive un jour trop tôt en médiane et 1,6 jour trop tôt en moyenne, avec un intervalle interquartile de trois jours d'avance à un jour de retard, sur 637 couples station-année. L'avance est réelle mais d'un ordre de grandeur inférieur à ce qu'il faudrait pour expliquer une erreur de crue printanière.

La mesure révèle en revanche un défaut que la moyenne masque. Le minimum journalier est trop élevé de 1,14 degré en moyenne annuelle et le maximum trop bas de 0,48, tous les mois de l'année sans exception. Le forçage écrase donc l'amplitude diurne d'environ un degré et demi, et sa moyenne n'est juste que parce que les deux erreurs se compensent. Cela ne déplace pas un calendrier de fonte piloté par la température moyenne, mais cela supprime des nuits de gel et adoucit le partage entre pluie et neige aux extrémités de la nuit.

## Ce que ces mesures ouvrent

Le retrait de la correction de volume sur les régions bien jaugées se teste à coût faible et ne changerait rien de mesurable si le facteur y vaut déjà un. Sur la Côte-Nord, la question n'est pas celle du produit météorologique mais celle de la densité du réseau hydrométrique, et aucune correction ne peut la résoudre.

L'écrasement de l'amplitude diurne est un défaut du forçage indépendant du volume. Il touche le regel nocturne et le partage pluie-neige, deux mécanismes où la colonne emploie le minimum et le maximum journaliers plutôt que la moyenne.
