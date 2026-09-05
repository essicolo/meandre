# Tests à lancer sur la grappe après l'audit du 5 septembre 2026

Principe : chaque test est une paire ou un trio de bras qui ne diffèrent que par un réglage, sur les mêmes régions, la même recette du socle et le même forçage CaSR corrigé ; le verdict est la différence, jamais un bras seul. Chaque run imprime désormais, à côté du KGE tenu de côté 2022-2024, un verdict de forme (platitude, suites plates, pointes) : un bras dont la forme est refusée est écarté quel que soit son KGE. Tous les tests passent par le même script, `.runs/quebec/alliance/flotte.sbatch`, avec ses trois surcharges d'environnement : `FLOTTE_AUX` remplace le bloc de contraintes auxiliaires, `FLOTTE_TAG` s'ajoute au nom du run, et toute variable `ETL_*` exportée (par exemple `ETL_LR`) est lue par le pilote. Indices 0 à 14 : bras A (un pas par bloc, historique amorcé) ; 15 à 29 : bras B (un pas par époque). Le paquet de code doit être celui du 5 septembre au soir ou plus récent : il porte la lecture correcte des lacs (R81), le gradient du KGE remis à l'échelle (R82), le minimum de trente qui compte l'historique et l'absence de pas sur le prior seul (R80), et le verdict de forme.

Préalable, une seule archive, une seule commande. Depuis le poste :

```
scp "D:/meandre-data/quebec/code-meandre.tgz" atlas01@narval.alliancecan.ca:meandre/code-meandre.tgz
```

Sur la grappe :

```
cd ~/meandre && tar -xzf code-meandre.tgz
```

## Test 1, en cours : GRACE est-elle la cause de l'effondrement gaspésien ?

La commande donnée le 5 septembre en matinée portait des indices faux (6 et 21, soit ABIT, région sans GRACE) : cette paire est sans valeur et ses journaux `abit-A-sansgrace.log` et `abit-B-sansgrace.log` sont à ignorer. La bonne paire est la Gaspésie, indices 1 et 16, bras A et B, MOD16 gardée en tendance, GRACE éteinte, tag `-sansgrace` :

```
sbatch --account=def-atlas01 --array=1,16 --export=ALL,FLOTTE_AUX="ETL_WET=0.4 ETL_WTWS=0 ETL_WTWSCLIM=0",FLOTTE_TAG=-sansgrace .runs/quebec/alliance/flotte.sbatch
``` Lecture : si les deux bras cessent de s'effondrer (validation qui ne chute plus de 0,78 vers 0,15 en huit époques) et que le tenu de côté ne baisse pas, GRACE à ses poids actuels est disqualifiée comme contrainte d'entraînement ; si le bras A s'effondre encore, la boucle corrigée est en cause indépendamment de GRACE, ce que les journaux du 4 septembre suggèrent déjà (six effondrements sur neuf hors GRACE).

Journaux attendus : `~/scratch/meandre/flotte/gasp-A-sansgrace.log` et `gasp-B-sansgrace.log`.

## Test 2 : le candidat de livraison, ancienne boucle avec les correctifs

L'ancienne boucle est celle qui a le mieux tenu le 4 septembre. On la relance telle quelle sur les quatorze régions avec les correctifs du 5 septembre, pour mesurer ce que les lacs corrigés et le KGE remis à l'échelle changent, région par région, contre le bras B de la veille. GRACE reste active là où elle l'est dans la recette : c'est le test 4 qui en décidera.

```
sbatch --account=def-atlas01 --array=15-29%6 --export=ALL,FLOTTE_TAG=-v2 .runs/quebec/alliance/flotte.sbatch
```

Lecture : tenu de côté et verdict de forme de `<region>-B-v2` contre `<region>-B` du 4 septembre. Un gain de forme (platitude ramenée sous le double de l'observé, suites plates sous trente jours) avec un KGE stable ou meilleur valide les correctifs ; une platitude inchangée dans les régions sans lac désigne la colonne, pas le routage.

## Test 3 : la boucle corrigée est-elle viable à un taux adapté ?

Le bras A du 4 septembre faisait 146 pas d'Adam par époque à un taux réglé pour un seul, dix fois plus sur la couche de sortie et cinquante fois plus sur les lacs ; huit régions sur quatorze se sont effondrées. On le rejoue à un taux cinq fois plus bas, seul réglage changé.

```
sbatch --account=def-atlas01 --array=0-14%6 --export=ALL,ETL_LR=1e-4,FLOTTE_TAG=-v2lr1e-4 .runs/quebec/alliance/flotte.sbatch
```

Lecture : `<region>-A-v2lr1e-4` contre `<region>-B-v2` du test 2. Si les effondrements disparaissent et que la forme s'améliore, la boucle corrigée a un taux et peut être retenue ; sinon elle est abandonnée et le bras B reste la boucle de référence.

## Test 4 : le poids de GRACE, sur les cinq régions où elle est active

À lancer après lecture du test 1. Trois bras sur OUTV, GASP, SAGU, MONT, SLNO (indices 15 à 19 pour le bras B ; l'ordre des régions dans le script est outv, gasp, sagu, mont, slno, slso, abit, cnda, cndb, cndc, cndd, cnde, labi, outm, vaud, indices 0 à 14 pour A et 15 à 29 pour B), ancienne boucle et correctifs : GRACE aux poids actuels (le test 2 le fournit), GRACE à un dixième, GRACE éteinte.

```
sbatch --account=def-atlas01 --array=15-19%5 --export=ALL,FLOTTE_AUX="ETL_WET=0.4 ETL_WTWS=0.02 ETL_WTWSCLIM=0.005",FLOTTE_TAG=-v2grace10 .runs/quebec/alliance/flotte.sbatch
sbatch --account=def-atlas01 --array=15-19%5 --export=ALL,FLOTTE_AUX="ETL_WET=0.4 ETL_WTWS=0 ETL_WTWSCLIM=0",FLOTTE_TAG=-v2sansgrace .runs/quebec/alliance/flotte.sbatch
```

Lecture : tenu de côté, forme, et écart de stockage à GRACE (la composante `tws` du journal, corrigée du facteur 146). Le poids retenu est le plus grand qui ne dégrade ni le KGE ni la forme par rapport au bras sans GRACE : c'est la définition opérationnelle d'une contrainte d'identifiabilité qui contraint sans détruire.

## Test 5, conditionnel : le drainage non linéaire de la troisième couche

Si le banc de colonne du 5 septembre confirme que le plateau d'été naît du drainage linéaire de la troisième couche (test A du banc `banc_plateaux_colonne.py`), le correctif opt-in `ETL_L3_EXP` (R34, R37) devient candidat au socle. Une paire sur SLSO et SAGU, les deux régions aux plateaux d'été les plus longs, ancienne boucle et correctifs :

```
sbatch --account=def-atlas01 --array=17,20%2 --export=ALL,ETL_L3_EXP=1,FLOTTE_TAG=-v2l3exp .runs/quebec/alliance/flotte.sbatch
```

Lecture : part de jours plats d'été et réponse aux pluies de `<region>-B-v2l3exp` contre `<region>-B-v2`.

## Ce qu'on lit, et dans quel ordre

Une commande rend tout ce qui compte pour tous les runs d'un tag :

```
cd ~/scratch/meandre/flotte && for f in *-v2*.log; do echo "== ${f%.log}"; grep -hE "HELD-OUT|FORME|VERDICT DE FORME" "$f" | cut -c1-200; grep -oE "\] Epoch +[0-9]+ .*val_kge=[-0-9.]+" "$f" | grep -oE "val_kge=[-0-9.]+" | cut -c9- | tr '\n' ' '; echo; done
```

Ordre de décision : la forme d'abord (un bras refusé sort), le tenu de côté ensuite (différence par région, médiane et pire décile), la trajectoire de validation enfin (effondrement, gel). Les dumps de débit `q-<region>-<bras><tag>.npz` permettent l'audit externe complet avec le prompt de l'auditeur de crédibilité.

Coût : tests 2 et 3, quinze tâches chacun, deux à six heures par tâche, six en parallèle, une demi-journée chacun ; test 4, dix tâches, deux heures ; test 5, deux tâches.
