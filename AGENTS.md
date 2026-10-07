# AGENTS.md : état du chantier au 6 octobre 2026, 17 h 15

Ce fichier passe le relais à un autre agent. Il dit où on en est, ce qui a été fait depuis le 5 octobre, quels fichiers ont changé, quelle est la prochaine étape exacte, et où lire l'histoire des hypothèses. Les règles de travail sont dans `CLAUDE.md` ; lire d'abord ses deux premières sections et la section « Le banc ne vaut pas le territoire ». Tout ce qui est écrit pour un humain est en français ; identifiants, clés TOML et noms de fichiers sont en anglais.

## 1. La tâche

Produire pour la prochaine présentation d'Essi deux cartes feuillage sur toute la province :

1. les prélèvements et rejets ponctuels (un point par site, débit net moyen) ;
2. la reconstruction des débits par meandre : débit modélisé avec prélèvements et rejets, débit naturalisé (même modèle, prélèvements à zéro), tronçons colorés par l'influence relative des prélèvements et rejets sur l'étiage, hydrogrammes au clic, les deux courbes quand les deux couches sont allumées.

Les cartes existent et fonctionnent sur deux territoires (Outaouais, Montérégie). Il manque les douze autres territoires, dont les modèles s'entraînent en ce moment sur Ubuntu. Une présentation est prévue à court terme ; Essi espérait la province pour le 7 octobre, elle sera complète le 8 au soir.

## 2. Où tourne quoi

Trois machines.

- Le poste Windows (ce dépôt, `C:\Users\parse01\documents-locaux\GitHub\meandre`, données sous `D:\meandre-data`). Les cartes se construisent ici. Le dépôt feuillage est à côté : `C:\Users\parse01\documents-locaux\GitHub\feuillage`, branche locale `cartes-meandre`, jamais poussée.
- Ubuntu (`ssh essi@192.168.40.165`), carte graphique de 8 Go, code dans `~/meandre-claude` (copie par `scp` ou `tar`, pas un clone git), Python `~/Documents/git/meandre/.venv/bin/python`, données sous `~/meandre-data`. Scripts lancés par `nohup setsid bash /tmp/<script>.sh`, après `sed -i 's/\r$//'`. Ne jamais faire `pkill -f` avec un motif qui apparaît dans la ligne de commande ssh : cela coupe la session (mémoire `project_ubuntu_pkill_autodestruction`).
- Narval (Alliance), compte `def-atlas01_gpu`, code cloné dans `~/meandre` (branche `requalification-masse`, `git pull` pour mettre à jour), données dans `/home/atlas01/projects/def-atlas01/atlas01/meandre/{donnees,plateformes,checkpoints-reference}`. Seule Essi peut s'y connecter (double authentification). Un seul lot y attend : les épreuves `4801184_[0-11]`, départ prévu le 7 octobre vers 15 h, fin dans la nuit. Le lot provincial y a été annulé : Ubuntu le fait.

## 3. Ce qui tourne sur Ubuntu en ce moment (état du 7 octobre, 14 h)

Journaux dans `~/meandre-data/carte-essai/`. Les chaînes 1, 4 et 5 sont finies ; treize territoires sur quatorze ont leurs exports et sont sur la carte.

- `/tmp/chaine_nuit6.sh` (journal `chaine_nuit6.log`) : Saint-Laurent sud-ouest (`slso`, départ 12 h 08, sans MOD16 ni GRACE, sa base ne les porte pas : `ETL_WET=0 ETL_WTWS=0 ETL_WTWSCLIM=0`), puis deux réentraînements de l'Outaouais, `-corr1006-sublim` (`ETL_SUBLIM=1`) et `-corr1006-gelsublim` (`ETL_SUBLIM=1 ETL_NAPPE_GEL=0.3`), chacun suivi d'une passe avant sur la dernière époque disponible (`passe_derniere.sh`, l'arrêt précoce peut s'arrêter avant la seizième).
- `/tmp/sonde_mont_depart.sh` (journal `sonde_mont_depart.log`) : sonde par terme au pilote sur la Montérégie, au point de départ (init littérature) puis à l'époque 4, pour dire quel terme pousse la conductivité de surface K_sat_1 vers le haut (elle finit à 0,40 m/j en médiane et 0,90 sur les nœuds agricoles, départ 0,08 ; registre R326). Les sondes au point d'ARRIVÉE (banc, 030905, `sonde_mont.log`) ne la désignent dans aucune cible : à convergence le gradient est à l'équilibre, seul le départ dit la poussée.

Chaque territoire : seize époques avec la recette `.runs/quebec/config/socle-2026-10-04.toml`, forçage `JOINT_FX_SUFFIX=-casr-brut`, un point de reprise par époque (`MEANDRE_SAUVER_EPOQUES=1`), puis une passe avant sur la dernière époque avec `ETL_DUMP_REACH` et `ETL_DUMP_NATUREL=1`, qui écrit `<reg>-q-journalier.npz` et `<reg>-sans-journalier.npz` dans `~/meandre-data/carte-essai/`. Le jugement se fait toujours sur la DERNIÈRE époque, jamais sur celle retenue par la validation (registre R282, R294).

Résultats sur 2022-2024, dernière époque, code corrigé, recette du socle (`indicateurs_stations.py`) :

| territoire | stations | KGE médian | volume sim/obs | minimum annuel sur 7 jours sim/obs |
| --- | --- | --- | --- | --- |
| Outaouais (outv) | 16 | 0,80 | 1,16 | 0,85 |
| Montérégie (mont) | 23 | 0,55 | 1,04 | 1,46 ; juillet et août à 0,45 et 0,52 |
| Montérégie, drainage et tête du temps de transfert (`-dt`) | 23 | 0,555 | | 1,63 ; le remède du banc ne se transfère pas (R323) |
| Saint-Laurent nord-ouest (slno) | | 0,72 | 1,14 | 0,59 |
| Saguenay (sagu) | | 0,75 | | |
| Gaspésie (gasp) | | 0,77 | 1,00 | 0,81 ; neige deux fois NEISIM |
| Abitibi (abit) | 3 | 0,79 | | |
| Outaouais moyen (outm) | | 0,56 | 1,30 | |
| Côte-Nord B, C, D, A, E | | 0,75 ; 0,77 ; 0,77 ; 0,70 ; 0,86 | cndc 1,30 | cndd une station régulée, cnde deux stations |
| Labrador (labi) | 1 | 0,63 | | |

Diagnostics de la Montérégie (registre R325 à R328) : l'été à moitié vide vient des ÉVÉNEMENTS (écoulement de base simulé 1,00 fois l'observé, pointes 0,40) ; K_sat_1 est appris dix fois trop haut et plus encore sur les nœuds agricoles ; le ramener à un dixième en passe avant rend les pointes mais fait tomber la corrélation de 0,73 à 0,54 ; la texture PHYSITEL ne donne aucun contraste agricole contre forêt ; CaSR brut manque les gros jours de pluie d'été sur 70 % des jauges de la Montérégie (stations canadiennes seules : 0,83 à 0,89 du mesuré). Forçage et modèle y comptent chacun pour environ la moitié. Le prétraitement du forçage est hors de méandre et appartient à Essi.

Prélèvements et rejets : le rattachement io-eau des sites aux tronçons est faux pour une part importante (107 sites à plus de 5 km de leur tronçon, 219 rattachés à un cours d'eau dont le nom diffère de celui déclaré ; listes dans `reports/diagnostics/`). Essi reprend le rattachement avec un autre agent ; TOUT devra être remodélisé ensuite. Jusque-là, les travaux portent sur le code et la physique, pas sur de nouveaux entraînements provinciaux.

Suite de tests : 541 tests passent (7 octobre). Huit échecs qui n'apparaissaient qu'en suite complète venaient de trois modules posant la double précision sans la rendre ; `tests/conftest.py` rend le dtype par défaut après chaque test.

## 4. La prochaine étape exacte

1. Quand un territoire écrit `=== <reg> fini` dans son journal, rapatrier ses deux fichiers sur le poste :
   `scp essi@192.168.40.165:meandre-data/carte-essai/<reg>-q-journalier.npz essi@192.168.40.165:meandre-data/carte-essai/<reg>-sans-journalier.npz D:/meandre-data/carte-essai/`
   (pour la Montérégie corrigée, les fichiers s'appellent `mont-corr1006-dt-*` ; les renommer en `mont-*` à la place des actuels si son KGE et son étiage sont meilleurs, en gardant les anciens dans `D:/meandre-data/carte-essai/ancien/`).
2. Reconstruire la carte : `PYTHONIOENCODING=utf-8 MEANDRE_CARTE_DUMPS=D:/meandre-data/carte-essai uv run --offline python .runs/quebec/carte_reconstruction.py`. Le script prend tous les territoires dont les deux fichiers existent et écrit dans le dépôt feuillage : `data/meandre/reconstruction-troncons.geojson`, `data/meandre/prelevements-rejets.geojson`, `data/meandre/debits.zarr` (ignoré par git), `config-meandre-reconstruction.json`, `config-meandre-prelevements.json`.
3. Vérifier à l'écran : un serveur `uv run --offline python -m http.server 8765 --directory C:/Users/parse01/documents-locaux/GitHub/feuillage` tourne en tâche de fond depuis cette session (le relancer s'il est mort), carte à `http://localhost:8765/index.html?config=config-meandre-reconstruction.json`. Scripts de capture Playwright dans le dossier de travail de la session : `verif_deux.py`, `verif_clic2.py` (`uv run --offline --no-project --with playwright python ...`, Edge via `channel="msedge"`).
4. Consigner dans feuillage (`git add` des quatre fichiers ci-dessus, jamais le zarr, jamais de `push`) et donner à Essi le tableau KGE / volume / étiage du territoire, calculé par `etiage_stations.py <q-<reg>-corr1006.npz>` (dossier de travail de la session) ou par la ligne `HELD-OUT` du journal.
5. Mercredi 8 au soir, province complète : refaire la carte, puis écrire une note d'état dans `reports/`.

Indicateurs d'étiage à juger d'abord (objectif d'Essi : les étiages, pour la naturalisation et les prélèvements) : minimum annuel du débit moyen sur 7 jours, jours sous le Q90 observé, volume d'août-septembre ; le KGE en dernier.

## 5. Ce qui a été fait depuis le 5 octobre, et les fichiers touchés

Deux défauts majeurs trouvés et corrigés dans l'entraîneur, tous deux poussés sur `origin/requalification-masse` :

- Tous les modèles s'entraînaient sans prélèvements ni rejets. La simulation lit les prélèvements à l'indice de pas compté depuis le début de l'appel, et l'entraîneur passait la série complète avec le forçage d'un bloc : chaque bloc lisait janvier 2000, nul. Correctif `WithdrawalData.slice` dans `meandre/routing/withdrawals.py`, dix appels dans `meandre/training/trainer.py`, test `tests/test_prelevements_tranche.py`. Registre R308.
- La référence du terme MOD16 centré n'était jamais posée au pilote régional (il évalue avant d'entraîner) ; le repli prenait la moyenne du premier bloc d'hiver (0,02 mm/j au lieu de 1,61), et le terme poussait K_c vers le bas quatre-vingts fois plus fort que les termes de débit. Correctif `_et_reference_exacte` appelée au début de l'époque dans `trainer.py` ; registre R309 à R311. Un mode `forme` (centré et réduit) a été ajouté à `meandre/training/loss.py` et `trainer.py` mais n'est pas retenu (R311).

Outils ajoutés :

- `meandre/training/trainer.py` : sonde par terme `MEANDRE_SONDE_TERMES=K_c,...` (gradient de chaque terme pondéré par rapport à un facteur par tronçon, cumulé sur une époque à `ETL_LR=0`), `MEANDRE_SONDE_TERMES_SEULS`, `MEANDRE_SONDE_TERMES_SORTIE`.
- `.runs/quebec/trajectoire_champ.py` : une sortie du champ (K_c, etc.) moyennée sur le bassin amont de stations, pour une liste de points de reprise, sans simuler.
- `.runs/quebec/depots_troncons.py` : dépôts quaternaires du SIGEOM par tronçon, quinze territoires, résultat `D:/meandre-data/derives/auxiliaires/depots-sigeom-troncons.parquet` (copié sur Ubuntu) ; branché au champ par `ETL_ATTRIBUTS_DEPOTS=1` dans `.runs/quebec/joint_data.py` (`_ajoute_depots`, six colonnes). Le module d'évapotranspiration appris du pilote ne lit que ses attributs d'origine (`_COLS_ETB` dans `.runs/quebec/etl_run.py`).
- `.runs/quebec/etl_run.py` : `ETL_SOIL_TOML` (profil de sol dans un fichier à part), tête du temps de transfert créée AVANT le départ à chaud (R306), export du débit naturalisé journalier avec `ETL_DUMP_NATUREL=1`.
- `.runs/quebec/carte_reconstruction.py` : construit les deux cartes (sites io-eau depuis `../io-eau/data/derived/io-eau-meandre.parquet`, tronçons PHYSITEL en lignes, lacs en polygones, zarr des débits à trois séries : modélisé, naturalisé, écart).
- `.runs/quebec/alliance/province.sbatch` (lot provincial, non utilisé finalement), `epreuves.sbatch` (douze tâches : témoin, dépôts, dépôts et tête sur l'Outaouais ; témoin, dépôts, dépôts, drainage et tête sur la Montérégie ; deux graines), `manifeste_province.sh` (liste Globus).
- feuillage (`index.html`, branche `cartes-meandre`) : `color_by`, `size_by`, `width_by`, `fill_opacity` avec échelle sous la tuile de couche ; séries nommées par couche (`layer`) avec légende ; `labels` dans les sections de propriétés ; axe log optionnel (`options.ylog`, non employé) ; clic sur la ligne la plus proche à une confluence ; barre latérale qui défile ; fonds Esri gris clair et OpenStreetMap humanitaire, Positron et Dark Matter retirés ; une couche colorée par attribut garde son dégradé à l'édition.

Diagnostics établis, à lire dans le registre : R305 (Narval reproduit le poste ; les bases DuckDB d'Ubuntu ont été resynchronisées sur celles du poste), R307 (sur 040841, toutes les cibles poussent K_c vers le haut), R312 (K_c est la variable d'ajustement d'une mauvaise répartition entre écoulement rapide et écoulement de base), R313 (réentraînement de l'Outaouais : 0,755 vers 0,796, volume +21 % vers +16 %), R314 (indice d'écoulement de base : le modèle plafonne vers 0,60 là où l'observé atteint 0,70, et aucun attribut ne distingue ces bassins), R315 (dépôts quaternaires : signal partiel, carte muette au nord), R316 (excès d'hiver : base trop forte au nord en décembre-janvier, fonte trop précoce vers l'hypodermique au sud en février-mars).

Non consigné au registre, à faire : le verdict de la Montérégie au socle (0,549) et, quand elle sortira, celui de la Montérégie avec drainage et tête.

## 5 bis. Ajouts de l'après-midi du 6 octobre (16 h à 17 h)

Chaînes sur Ubuntu, dans l'ordre : `chaine_nuit.sh` (Montérégie faite, Saint-Laurent nord-ouest en cours, Saguenay), puis `chaine_nuit4.sh` (Montérégie avec drainage et tête `-corr1006-dt`, Gaspésie, Abitibi, Outaouais moyen, Côte-Nord B C D A E, Labrador, Saint-Laurent sud-ouest), puis `chaine_nuit5.sh` (Outaouais `-corr1006-sublim`, puis `-corr1006-gelsublim`). Journaux `chaine_nuit*.log` dans `~/meandre-data/carte-essai/`.

Mécanismes ajoutés, tous opt-in, testés en passe avant seulement :

- porte de gel sur la vidange de la nappe libre, `ETL_NAPPE_GEL=0.3` (pilote et banc), `nappe_gel_facteur` dans `meandre/vertical/hydrotel_column.py` ; règle l'hiver du nord mais déplace l'eau en mai (R318) ;
- porte de gel par processus de sol déclaré, clé `frost_factor` dans un `[[soil.process]]` (`meandre/vertical/soil_processes.py`, `hydrotel_clone/bv3c2.py`), test `tests/test_porte_gel_declaree.py` ; pas encore éprouvée ;
- sublimation de Kuzmin au banc, `ETL_SUBLIM=1` (existait au pilote depuis le 22 août, R36) : sous CaSR brut elle ramène le manteau du nord sur NEISIM et retire la moitié de l'excès de volume (R320).

Diagnostics : le manteau du modèle régional dépasse NEISIM de 16 à 36 % et fond en mai (R319, qui corrige la lecture de R316 au sud) ; CaSR brut est 1,3 à 1,44 fois les stations en hiver sur l'Outaouais et 0,69 à 0,86 fois les stations en été en Montérégie (R321, `.runs/quebec/pluie_casr_vs_stations.py` avec `JOINT_FX_SUFFIX=-casr-brut MEANDRE_PLUIE_MARGE=1.0 MEANDRE_PLUIE_ANNEES=2011,2024`). La correction du forçage par territoire et par saison est une décision d'Essi, hors de meandre.

Jugement des épreuves de Narval quand elles sortiront : `python .runs/quebec/indicateurs_stations.py <q-*.npz>` sur les fichiers de `~/scratch/meandre/epreuves/`, rapatriés sur le poste.

## 6. Où lire l'histoire

- `reports/registre_hypotheses.md` : l'état courant, une entrée par hypothèse avec son statut (établi, réfuté, caduc, ouvert). Les entrées du 5 et du 6 octobre vont de R305 à R316. À LIRE AVANT de citer un chiffre, à RÉVISER à chaque verdict. Ne jamais citer un numéro R à Essi : énoncer le fait.
- `reports/experiment_log.md` : le journal chronologique.
- `reports/etat_2026-10-04.md` : la synthèse de la fin de semaine du 2 au 4 octobre (banc, correctifs, routage).
- `reports/chantiers_a_venir.md` : les chantiers ouverts et leur ordre.
- `CLAUDE.md` : les règles, dont « Le banc ne vaut pas le territoire ».
- Mémoire de l'agent (`C:\Users\parse01\.claude\projects\C--Users-parse01-documents-locaux-GitHub-meandre\memory\MEMORY.md`) : préférences d'Essi et faits de projet non déductibles du code.

## 7. Ce qu'Essi a demandé et refusé

- Carte feuillage et présentation sont deux objets séparés : feuillage n'entre pas dans `presentation.qmd` ; Essi bascule de l'une à l'autre.
- Hydrogrammes en axe normal, pas logarithmique.
- Jamais de copie de l'application feuillage à côté des données : les données vont dans le dépôt feuillage selon sa convention (`data/meandre/`, configuration à la racine).
- Un lac est un polygone, un tronçon de rivière une ligne.
- Pas de longs tests là où un calcul sur des sorties existantes répond ; avant tout entraînement, une phrase qui dit ce qu'aucun test rapide ne peut dire.
- Décision en suspens : publier ou non le magasin `debits.zarr` sur HuggingFace (données publiques) ; en attendant il reste sur le poste.
