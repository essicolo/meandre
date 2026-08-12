#!/usr/bin/env bash
# DEUXIÈME FILE, enchaînée derrière la première (attend son marqueur TERMINE).
#
# Motif : le pli 0/4 montre que le champ appris est battu par les paramètres ANCRÉS
# même sur les jauges dont il a les observations (0.595 contre 0.739). Le problème
# n'est donc pas la régionalisation mais l'apprentissage. Ces trois runs testent
# l'ancrage PENDANT l'entraînement, ce qui n'a jamais été fait sur la base saine.
#
# La « LOI DES ANCRAGES » de CLAUDE.md (ancrer les processus scalaires, JAMAIS le champ
# de sol) vient de deux échecs de juillet, expliqués à l'époque par le fait que le NeRF
# devait compenser par le sol les divergences structurelles de la colonne. Ces
# divergences sont maintenant corrigées (occupation, milieux humides, ETR complète,
# masse, seuil pluie/neige). La loi est donc à REJUGER, pas à appliquer.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1

while ! grep -q "TERMINE" /d/meandre-data/quebec/log-file-nuit.txt 2>/dev/null; do sleep 300; done

attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}

lancer() {
  local tag="$1"; shift
  attendre
  echo "[file2] $(date +%H:%M) demarrage $tag"
  # Les affectations de l'APPELANT viennent en DERNIER : env applique de gauche à
  # droite, donc elles doivent pouvoir écraser les valeurs communes (sans quoi le run
  # prévu à 5 époques en ferait 30).
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_EPOCHS=30 ETL_WET=0 ETL_FORCE=1 \
      ETL_REGION=outv ETL_KSAT1=0.04 ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KREC=5e-5 \
      ETL_KGW=0.0645 ETL_DEMAND_SCALE=0.963 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${tag}.txt" 2>&1
  echo "[file2] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${tag}.txt" | tr '\n' ' ')"
}

# O5-a : sol ENTIÈREMENT ancré sur bv3c.csv, le NeRF n'apprend que le reste.
lancer "outv-solcalib" ETL_TAG="-solcalib" ETL_SOIL_CALIB=1

# O5-b : ancrage de la TEXTURE seulement (épaisseurs laissées au champ).
lancer "outv-soltexture" ETL_TAG="-soltexture" ETL_SOIL_CALIB=1 ETL_SOIL_CALIB_TEXTURE=1

# O5-c : affinage COURT depuis l'ancré (5 époques) — l'optimiseur s'éloigne-t-il tout
# de suite de l'optimum ancré, ou faut-il 30 époques pour le perdre ?
lancer "outv-solcalib5" ETL_TAG="-solcalib5" ETL_SOIL_CALIB=1 ETL_EPOCHS=5

echo "[file2] $(date +%H:%M) TERMINE"
