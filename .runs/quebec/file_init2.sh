#!/usr/bin/env bash
# Depart Hydrotel, DEUXIEME essai : le premier transferait la permeabilite saturee mais
# pas la COURBE de retention (b=2.65 contre 3.97), d'ou une conductivite effective 4x
# trop forte a humidite realiste et un score effondre a 0.40. Courbe posee cette fois.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}
lancer() {
  local tag="$1"; shift
  attendre
  echo "[init2] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_WET=0 ETL_FORCE=1 ETL_REGION=outv \
      ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KGW=0.0645 ETL_DEMAND_SCALE=0.963 \
      ETL_INIT_HYDROTEL=1 ETL_EPOCHS=30 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${tag}.txt" 2>&1
  echo "[init2] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${tag}.txt" | tr '\n' ' ')"
}
lancer "outv-initK1"  ETL_TAG="-initK1" ETL_EPOCHS=1     # temoin : le depart vaut-il deja 0.77 ?
lancer "outv-initK30" ETL_TAG="-initK30"                  # puis 30 epoques d'optimisation libre
echo "[init2] $(date +%H:%M) TERMINE"
