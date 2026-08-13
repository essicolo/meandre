#!/usr/bin/env bash
# DEPART SUR LE CHAMP HYDROTEL puis optimisation libre (proposition d'Essi).
# Critere de succes fixe AVANT la mesure : depasser 0.7748, ce que la physique ancree
# obtient deja sans apprendre. En dessous, l'apprentissage reste une perte de valeur.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}
lancer() {
  local tag="$1"; shift
  attendre
  echo "[init] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_WET=0 ETL_FORCE=1 ETL_REGION=outv \
      ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KREC=5e-5 ETL_KGW=0.0645 ETL_DEMAND_SCALE=0.963 \
      ETL_INIT_HYDROTEL=1 ETL_EPOCHS=30 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${tag}.txt" 2>&1
  echo "[init] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${tag}.txt" | tr '\n' ' ')"
}
lancer "outv-initH"    ETL_TAG="-initH"                    # depart Hydrotel, 30 epoques
lancer "outv-initH5"   ETL_TAG="-initH5" ETL_EPOCHS=5      # affinage court : l'optimiseur s'eloigne-t-il tout de suite ?
lancer "outv-initH0"   ETL_TAG="-initH0" ETL_EPOCHS=1      # temoin quasi sans apprentissage
echo "[init] $(date +%H:%M) TERMINE"
