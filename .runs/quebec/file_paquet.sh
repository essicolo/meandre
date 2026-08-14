#!/usr/bin/env bash
# PAQUET COHERENT : sol d'Hydrotel + ETP Linacre calee, c'est-a-dire exactement la
# configuration qui donne 0.7748 en inference. Un run a 1 epoque sert de CONTROLE avant
# d'engager les 30 : lecon du 14 aout, valider l'effet de bout en bout AVANT le long run.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}
lancer() {
  local tag="$1"; shift
  attendre
  echo "[paquet] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_WET=0 ETL_FORCE=1 ETL_REGION=outv \
      ETL_WSNOW=0.3 ETL_AQUIFER=0 ETL_NO_LATENT=1 ETL_ETP=linacre ETL_SEUIL_NEIGE=1 \
      ETL_EPOCHS=30 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-${tag}.txt" 2>&1
  echo "[paquet] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-${tag}.txt" | tr '\n' ' ')"
}
lancer "outv-paq1"  ETL_TAG="-paq1"  ETL_INIT_HYDROTEL=1 ETL_EPOCHS=1   # CONTROLE : doit valoir ~0.77
lancer "outv-paq30" ETL_TAG="-paq30" ETL_INIT_HYDROTEL=1                # puis optimisation libre
echo "[paquet] $(date +%H:%M) TERMINE"
