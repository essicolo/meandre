#!/usr/bin/env bash
# OBJECTIF : depasser le MEILLEUR membre d'Hydrotel (0.83-0.85 par station) depuis le
# socle a 0.7810 sur OUTV. On mesure D'ABORD ou est l'ecart, puis on teste les leviers.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
run() { local tag="$1"; shift; attendre
  echo "[pousser] $(date +%H:%M) $tag"
  env MEANDRE_ABANDON=4:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_EPOCHS=30 ETL_WET=0 ETL_WSNOW=0.3 ETL_AQUIFER=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-${tag}.txt" 2>&1
  echo "[pousser] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-${tag}.txt" | tr '\n' ' ')"
}
# 0. DIAGNOSTIC d'abord : ou est l'ecart au meilleur membre ?
attendre
echo "[pousser] $(date +%H:%M) diagnostic"
PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb MEANDRE_NSUBSTEP=64 \
  .venv/Scripts/python.exe .runs/quebec/diagnostic_ecart.py outv \
  > /d/meandre-data/quebec/log-diag-ecart-outv.txt 2>&1
echo "[pousser] diagnostic fait"
# 1-4. les leviers, chacun un seul changement contre le socle a 0.7810
run "p-casr"   ETL_TAG="-p-casr"   JOINT_FX_SUFFIX=-none
run "p-etpmlp" ETL_TAG="-p-etpmlp" ETL_ETP=appris ETL_DEMAND_SCALE=0.963
run "p-wet"    ETL_TAG="-p-wet"    ETL_WET=0.3
run "p-sanstws" ETL_TAG="-p-sanstws" ETL_WTWS=0
echo "[pousser] $(date +%H:%M) TERMINE"
