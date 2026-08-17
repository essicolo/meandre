#!/usr/bin/env bash
# TAUX REDUIT POUR DEPARTS A CHAUD. La trajectoire mgk30 montre une DESTABILISATION,
# pas un sur-ajustement : val 0.834 -> 0.855 -> effondrement a 0.61 (ep. 5) -> vingt
# epoques a remonter. Le lr=5e-4 casse le paquet co-adapte au depart ; on teste 2e-4
# (le lr_finetune de la config) et 1e-4 sur les DEUX socles (LN et MG24HK).
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
run() { local tag="$1"; shift; attendre
  echo "[dx] $(date +%H:%M) $tag"
  env MEANDRE_ABANDON=6:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_EPOCHS=30 ETL_WET=0 ETL_WSNOW=0.3 ETL_AQUIFER=0 ETL_NO_LATENT=1 \
      ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-${tag}.txt" 2>&1
  echo "[dx] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-${tag}.txt" | tr '\n' ' ')"
}
run "mgk-lr2" ETL_TAG="-mgk-lr2" ETL_MEMBRE=MG24HK ETL_ETP=mcguinness ETL_LR=2e-4
run "ln-lr2"  ETL_TAG="-ln-lr2"  ETL_ETP=linacre ETL_LR=2e-4
run "mgk-lr1" ETL_TAG="-mgk-lr1" ETL_MEMBRE=MG24HK ETL_ETP=mcguinness ETL_LR=1e-4
echo "[dx] $(date +%H:%M) TERMINE"
