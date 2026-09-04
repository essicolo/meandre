#!/usr/bin/env bash
# BALAYAGE DE RECHARGE dans le protocole du PILOTE (le seul comparable au record 0.7880).
# Le diagnostic hivernal du 2026-08-19 montre que fevrier est un deficit de STOCKAGE :
# quand la neige s'accumule (2022, 2023) le modele ne rend que 0.39 et 0.62 de l'observe,
# quand elle fond (2024) il fait 1.00, alors que l'observe coule a 13-15 m3/s dans les
# trois cas. La riviere reelle est portee par une reserve remplie a l'automne.
# Les recessions hivernales pures des jauges (1316 segments) donnent 0.0273 /j.
# Le champion, lui, a un aquifere AFFAME (recharge 0.26 mm/an, 0.1 % du debit).
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
# GARDE-FOU RESTREINT A MEANDRE (correctif du 2026-08-19). Attendre la fin de TOUT
# python.exe bloquait la file indefiniment : Positron et l'environnement rqh d'Essi en
# font tourner en permanence. On ne compte que les processus dont la ligne de commande
# porte le depot meandre. (Rappel : pgrep/ps du shell POSIX ne voient pas les processus
# Windows, il faut passer par PowerShell.)
meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '
 '
}
attendre() {
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do
    sleep 120
  done
}
run() { local tag="$1"; shift; attendre
  echo "[rech] $(date +%H:%M) $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WET=0 ETL_WSNOW=0.3 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" ETL_AQUIFER=1 ETL_KGW=0.0273 \
      ETL_KREC_LIBRE=1 ETL_KREC_GEL=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-${tag}.txt" 2>&1
  echo "[rech] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-${tag}.txt" | tr '\n' ' ')"
}
# 1) ecran en INFERENCE PURE sur les poids du champion (copie -ctl), 0 epoque
for K in 5e-6 1e-5 2e-5 3e-5; do
  run "rech0-$K" ETL_TAG="-ctl" ETL_EPOCHS=0 ETL_KREC="$K"
done
echo "[rech] $(date +%H:%M) ECRAN TERMINE"
