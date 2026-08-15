#!/usr/bin/env bash
# QUATRE PISTES D'AMELIORATION sur le SOCLE (questions d'Essi, 15 aout). Chacune commence
# par un controle a zero epoque quand c'est possible, et l'abandon coupe a l'epoque 4
# sous 0.60. Reference a battre : OUTV 0.7810.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
attendre() {
  while (tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'); do
    sleep 120
  done
}
run() {  # $1 = tag, reste = surcharges
  local tag="$1"; shift
  attendre
  echo "[amel] $(date +%H:%M) $tag"
  env MEANDRE_ABANDON=4:0.60 MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_EPOCHS=30 ETL_WET=0 ETL_WSNOW=0.3 ETL_AQUIFER=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_SEUIL_NEIGE=1 ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "/d/meandre-data/quebec/log-outv-${tag}.txt" 2>&1
  echo "[amel] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "/d/meandre-data/quebec/log-outv-${tag}.txt" | tr '\n' ' ')"
}
# 1. FORCAGE : le socle tient-il sur le CaSR brut (+16 % de pluie) ? Controle puis 30 ep.
run "am-casr0"  ETL_TAG="-am-casr0"  JOINT_FX_SUFFIX=-none ETL_EPOCHS=0
run "am-casr30" ETL_TAG="-am-casr30" JOINT_FX_SUFFIX=-none
# 2. ETP APPRISE (module MLP) contre Linacre calee, tout le reste identique.
run "am-etpmlp" ETL_TAG="-am-etpmlp" ETL_ETP=appris ETL_DEMAND_SCALE=0.963
# 3. MODIS ET rallume, maintenant que l'ETR couvre tout le territoire et que
#    l'appariement 8 jours est corrige (les deux causes de son echec de juillet).
run "am-wet"    ETL_TAG="-am-wet"    ETL_WET=0.3
# 4. GRACE seule (w_tws=0.2 du fichier de config) : deja actif, on mesure SANS pour voir
#    ce qu'il apporte ou coute.
run "am-sanstws" ETL_TAG="-am-sanstws" ETL_WTWS=0
echo "[amel] $(date +%H:%M) TERMINE"
