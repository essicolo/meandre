#!/usr/bin/env bash
# NUIT 2026-08-22 -> 23 : FAIRE RESPIRER L3 (drainage non lineaire a la vanne, R37).
#
# Ce que la partition des flux d'INT1 a montre : le debit de base est PLAT (8.9 a 11.8
# mm/mois toute l'annee) et 90 des 130 mm de la production d'avril partent en SURFACE,
# parce que L3, pleine, ne peut rien recevoir de la fonte. GRACE reclame l'inverse :
# un stockage de mars plus bas (biais +58) et une reserve tenue en mai (-45).
# q3 = krec*z3*ths3*(theta/ths)^n draine au meme plafond a saturation mais se coupe en
# dessous : L3 se vide l'hiver, la fonte la re-remplit. n=1 == fidele exact (teste).
#
# G1-G3 : n = 4, 8, 16 en INFERENCE sur le point de reprise d'INT1 (6 min le point).
#   Juges : biais GRACE de mars (+58 a faire baisser) et de mai (-45 a remonter),
#   ligne nappe, mois de debit. G0 = INT1 relu tel quel (controle, n absent).
# INT2 : entrainement complet avec n=8 (mediane), pour laisser krec et le reste se
#   co-adapter a la nouvelle vanne. patience=8 coupera seule.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}

# recette INT1, a l'identique ; seul ETL_L3_EXP varie via "$@"
lancer() {
  local tag="$1"; local epochs="$2"; shift 2
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 120; done
  echo "[l3] $(date +%H:%M) demarrage $tag"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION=outv ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/OUTV_LN24HA_2020" \
      ETL_WET=0.4 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 ETL_MELT_SAISON=0.5 \
      ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW=0.0273 \
      ETL_EPOCHS="$epochs" ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-${tag}.txt" 2>&1
  echo "[l3] $(date +%H:%M) fini $tag : $(grep -a 'HELD-OUT' "$JOURNAL/log-${tag}.txt" | tr '\n' ' ')"
}

lancer "l3-G0" 0 ETL_TAG="-int1"
lancer "l3-G1" 0 ETL_TAG="-int1" ETL_L3_EXP=4
lancer "l3-G2" 0 ETL_TAG="-int1" ETL_L3_EXP=8
lancer "l3-G3" 0 ETL_TAG="-int1" ETL_L3_EXP=16
lancer "l3-INT2" 30 ETL_TAG="-int2" ETL_L3_EXP=8

echo "[l3] $(date +%H:%M) TERMINE"
