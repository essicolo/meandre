#!/usr/bin/env bash
# FONTE ETI (radiation reelle) sur OUTV et GASP -- demarre a la fin de la flotte gen3
# (accord d'Essi, 2026-08-23 : « ok demarre des la fin de la flotte »).
#
# LA THESE (R42) : l'amplitude saisonniere amp=0.5 est un SUBSTITUT du cycle radiatif
# que le degre-jour n'a pas. Juste en continental (OUTV), fausse en maritime (GASP,
# suspect n° 1 du manteau a 2x : la fonte de mi-hiver y est turbulente et pluviale,
# pas radiative). L'ETI remplace le substitut par la variable physique : fonte =
# tf*(T-seuil) + srf*(1-albedo)*FB, FB reel de CaSR (canal 6, build_swin_region.py).
# PAS de sinusoide en ETI : si la these tient, OUTV garde son manteau ~1.0 SANS amp,
# et GASP redescend vers 1.0. Le dernier scalaire nival universel se dissout dans le
# forcage, comme le seuil s'est dissous dans le bulbe humide (R35).
#
# JUGES, dans l'ordre : ligne CanSWE (OUTV doit RESTER ~1.0 ; GASP doit DESCENDRE de
# 1.9 vers 1.0) ; date de disparition ; puis les scores contre gen1 (OUTV 0.7963 en
# recette int1, GASP 0.7134).
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
PLAT="C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
JOURNAL=/d/meandre-data/quebec

meandre_actifs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*GitHub*meandre*' }).Count" 2>/dev/null | tr -d '\r\n '
}

lancer() {
  local reg="$1"; shift
  while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 180; done
  # le cache sw_in doit exister (construit en parallele de la flotte)
  if [ ! -f "$JOURNAL/forcing-${reg}-swin.nc" ]; then
    echo "[eti] $(date +%H:%M) $reg SAUTE : forcing-${reg}-swin.nc absent"
    return
  fi
  echo "[eti] $(date +%H:%M) demarrage $reg"
  env MEANDRE_NSUBSTEP=64 JOINT_FX_SUFFIX=-hyb ETL_FORCE=1 \
      ETL_REGION="$reg" ETL_WSNOW=0 ETL_NO_LATENT=1 \
      ETL_ETP=linacre ETL_INIT_HYDROTEL=sauf_ks \
      ETL_MELT_DIR="$PLAT/$(echo $reg | tr a-z A-Z)_LN24HA_2020" \
      ETL_WET=0.4 ETL_SEUIL_NEIGE=0 ETL_SEUIL_TWB=-0.8 \
      ETL_ETI=1 \
      ETL_AQUIFER=1 ETL_KREC_LIBRE=1 ETL_KGW_FIELD=1 \
      ETL_TAG="-eti1" ETL_EPOCHS=30 ETL_CMP_NEIGE=1 ETL_STOCKS=1 ETL_AUX=1 "$@" \
      .venv/Scripts/python.exe .runs/quebec/etl_run.py \
      > "$JOURNAL/log-eti-${reg}.txt" 2>&1
  echo "[eti] $(date +%H:%M) fini $reg : $(grep -a 'HELD-OUT' "$JOURNAL/log-eti-${reg}.txt" | tr '\n' ' ')$(grep -a 'neige, simule/mesure' "$JOURNAL/log-eti-${reg}.txt" | tr '\n' ' ')"
}

# attendre la fin de la flotte gen3 (7 regions en cours)
while [ "$(meandre_actifs)" != "0" ] && [ -n "$(meandre_actifs)" ]; do sleep 300; done

lancer "outv"
lancer "gasp"

echo "[eti] $(date +%H:%M) TERMINE"
