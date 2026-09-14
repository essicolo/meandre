#!/usr/bin/env bash
# Phase probabiliste locale : entraine la tete de quantiles sur le socle GELE, region
# par region, en repartant du point de reprise de la flotte.
#
# POURQUOI CE SCRIPT EXISTE (2026-09-10). Les configurations .runs/quebec/config/
# <reg>-quantile.toml, passees a .runs/slso/slso.py, produisent des exports
# INUTILISABLES : leur `warm_start_from` designe checkpoints/best-<reg>.pt, qui
# n'existe pas, et le pilote saute alors le depart a chaud SANS RIEN DIRE
# (slso.py : `if WARM_START and _ws_path.exists()`). La tete s'entraine donc sur une
# physique jamais entrainee, puis `freeze_backbone` la fige. Mesure sur cinq regions :
# le debit median de l'export s'ecarte de 27 a 49 % de celui de la flotte. Ces
# configurations ne portent pas non plus la recette du socle ni le forcage -budyko.
#
# Ce script passe par le pilote de la flotte, qui porte la recette, le bon forcage et
# le mode quantile, et qui REFUSE de demarrer sans depart a chaud.
#
#   bash .runs/quebec/file_quantile.sh            # les 9 regions disponibles
#   bash .runs/quebec/file_quantile.sh gasp sagu  # une selection
#   ETL_EPOCHS=4 bash .runs/quebec/file_quantile.sh gasp   # essai court
set -u
RACINE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$RACINE"
DATA=${MEANDRE_DATA:-D:/meandre-data}
PLATES=${MEANDRE_PLATEFORMES:-C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel}
FLOTTE="$DATA/quebec/flotte"
SORTIE="$DATA/quebec/rapport"
DEPART=${DEPART:--etl-fdsA-v4}
# Interpreteur : le venv du depot d'abord (Git Bash sous Windows n'a pas `python` dans
# le PATH), sinon celui de l'environnement (grappe, WSL).
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for _c in "$RACINE/.venv/Scripts/python.exe" "$RACINE/.venv/bin/python" "$(command -v python3 || true)" "$(command -v python || true)"; do
    [ -n "$_c" ] && [ -x "$_c" ] && PY="$_c" && break
  done
fi
if [ -z "$PY" ]; then
  echo "ARRET : aucun interpreteur python trouve. Poser PYTHON=<chemin>." >&2
  exit 2
fi
echo "[quantile] interpreteur : $PY"
EPOCHS=${ETL_EPOCHS:-8}
REGIONS=${*:-"cndc cndb abit sagu mont slso slno outv gasp"}   # la plus petite en tete : le controle prealable est plus rapide
mkdir -p "$SORTIE"

# CONTROLE PREALABLE (2026-09-10). Avant de depenser des heures, une passe a ZERO epoque sur la
# premiere region : elle charge le point de reprise, applique la recette et exporte le
# debit. Si ce debit n'est pas celui de l'export de la flotte au millionieme pres, la
# tete s'entrainerait sur un autre modele et la file s'arrete ici. C'est le controle
# qui manquait quand neuf regions ont ete entrainees pour rien.
PREM=$(echo $REGIONS | awk '{print $1}')
if [ "${SANS_CONTROLE PREALABLE:-0}" != 1 ]; then
  CK0="$FLOTTE/best-${PREM}${DEPART}.pt"
  MAJ0=$(echo "$PREM" | tr '[:lower:]' '[:upper:]')
  echo "[quantile] CONTROLE PREALABLE : controle a zero epoque sur $PREM"
  env ETL_CONFIG=.runs/quebec/config/socle.toml JOINT_FX_SUFFIX=-budyko       ETL_FORCE=1 ETL_REGION=$PREM ETL_COMPILE=0       ETL_MELT_DIR="$PLATES/LN24HA/${MAJ0}_LN24HA_2020"       ETL_QUANTILE=1 ETL_WARM_FROM="$CK0" ETL_TAG=-q1ctrl ETL_EPOCHS=0       ETL_DUMP_Q="$SORTIE/ctrl-$PREM.npz"       "$PY" -u .runs/quebec/etl_run.py > "$SORTIE/log-ctrl-$PREM.txt" 2>&1
  if [ $? -ne 0 ] || [ ! -f "$SORTIE/ctrl-$PREM.npz" ]; then
    echo "[quantile] CONTROLE PREALABLE : la passe de controle n'a pas abouti (le pilote a echoue)." >&2
    tail -20 "$SORTIE/log-ctrl-$PREM.txt" >&2
    echo "           File NON lancee." >&2
    exit 5
  fi
  if ! "$PY" .runs/quebec/controle_quantile.py --fichier "$SORTIE/ctrl-$PREM.npz" --region "$PREM"; then
    echo "[quantile] CONTROLE PREALABLE REFUSE : le debit exporte ne reproduit pas celui de la flotte." >&2
    echo "           Voir $SORTIE/log-ctrl-$PREM.txt. File NON lancee." >&2
    exit 6
  fi
  echo "[quantile] controle prealable passe, la file demarre."
fi
for REG in $REGIONS; do
  CKPT="$FLOTTE/best-${REG}${DEPART}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[quantile] $REG SAUTE : $CKPT absent." >&2
    continue
  fi
  MAJ=$(echo "$REG" | tr '[:lower:]' '[:upper:]')
  echo "[quantile] $REG : depart a chaud depuis $(basename "$CKPT"), $EPOCHS epoques"
  env ETL_CONFIG=.runs/quebec/config/socle.toml \
      JOINT_FX_SUFFIX=-budyko \
      ETL_FORCE=1 ETL_REGION=$REG ETL_COMPILE=${ETL_COMPILE:-0} \
      ETL_MELT_DIR="$PLATES/LN24HA/${MAJ}_LN24HA_2020" \
      ETL_QUANTILE=1 ETL_WARM_FROM="$CKPT" \
      ETL_TAG=-q1 ETL_EPOCHS=$EPOCHS \
      ETL_DUMP_Q="$SORTIE/quant-$REG.npz" \
      "$PY" -u .runs/quebec/etl_run.py 2>&1 | tee "$SORTIE/log-quant-$REG.txt"
  echo "[quantile] $REG termine."
done
echo "[quantile] file terminee. Controle du millionieme :"
echo "  $PY .runs/quebec/controle_quantile.py"
