#!/usr/bin/env bash
# Test APPARIE du verrou de fonte appris (R56).
#
# Bras 0 : verrou d'Hydrotel, seuils de fonte par classe ancres au calage.
# Bras 1 : verrou APPRIS, T_melt sur le decouvert plus deux retards de canopee non
#          negatifs empiles (conifere >= feuillu >= decouvert par construction).
#
# Seule PROV_CANOPEE change. Meme graine, meme nombre d'epoques, meme plateforme.
#
# Le juge n'est PAS le KGE seul. Le verrou existe pour placer la fonte dans le temps,
# donc ce qui compte d'abord est la neige : rapport de manteau mensuel contre CanSWE
# apparie site et jour, et date de disparition. Le debit d'avril et de mai vient
# ensuite. Un gain de KGE sans gain sur la neige serait une compensation de plus.
set -u
REG="${1:-sagu}"
EP="${PROV_EPOCHS:-5}"
OUT="/d/meandre-data/quebec/log-canopee-apparie-${REG}.txt"
: > "$OUT"

for BRAS in 0 1; do
  LOG="/d/meandre-data/quebec/log-canopee-${REG}-${BRAS}.txt"
  T0=$SECONDS
  # PROV_TAG DISTINCT PAR BRAS. Sans lui les deux bras ecrivent le meme
  # best-<tag>.pt et le second ECRASE le premier : les KGE survivent dans les
  # journaux, mais le checkpoint du temoin est perdu et on ne peut plus juger
  # la NEIGE des deux cotes, qui est pourtant le juge annonce par R56.
  PROV_TAG=canopee-${REG}-${BRAS} PROV_CANOPEE=$BRAS PROV_EPOCHS=$EP \
    .venv/Scripts/python.exe -u .runs/quebec/province.py "$REG" > "$LOG" 2>&1
  {
    echo "=== PROV_CANOPEE=$BRAS : $(( (SECONDS - T0) / 60 )) min ==="
    grep -E "Epoch |kge median|mediane provinciale|manteau|CanSWE|disparition" "$LOG" | tail -30
    echo
  } >> "$OUT"
done
echo "[canopee] TERMINE" >> "$OUT"
cat "$OUT"
