#!/bin/bash
# Veilleur : sort dès que "Epoch    0 |" apparaît dans le log (ou timeout 30 min).
LOG=/mnt/c/Users/parse01/documents-locaux/GitHub/meandre/.runs/slso/logs/phase1-grace.log
DEADLINE=$((SECONDS + 1800))   # 30 min max
while [ $SECONDS -lt $DEADLINE ]; do
    if grep -qE "Epoch +0 \|" "$LOG" 2>/dev/null; then
        echo "EPOCH0_DONE"
        exit 0
    fi
    sleep 30
done
echo "TIMEOUT"
exit 1
