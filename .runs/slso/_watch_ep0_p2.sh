#!/bin/bash
LOG=/mnt/c/Users/parse01/documents-locaux/GitHub/meandre/.runs/slso/logs/phase2-grace.log
DEADLINE=$((SECONDS + 2700))   # 45 min (Phase 2 frozen, ~25min/epoch attendu)
while [ $SECONDS -lt $DEADLINE ]; do
    if grep -qE "Epoch +0 \|" "$LOG" 2>/dev/null; then
        echo "EPOCH0_DONE"
        exit 0
    fi
    sleep 30
done
echo "TIMEOUT"
exit 1
