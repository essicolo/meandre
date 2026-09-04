#!/bin/bash
LOG=/mnt/c/Users/parse01/documents-locaux/GitHub/meandre/.runs/slso/logs/phase2-quantile.log
DEADLINE=$((SECONDS + 2700))   # 45 min
while [ $SECONDS -lt $DEADLINE ]; do
    if grep -qE "Epoch +0 \|" "$LOG" 2>/dev/null; then echo "EPOCH0_DONE"; exit 0; fi
    if grep -qE "Traceback|Error" "$LOG" 2>/dev/null; then echo "ERROR_DETECTED"; exit 2; fi
    sleep 30
done
echo "TIMEOUT"
exit 1
