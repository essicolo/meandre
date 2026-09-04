#!/usr/bin/env bash
# Superviseur du run occupation (consigne Essi 2026-08-18) : relance automatique
# jusqu'a 2 fois si arret EXTERNE (pas de HELD-OUT dans le log = fin anormale),
# registre horodate des interruptions pour correlation avec les jobs rqh/snakemake.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
LOG=.runs/slso/_zn_occ_train2.log
LEDGER=.runs/slso/_zn_occ_interruptions.log
MAX=2
N=0
while :; do
  START=$(date "+%Y-%m-%d %H:%M:%S")
  echo "$START DEMARRAGE tentative $N (warm-start du best courant)" >> "$LEDGER"
  MARK=$(wc -c < "$LOG" 2>/dev/null || echo 0)
  .venv/Scripts/python.exe -u .runs/slso/slso.py .runs/slso/config/slso-casr-zn-occ.toml >> "$LOG" 2>&1
  RC=$?
  END=$(date "+%Y-%m-%d %H:%M:%S")
  if tail -c +$MARK "$LOG" | grep -aq "HELD-OUT"; then
    echo "$END FIN NORMALE tentative $N (rc=$RC, held-out present)" >> "$LEDGER"
    break
  fi
  echo "$END INTERRUPTION tentative $N (rc=$RC, pas de held-out)" >> "$LEDGER"
  N=$((N+1))
  if [ $N -gt $MAX ]; then
    echo "$END ABANDON apres $MAX relances automatiques" >> "$LEDGER"
    break
  fi
  sleep 60
done
