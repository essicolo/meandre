#!/usr/bin/env bash
# Deux évaluations COURTES (inférence), à glisser entre deux entraînements de la file.
# Sépare les trois causes possibles de l'écart validation/tenu de côté : perte fautive,
# sur-ajustement, ou simple différence de période climatique.
cd /c/Users/parse01/documents-locaux/GitHub/meandre || exit 1
attendre() {
  while tasklist //FI "IMAGENAME eq python.exe" //FO CSV 2>/dev/null | grep -qi '"python.exe"'; do
    sleep 120
  done
}
attendre
echo "[periodes] $(date +%H:%M) modele ANCRE"
PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb MEANDRE_NSUBSTEP=64 \
  .venv/Scripts/python.exe .runs/quebec/eval_periodes.py outv \
  > /d/meandre-data/quebec/log-periodes-ancre.txt 2>&1
attendre
echo "[periodes] $(date +%H:%M) modele ENTRAINE"
PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb MEANDRE_NSUBSTEP=64 \
  .venv/Scripts/python.exe .runs/quebec/eval_periodes.py outv .runs/quebec/checkpoints/best-outv-etl-sain.pt \
  > /d/meandre-data/quebec/log-periodes-entraine.txt 2>&1
echo "[periodes] $(date +%H:%M) TERMINE"
