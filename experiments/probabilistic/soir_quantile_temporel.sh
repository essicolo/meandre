#!/usr/bin/env bash
# 2026-11-13, soir : tête de quantiles avec terme temporel (R551) sur la physique figée de la graine 7,
# départ de la tête d'hier (e12), 12 époques, puis évaluation 2022-2024 de la dernière époque avec
# l'enveloppe aux stations. Démarrage différé à 17 h.
set -u
while [ "$(date +%H%M)" -lt 1700 ]; do sleep 60; done
C=/home/essi/Documents/git/meandre-worktrees/c2980a1
K=/home/essi/meandre-data/checkpoints/cultures-vent
O=/home/essi/meandre-data/analysis/quantile-temporel-2026-11-13
PY=/home/essi/Documents/git/meandre/.venv/bin/python
ENV="PYTHONPATH=$C MEANDRE_DATA=/home/essi/meandre-data MEANDRE_DERIVES=/home/essi/meandre-data/derives MEANDRE_PLATEFORMES=/home/essi/plateformes-hydrotel PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ETL_FORCE=1 MEANDRE_EMPREINTES=warn"
mkdir -p $O; cd $C
env $ENV MEANDRE_SAUVER_EPOQUES=1 $PY -u -m meandre run experiments/probabilistic/quantile-gele-2026-11-temporel.toml --region quebec --epochs 12 --seed 7 --tag=-quantile-temporel-s7 --warm-from $K/best-quebec-etl-quantile-gele-s7-e12.pt > $O/entrainement.log 2>&1
echo "entrainement $? $(date +%H:%M)" >> $O/marker.txt
E12=$C/.runs/quebec/checkpoints/best-quebec-etl-quantile-temporel-s7-e12.pt
cp $E12 ${E12%.pt}.recipe.toml $K/ 2>/dev/null
env $ENV $PY -u -m meandre run experiments/probabilistic/quantile-gele-2026-11-temporel.toml --region quebec --epochs 0 --seed 7 --tag=-eval-quantile-temporel-s7 --warm-from $E12 --dump-q $O/q-quantile-temporel-s7.npz > $O/evaluation.log 2>&1
echo "evaluation $? $(date +%H:%M)" >> $O/marker.txt
