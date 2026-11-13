#!/usr/bin/env bash
# Nuit du 2026-11-13 : modele provincial reentraine sur les donnees corrigees du 14 novembre
# (texture R367, SIIGSOL partout, milieux humides MONT et SLNO, OUTM). Chaque graine part de sa
# version figee du 12 novembre, 8 epoques, recette .runs/quebec/recipe.toml inchangee ; puis
# passe 2022-2024 de chaque graine et tete de quantiles de la graine 7 (regle R388). Depart a 17 h.
set -u
while [ "$(date +%H%M)" -lt 1700 ]; do sleep 60; done
C=/home/essi/Documents/git/meandre-worktrees/4ce37ed
K=/home/essi/meandre-data/checkpoints/cultures-vent
O=/home/essi/meandre-data/analysis/donnees-2026-11-14
PY=/home/essi/Documents/git/meandre/.venv/bin/python
ENV="PYTHONPATH=$C MEANDRE_DATA=/home/essi/meandre-data-2026-11-14 MEANDRE_DERIVES=/home/essi/meandre-data/derives MEANDRE_PLATEFORMES=/home/essi/plateformes-hydrotel-2026-11-14 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ETL_FORCE=1 MEANDRE_EMPREINTES=warn"
CK=$C/.runs/quebec/checkpoints
cd $C
for s in 7 1234 2024; do
  env $ENV $PY -u -m meandre run .runs/quebec/recipe.toml --region quebec --epochs 8 --seed $s --tag=-donnees-1114-s$s --warm-from $K/best-quebec-etl-cultures-vent-s$s.pt > $O/entrainement-s$s.log 2>&1
  echo "entrainement s$s $? $(date +%H:%M)" >> $O/marker.txt
  env $ENV $PY -u -m meandre run .runs/quebec/recipe.toml --region quebec --epochs 0 --seed $s --tag=-eval-donnees-1114-s$s --warm-from $CK/best-quebec-etl-donnees-1114-s$s.pt --dump-q $O/q-donnees-1114-s$s.npz > $O/evaluation-s$s.log 2>&1
  echo "evaluation s$s $? $(date +%H:%M)" >> $O/marker.txt
done
env $ENV MEANDRE_SAUVER_EPOQUES=1 $PY -u -m meandre run experiments/probabilistic/quantile-gele-2026-11.toml --region quebec --epochs 12 --seed 7 --tag=-quantile-donnees-1114-s7 --warm-from $CK/best-quebec-etl-donnees-1114-s7.pt > $O/quantile.log 2>&1
echo "quantile $? $(date +%H:%M)" >> $O/marker.txt
env $ENV $PY -u -m meandre run experiments/probabilistic/quantile-gele-2026-11.toml --region quebec --epochs 0 --seed 7 --tag=-eval-quantile-donnees-1114-s7 --warm-from $CK/best-quebec-etl-quantile-donnees-1114-s7-e12.pt --dump-q $O/q-quantile-donnees-1114-s7.npz > $O/evaluation-quantile.log 2>&1
echo "evaluation quantile $? $(date +%H:%M)" >> $O/marker.txt
mkdir -p /home/essi/meandre-data/checkpoints/donnees-1114 && cp $CK/best-quebec-etl-*donnees-1114* /home/essi/meandre-data/checkpoints/donnees-1114/
