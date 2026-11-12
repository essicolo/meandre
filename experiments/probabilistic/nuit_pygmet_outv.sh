#!/usr/bin/env bash
# 2026-11-12, nuit : l'ensemble de forçage PyGMET corrige-t-il la forme de l'enveloppe ?
# Outaouais seul, modèle figé (graine 7) à poids figés, 2000-2024, évaluation 2022-2024.
# 1. Dix membres PyGMET recentrés sur CaSR (vent en m/s) à partir de leurs écarts à la moyenne des membres.
# 2. Passe de référence avec la tête de quantiles (débits et enveloppe aux stations).
# 3. Une passe par membre, débits aux stations.
set -u
C=/home/essi/Documents/git/meandre-worktrees/c2980a1
K=/home/essi/meandre-data/checkpoints/cultures-vent
O=/home/essi/meandre-data/analysis/pygmet-outv-2026-11-12
Q=/home/essi/meandre-data/quebec
PY=/home/essi/Documents/git/meandre/.venv/bin/python
ENV="PYTHONPATH=$C MEANDRE_DATA=/home/essi/meandre-data MEANDRE_DERIVES=/home/essi/meandre-data/derives MEANDRE_PLATEFORMES=/home/essi/plateformes-hydrotel PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ETL_FORCE=1 MEANDRE_EMPREINTES=warn"
mkdir -p $O
for m in 01 02 03 04 05 06 07 08 09 10; do
  [ -f $Q/forcing-outv-ens0$m.nc ] || $PY /home/essi/Documents/git/meandre-ubuntu/lancements/member_forcing.py $Q/forcing-outv-casr-brut-vent.nc /home/essi/meandre-data/pygmet-anomalies/anomalies-outv-ens0$m.nc $Q/forcing-outv-ens0$m.nc >> $O/membres.log 2>&1
done
echo "membres $(date +%H:%M)" >> $O/marker.txt
cd $C
env $ENV $PY -u -m meandre run experiments/probabilistic/quantile-gele-2026-11.toml --region outv --epochs 0 --seed 7 --tag=-pygmet-ref --warm-from $K/best-quebec-etl-quantile-gele-s7-e12.pt --dump-q $O/q-ref.npz > $O/ref.log 2>&1
echo "ref $? $(date +%H:%M)" >> $O/marker.txt
for m in 01 02 03 04 05 06 07 08 09 10; do
  env $ENV JOINT_FX_SUFFIX=-ens0$m $PY -u -m meandre run .runs/quebec/recipe.toml --region outv --epochs 0 --seed 7 --tag=-pygmet-ens0$m --warm-from $K/best-quebec-etl-cultures-vent-s7.pt --dump-q $O/q-ens0$m.npz > $O/ens0$m.log 2>&1
  echo "ens0$m $? $(date +%H:%M)" >> $O/marker.txt
done
echo "fin $(date +%H:%M)" >> $O/marker.txt
