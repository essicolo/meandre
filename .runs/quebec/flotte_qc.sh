#!/usr/bin/env bash
# FLOTTE PROVINCIALE : calibration LOCALE de chaque région avec la recette canonique
# (ET apprise MOD16 x debias bilan regional + fonte supervisee MOD10 + K_sat_1 0.04
#  + aquifere krec 5e-5 + k_gw = queue de recession regionale, forcage CaSR brut).
# La calibration locale vaut ~+0.10 de KGE tenu de cote vs le transfert gaspesien
# (mesure sur SLSO 0.532->0.631 et SLNO). Sequentiel : une seule carte graphique.
set -u
cd "$(dirname "$0")/../.."
REGIONS="${1:-outv abit outm labi cnda cndb cndc cndd cnde}"
for R in $REGIONS; do
  DS=$(.venv/Scripts/python.exe -c "import json;print(json.load(open('reports/deploy_adapters.json'))['$R']['debias_et'])")
  KGW=$(.venv/Scripts/python.exe -c "import json;v=json.load(open('reports/deploy_adapters.json'))['$R']['k_recession_queue'];print(v if v else 0.05)")
  echo "=== [$R] debias_et=$DS k_gw=$KGW $(date '+%H:%M') ==="
  JOINT_FX_SUFFIX=-none ETL_REGION=$R ETL_EPOCHS=${ETL_EPOCHS:-12} ETL_TAG=-qc \
    ETL_KSAT1=0.04 ETL_WSNOW=0.3 ETL_AQUIFER=1 ETL_KREC=5e-5 \
    ETL_KGW=$KGW ETL_DEMAND_SCALE=$DS \
    .venv/Scripts/python.exe .runs/quebec/etl_run.py 2>&1 | tee "D:/meandre-data/quebec/log-qc-$R.txt" \
    | grep -E "held-out|HELD|best|epoch .* val_kge|ERREUR|Traceback"
  echo "=== [$R] fini $(date '+%H:%M') ==="
done
