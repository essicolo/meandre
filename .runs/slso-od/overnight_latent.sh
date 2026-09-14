#!/bin/bash
# Expérience nuit 2026-06-15 : effet aléatoire ADDITIF (mixed-effects) sur le
# mini-bassin multi-jauges. Balayage du shrinkage (biais-variance) + diagnostic
# train vs val (overfit). Scientifiquement sound : per-station, held-out,
# marge exigée, résultats négatifs rapportés.
cd "C:/Users/parse01/documents-locaux/GitHub/meandre" || exit 1
RES=.runs/slso-od/_overnight_results.txt
echo "=== Expérience codes latents ADDITIFS — $(date) ===" > $RES

gpu_wait(){ until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" -lt 500 ]; do sleep 3; done; }
evalck(){ uv run python .runs/slso-od/eval_mini_2win.py "$1" "$2" 2>/dev/null | tail -1; }

# Référence : base sans codes (déjà entraîné).
echo "--- BASE (sans codes) ---" | tee -a $RES
evalck .runs/slso-od/config/slso-od-mini.toml .runs/slso-od/checkpoints/best-mini.pt | tee -a $RES

# Balayage shrinkage, effet ADDITIF, 20 epochs.
BASE=.runs/slso-od/config/slso-od-mini-latent.toml
for REG in 0.0 1e-4 1e-3 1e-2; do
  TAG=$(echo $REG | tr '.' 'p' | tr '-' 'm')
  CFG=.runs/slso-od/config/_mini_lat_${TAG}.toml
  CK=checkpoints/best-mini-lat-${TAG}.pt
  python - "$BASE" "$CFG" "$REG" "$CK" <<'PY'
import sys,re
base,cfg,reg,ck=sys.argv[1:5]
s=open(base,encoding='utf-8').read()
s=re.sub(r'w_latent_reg = [^\n]+', f'w_latent_reg = {reg}', s)
if 'latent_mode' not in s:
    s=s.replace('use_latent_codes = true','use_latent_codes = true\nlatent_mode = "additive"')
s=re.sub(r'checkpoint = "[^"]+"', f'checkpoint = "{ck}"', s)
s=re.sub(r'n_epochs = \d+','n_epochs = 20',s)
open(cfg,'w',encoding='utf-8').write(s)
PY
  echo "--- ADDITIF shrinkage w_latent_reg=$REG ---" | tee -a $RES
  gpu_wait
  uv run python .runs/slso/slso.py "$CFG" > ".runs/slso-od/_mini_lat_${TAG}.log" 2>&1
  evalck "$CFG" ".runs/slso-od/$CK" | tee -a $RES
done

echo "=== DONE $(date) ===" | tee -a $RES
