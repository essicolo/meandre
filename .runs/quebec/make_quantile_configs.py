"""Configs tête quantile par région (recette zn-quantile prouvée : backbone gelé,
médiane = mu, pinball K=6, 15 epochs). À lancer APRÈS la file d'entraînement.
  python .runs/quebec/make_quantile_configs.py
"""
import os, re, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import tomllib

REGIONS = ["labi", "cndc", "cnda", "cnde", "cndb", "cndd", "abit", "mont",
           "sagu", "outm", "slno", "outv", "gasp"]
OUTDIR = Path(".runs/quebec/config")
for r in REGIONS:
    base = Path(f".runs/quebec/config/{r}.toml")
    if not base.exists(): continue
    s = base.read_text(encoding="utf-8")
    s = re.sub(r'checkpoint = "[^"]*"', f'checkpoint = "checkpoints/best-{r}-quantile.pt"', s)
    s = re.sub(r'fields_nc = "[^"]*"', f'fields_nc = "D:/meandre-data/quebec/results/fields-{r}-quantile.nc"', s)
    s = re.sub(r'reach_parquet = "[^"]*"', f'reach_parquet = "D:/meandre-data/quebec/results/reach-{r}-quantile.parquet"', s)
    s = s.replace("n_epochs = 30", "n_epochs = 15")
    s = re.sub(r'^lr = [^\n]*', 'lr = 1e-3  # tete quantile fresh-init', s, flags=re.M)
    s = re.sub(r'best_metric = "kge_median"[^\n]*', 'best_metric = "nll"\nbest_metric_tolerance = 0.005', s)
    s = s.replace('warm_start = false', f'warm_start = true\nwarm_start_from = "checkpoints/best-{r}.pt"\nfreeze_spatial = true\nfreeze_temporal = true\nfreeze_backbone = true')
    s = s.replace("[loss]", '[loss]\nnll_distribution = "quantile"\nquantile_taus = [0.05, 0.10, 0.25, 0.75, 0.90, 0.95]\nw_quantile = 1.0', 1)
    out = OUTDIR / f"{r}-quantile.toml"
    out.write_text(s, encoding="utf-8")
    tomllib.load(open(out, "rb"))
    print(f"{r}: ok")
print("configs quantile prêtes")
