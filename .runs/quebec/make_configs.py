"""Génère les configs d'entraînement des régions Québec depuis la recette championne
(slso-casr-zn.toml : z_n latents + CaSR-corr + McGuinness + kge_median + multi-obj MODIS).
Adaptations : basin_db/forcing absolus (D:), n_forcing 6 (pas de DT_eff), sorties par région.
SLSO exclu (champion existant), VAUD exclu (aucune jauge — régionalisation plus tard).
  python .runs/quebec/make_configs.py
"""
import os, sys, re
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pathlib import Path
import tomllib

BASE = Path(".runs/slso/config/slso-casr-zn.toml").read_text(encoding="utf-8")
OUTDIR = Path(".runs/quebec/config"); OUTDIR.mkdir(parents=True, exist_ok=True)
Path(".runs/quebec/checkpoints").mkdir(exist_ok=True)
Path("D:/meandre-data/quebec/results").mkdir(exist_ok=True)
REGIONS = ["LABI", "CNDC", "CNDA", "CNDE", "CNDB", "CNDD", "ABIT", "MONT",
           "SAGU", "OUTM", "SLNO", "OUTV", "GASP"]  # ordre = petit -> gros

for reg in REGIONS:
    r = reg.lower()
    s = BASE
    s = re.sub(r'basin_db = "[^"]*"', f'basin_db = "D:/meandre-data/quebec/{r}.duckdb"', s)
    s = re.sub(r'basin_db_pc = "[^"]*"', f'basin_db_pc = "D:/meandre-data/quebec/{r}.duckdb"', s)
    s = re.sub(r'forcing_cache = "[^"]*"', f'forcing_cache = "D:/meandre-data/quebec/forcing-{r}.nc"', s)
    s = re.sub(r'checkpoint = "[^"]*"', f'checkpoint = "checkpoints/best-{r}.pt"', s)
    s = re.sub(r'fields_nc = "[^"]*"', f'fields_nc = "D:/meandre-data/quebec/results/fields-{r}.nc"', s)
    s = re.sub(r'reach_parquet = "[^"]*"', f'reach_parquet = "D:/meandre-data/quebec/results/reach-{r}.parquet"', s)
    s = re.sub(r'n_forcing = 7[^\n]*', 'n_forcing = 6      # P, T_min, T_max, R_n, u2, e_a (pas de DT_eff)', s)
    out = OUTDIR / f"{r}.toml"
    out.write_text(s, encoding="utf-8")
    c = tomllib.load(open(out, "rb"))
    print(f"{reg}: db={Path(c['paths']['basin_db']).name} forcing={Path(c['paths']['forcing_cache']).name} "
          f"nf={c['model']['n_forcing']} latent={c['model'].get('use_latent_codes')} ep={c['training']['n_epochs']}")
print(f"\n{len(REGIONS)} configs -> {OUTDIR}")
