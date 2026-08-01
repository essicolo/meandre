"""Régionalisation amortie : fonction CONTINUE (coordonnées + attributs bruts) ->
paramètres, ajustée sur les calibrations régionales déjà faites. L'étiquette de
région n'est JAMAIS une variable ; elle sert uniquement de bloc de validation
(leave-one-region-out) pour éviter la fuite spatiale entre tronçons voisins.

  python .runs/quebec/amortized.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, torch, xgboost as xgb
from meandre.model import HydroModel
from meandre.data.basin_cache import BasinCache

# régions dont on possède une calibration (checkpoint) — sources d'expérience
SRC = {
    "gasp": ".runs/quebec/checkpoints/best-gasp-etl-ds.pt",
    "sagu": ".runs/quebec/checkpoints/best-sagu-etl-aquifer.pt",
    "mont": ".runs/quebec/checkpoints/best-mont-etl-ds.pt",
    "outv": ".runs/quebec/checkpoints/best-outv-etl-canon.pt",
}
CIBLES = ["K_sat_1", "K_sat_2", "K_sat_3", "porosity_1", "Z2", "Z3", "C_f", "T_melt",
          "K_c", "k_gw", "K_musk_hours", "x_musk"]
LOG = {"K_sat_1", "K_sat_2", "K_sat_3", "k_gw"}
raw = pd.read_parquet("D:/meandre-data/quebec/territorial-raw-QC.parquet")

rows = []
for reg, ck in SRC.items():
    if not os.path.exists(ck):
        print(f"[{reg}] checkpoint absent, ignoré"); continue
    h = BasinCache(f"D:/meandre-data/quebec/{reg}.duckdb" if reg != "slso" else ".runs/slso/data/slso.duckdb").load(device="cpu")
    m = HydroModel(n_nodes=h["n_nodes"], n_territorial=h["territorial"].n_features, n_forcing=6,
                   use_temporal=False, use_residual=False, use_travel_time_attn=False, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                   use_latent_codes=False, latent_mode="additive", spatial_melt=True,
                   routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
                   use_aquifer=True)
    m.load(ck); m.eval()
    with torch.no_grad():
        sp = m.spatial_encoder(h["node_coords"], h["territorial"].data)
    sub = raw[raw.region == reg].reset_index(drop=True)
    if len(sub) != h["n_nodes"]:
        print(f"[{reg}] désaccord nœuds ({len(sub)} vs {h['n_nodes']}), ignoré"); continue
    d = sub.drop(columns=["region"]).copy()
    nc = h["node_coords"].numpy()
    d["lon"], d["lat"] = nc[:, 0], nc[:, 1]
    for c in CIBLES:
        v = getattr(sp, c).detach().numpy().astype(float)
        d["y_" + c] = np.log(np.clip(v, 1e-9, None)) if c in LOG else v
    d["region"] = reg
    rows.append(d)
    print(f"[{reg}] {len(d)} exemples extraits", flush=True)

data = pd.concat(rows, ignore_index=True)
feats = [c for c in data.columns if not c.startswith("y_") and c != "region"]
print(f"\njeu : {len(data)} exemples, {len(feats)} variables, {data.region.nunique()} régions\n")

print(f"{'cible':>13} | " + " | ".join(f"{r:>7}" for r in SRC) + " |  moyenne")
resume = {}
for cible in CIBLES:
    y = data["y_" + cible].values
    scores = {}
    for held in SRC:
        tr = data.region != held; te = ~tr
        if te.sum() == 0: continue
        mdl = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0)
        mdl.fit(data.loc[tr, feats], y[tr.values])
        pred = mdl.predict(data.loc[te, feats])
        ytrue = y[te.values]
        ss = 1 - ((pred - ytrue) ** 2).sum() / max(((ytrue - ytrue.mean()) ** 2).sum(), 1e-12)
        scores[held] = ss
    resume[cible] = scores
    print(f"{cible:>13} | " + " | ".join(f"{scores.get(r, float('nan')):7.2f}" for r in SRC) +
          f" | {np.mean(list(scores.values())):8.2f}")
print("\n(R² leave-one-region-out : >0 = la fonction continue transfère à une région jamais vue ;")
print(" <0 = la moyenne des autres régions ferait mieux, donc aucun signal transférable)")
