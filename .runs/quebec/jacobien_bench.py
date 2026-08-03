"""BANC JACOBIEN SIGNATURES x PARAMETRES. Généralise freshet_bench (qui mesurait une
seule dérivée : 3 j de centre de masse par °C de seuil de fonte) à la matrice complète.
Pour chaque paramètre perturbé en INFERENCE PURE, mesure le déplacement de chaque
signature hydrologique aux stations. La matrice obtenue s'inverse ensuite au moindre
carré, nœud par nœud, contre le champ krigé des signatures : le modèle est alors
contraint de reproduire simultanément récession, date de crue et variabilité observées,
avec les paramètres physiques comme seules inconnues (problème inverse mesuré).

  PYTHONIOENCODING=utf-8 python .runs/quebec/jacobien_bench.py gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand

REG = (sys.argv[1] if len(sys.argv) > 1 else "gasp").lower()
CKPT = os.environ.get("JAC_CKPT", ".runs/quebec/checkpoints/best-gasp-etl-ds.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("JOINT_FX_SUFFIX", "-none")
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True,
    column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
    use_temperature=False, use_latent_codes=False, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=True).to(DEVICE)
m.load(CKPT); m.eval(); m.vertical_column.etp_channel = 6
m.vertical_column.compile_column = False
_orig = m.spatial_encoder.forward
times = pd.to_datetime(r["times"]); t0 = td.train_slice.start
tt = pd.DatetimeIndex(times[t0:])


def sim(fn):
    m.spatial_encoder.forward = _orig if fn is None else \
        (lambda *a, **k: (lambda sp: (fn(sp), sp)[1])(_orig(*a, **k)))
    with torch.no_grad():
        Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
    return Q[:, td.station_idx].cpu().numpy()


def signatures(Q):
    """Vecteur de signatures par station, mêmes définitions que reports/signatures_*.csv."""
    out = {k: [] for k in ["k_recession", "cm_freshet", "cv_debit", "ratio_pic_base"]}
    for s in range(Q.shape[1]):
        q = pd.Series(Q[:, s], index=tt)
        # récession de queue (juil-oct, segments >= 5 j sous la médiane)
        qmed = q.median(); dq = q.diff()
        dec = (dq < 0) & q.index.month.isin(range(7, 11)) & q.notna()
        ks, seg = [], []
        for day, isd in dec.items():
            if isd: seg.append(day)
            else:
                if len(seg) >= 5:
                    qs = q.loc[seg].values
                    if np.all(qs > 0) and qs[0] < qmed:
                        k = -np.polyfit(np.arange(len(qs)), np.log(qs), 1)[0]
                        if 0.001 < k < 0.5: ks.append(k)
                seg = []
        out["k_recession"].append(np.median(ks) if len(ks) >= 5 else np.nan)
        cms, rats = [], []
        for y in range(2001, 2025):
            w = q[f"{y}-03-01":f"{y}-06-30"].dropna()
            if len(w) < 90 or w.sum() <= 0: continue
            doy = w.index.dayofyear.values.astype(float)
            cms.append(float((doy * w.values).sum() / w.values.sum()))
            base = np.nanmedian(q[f"{y}-01-15":f"{y}-02-28"].values)
            if np.isfinite(base) and base > 0: rats.append(float(w.values.max() / base))
        out["cm_freshet"].append(np.median(cms) if len(cms) >= 5 else np.nan)
        out["ratio_pic_base"].append(np.median(rats) if len(rats) >= 5 else np.nan)
        out["cv_debit"].append(float(q.std() / q.mean()) if q.mean() > 0 else np.nan)
    return {k: np.array(v) for k, v in out.items()}


def shift(nm, d): return lambda sp: setattr(sp, nm, getattr(sp, nm) + d)
def scale(nms, s):
    def f(sp):
        for nm in nms: setattr(sp, nm, getattr(sp, nm) * s)
    return f

# perturbations RELATIVES modestes (régime linéaire) ; l'unité de la dérivée est
# indiquée pour que l'inversion soit interprétable
PERTS = [
    ("T_melt +1C", shift("T_melt", 1.0), "degC"),
    ("C_f x1.3", scale(["C_f"], 1.3), "log"),
    ("K_sat_1 x1.5", scale(["K_sat_1"], 1.5), "log"),
    ("K_sat_3 x1.5", scale(["K_sat_3"], 1.5), "log"),
    ("Z2Z3 x1.3", scale(["Z2", "Z3"], 1.3), "log"),
    ("k_gw x1.5", scale(["k_gw"], 1.5), "log"),
    ("krec x2", scale(["krec"], 2.0), "log"),
    ("K_c x1.15", scale(["K_c"], 1.15), "log"),
    ("K_musk x1.5", scale(["K_musk_hours"], 1.5), "log"),
]
S0 = signatures(sim(None))
print(f"=== jacobien {REG} | {os.path.basename(CKPT)} | {S0['cm_freshet'].size} stations ===", flush=True)
print("base : " + " | ".join(f"{k} {np.nanmedian(v):.3f}" for k, v in S0.items()), flush=True)
rows = []
for nm, fn, unit in PERTS:
    S = signatures(sim(fn))
    d = {k: float(np.nanmedian(S[k] - S0[k])) for k in S0}
    rel = {k: float(np.nanmedian((S[k] - S0[k]) / np.abs(S0[k]))) for k in S0}
    rows.append(dict(perturbation=nm, unite=unit,
                     **{f"d_{k}": round(d[k], 4) for k in d},
                     **{f"rel_{k}": round(rel[k], 4) for k in rel}))
    print(f"  {nm:14s} -> " + " | ".join(f"{k} {rel[k]*100:+6.1f}%" for k in S0), flush=True)
pd.DataFrame(rows).to_csv(f"reports/jacobien_{REG}.csv", index=False)
print(f"-> reports/jacobien_{REG}.csv")
