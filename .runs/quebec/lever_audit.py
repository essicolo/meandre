"""AUDIT DE LEVIERS (principe Essi 2026-07-24 : « savoir que tous les leviers sont
à leur optimum avant de perdre des heures »). Sur un checkpoint ENTRAÎNÉ, perturbe
chaque groupe de paramètres EN INFÉRENCE (aucun réentraînement) et mesure la
réponse du KGE/r held-out. Tout levier qu'une perturbation AMÉLIORE n'est pas à
son optimum -> l'entraînement le laisse sur la table (prior faux / bound serré /
loss aveugle). Une simulation par perturbation (~1-3 min) ; carte complète en ~1h.

  JOINT_FX_SUFFIX=-hyb AUDIT_CKPT=.runs/quebec/checkpoints/best-gasp-etl.pt \
    python .runs/quebec/lever_audit.py gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand

REG = (sys.argv[1] if len(sys.argv) > 1 else "gasp").lower()
CKPT = os.environ.get("AUDIT_CKPT", f".runs/quebec/checkpoints/best-{REG}-etl.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True,
    column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
    use_temperature=False, use_latent_codes=True, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False).to(DEVICE)
m.load(CKPT); m.eval(); m.vertical_column.etp_channel = 6
_orig = m.spatial_encoder.forward

times = r["times"]; slm = (times >= "2022-01-01") & (times <= "2024-12-31")
sl = slm.values if hasattr(slm, "values") else np.asarray(slm)
t0 = td.train_slice.start
qo = td.q_obs[np.flatnonzero(sl)[0]-t0 : np.flatnonzero(sl)[-1]-t0+1].cpu().numpy()


def evaluate(applyfn):
    m.spatial_encoder.forward = _orig if applyfn is None else \
        (lambda *a, **k: (lambda sp: (applyfn(sp), sp)[1])(_orig(*a, **k)))
    with torch.no_grad():
        Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
    Qs = Q[torch.tensor(sl, device=DEVICE)][:, td.station_idx].cpu().numpy()
    ks, rs, bs, gs = [], [], [], []
    for s in range(Qs.shape[1]):
        o, si = qo[:, s], Qs[:, s]
        v = np.isfinite(o) & np.isfinite(si)
        if v.sum() < 60: continue
        rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean() / o[v].mean()
        g = (si[v].std() / si[v].mean()) / (o[v].std() / o[v].mean())
        ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2)); rs.append(rr); bs.append(b); gs.append(g)
    return float(np.median(ks)), float(np.median(rs)), float(np.median(bs)), float(np.median(gs))


def scale(names, s):
    def f(sp):
        for nm in names: setattr(sp, nm, getattr(sp, nm) * s)
    return f
def shift(names, d):
    def f(sp):
        for nm in names: setattr(sp, nm, getattr(sp, nm) + d)
    return f

LEVERS = [
    ("K_sat_1 ×0.5", scale(["K_sat_1"], 0.5)), ("K_sat_1 ×2", scale(["K_sat_1"], 2.0)),
    ("K_sat_3 ×0.5", scale(["K_sat_3"], 0.5)), ("K_sat_3 ×2", scale(["K_sat_3"], 2.0)),
    ("Z2+Z3 ×0.5", scale(["Z2", "Z3"], 0.5)), ("Z2+Z3 ×2", scale(["Z2", "Z3"], 2.0)),
    ("T_melt -2°C", shift(["T_melt"], -2.0)), ("T_melt +2°C", shift(["T_melt"], 2.0)),
    ("C_f ×0.6", scale(["C_f"], 0.6)), ("C_f ×1.6", scale(["C_f"], 1.6)),
    ("K_c ×0.8", scale(["K_c"], 0.8)), ("K_c ×1.2", scale(["K_c"], 1.2)),
    ("K_musk ×0.5", scale(["K_musk_hours"], 0.5)), ("K_musk ×2", scale(["K_musk_hours"], 2.0)),
    ("k_gw ×0.5", scale(["k_gw"], 0.5)), ("k_gw ×2", scale(["k_gw"], 2.0)),
    ("krec ×0.3", scale(["krec"], 0.3)), ("krec ×3", scale(["krec"], 3.0)),
]

k0, r0, b0, g0 = evaluate(None)
print(f"BASELINE {REG} ({os.path.basename(CKPT)}) : KGE {k0:.3f} | r {r0:.3f} | beta {b0:.2f} | gamma {g0:.2f}\n")
print(f"{'levier':>14} | {'dKGE':>7} | {'dr':>7} | verdict")
worst_gap = []
for name, fn in LEVERS:
    try:
        k, rr, b, g = evaluate(fn)
    except Exception as e:
        print(f"{name:>14} |   erreur : {type(e).__name__}"); continue
    dk, dr = k - k0, rr - r0
    verdict = "SOUS-OPTIMAL <<<" if dk > 0.01 else ("~optimum" if dk > -0.01 else "ok (dégrade)")
    if dk > 0.01: worst_gap.append((dk, name))
    print(f"{name:>14} | {dk:>+7.3f} | {dr:>+7.3f} | {verdict}", flush=True)
print("\nLeviers PAS à l'optimum (perturbation améliore le held-out) :")
for dk, name in sorted(worst_gap, reverse=True):
    print(f"  {name} : +{dk:.3f} KGE")
if not worst_gap:
    print("  aucun — tous les leviers audités sont à leur optimum local.")
