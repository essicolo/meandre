"""PLAFOND DU MODÈLE : KGE sur la période d'ENTRAÎNEMENT (2000-2018), donc capacité
d'ajustement et non généralisation. Compare le champion (paramètres contraints par le
réseau spatial + prior) au modèle de capacité (décalage LIBRE par nœud sur les 37
paramètres, sans prior ni régularisation). Si le second ne fait pas mieux en
CORRÉLATION, la limite est dans la donnée et aucune architecture n'y changera rien.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-none python .runs/quebec/plafond.py gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand

REG = (sys.argv[1] if len(sys.argv) > 1 else "gasp").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("JOINT_FX_SUFFIX", "-none")
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
    * AD.get(REG, {}).get("debias_et", 1.0)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
qo = td.q_obs.cpu().numpy()[:len(tt)]
FEN = {"entrainement": ("2001-01-01", "2018-12-31"), "validation": ("2019-01-01", "2021-12-31"),
       "tenu": ("2022-01-01", "2024-12-31")}

def stats(Qa, d0, d1):
    s = np.asarray((tt >= d0) & (tt <= d1))
    o_, si_ = qo[s], Qa[s]
    ks, rs = [], []
    for i in range(si_.shape[1]):
        o, si = o_[:, i], si_[:, i]
        v = np.isfinite(o) & np.isfinite(si)
        if v.sum() < 60: continue
        rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
        g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
        ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2)); rs.append(rr)
    return float(np.median(ks)), float(np.median(rs))

for nom, ck in [("champion (contraint)", "best-gasp-etl-ds"), ("capacité (libre)", f"best-{REG}-etl-capacite")]:
    p = f".runs/quebec/checkpoints/{ck}.pt"
    if not os.path.exists(p):
        print(f"{nom}: {p} absent"); continue
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
        use_latent_codes=True, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(p); m.eval(); m.vertical_column.etp_channel = 6
    with torch.no_grad():
        Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
    Qa = Q[:, td.station_idx].cpu().numpy()
    out = " | ".join(f"{f}: KGE {stats(Qa,*d)[0]:.4f} r {stats(Qa,*d)[1]:.3f}" for f, d in FEN.items())
    print(f"{nom:22s} {out}", flush=True)
    del m, Q; torch.cuda.empty_cache()
