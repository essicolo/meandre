"""BANC DE TIMING DE FONTE : le centre de masse du freshet SIMULÉ est-il actionnable ?
Mesure, en inférence pure sur un checkpoint entraîné, (a) le biais de timing simulé vs
observé aux stations, (b) la réponse en jours du CM à une perturbation de T_melt / C_f.
Sans réponse mesurable, le champ krigé du CM ne peut pas servir de contrainte.

  FRESHET_CKPT=.runs/quebec/checkpoints/best-gasp-etl-ds.pt python .runs/quebec/freshet_bench.py gasp
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
CKPT = os.environ.get("FRESHET_CKPT", ".runs/quebec/checkpoints/best-gasp-etl-ds.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("JOINT_FX_SUFFIX", "-none")
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
DS = float(os.environ.get("FRESHET_DS", "1.0"))
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) * DS
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True,
    column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
    use_temperature=False, use_latent_codes=False, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=True).to(DEVICE)
m.load(CKPT); m.eval(); m.vertical_column.etp_channel = 6
_orig = m.spatial_encoder.forward

times = pd.to_datetime(r["times"]); t0 = td.train_slice.start
qo_all = td.q_obs.cpu().numpy()


def cm_series(Q, tt):
    """CM du freshet (mars-juin) par station : médiane sur les années."""
    df_i = pd.DatetimeIndex(tt)
    out = []
    for s in range(Q.shape[1]):
        cms = []
        for y in range(2001, 2025):
            sel = (df_i >= f"{y}-03-01") & (df_i <= f"{y}-06-30")
            v = Q[sel, s]; doy = df_i[sel].dayofyear.values.astype(float)
            ok = np.isfinite(v) & (v >= 0)
            if ok.sum() < 90 or v[ok].sum() <= 0: continue
            cms.append(float((doy[ok]*v[ok]).sum()/v[ok].sum()))
        out.append(np.median(cms) if len(cms) >= 5 else np.nan)
    return np.array(out)


def run(applyfn):
    m.spatial_encoder.forward = _orig if applyfn is None else \
        (lambda *a, **k: (lambda sp: (applyfn(sp), sp)[1])(_orig(*a, **k)))
    with torch.no_grad():
        Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
    return Q[:, td.station_idx].cpu().numpy()


def shift(nm, d):
    return lambda sp: setattr(sp, nm, getattr(sp, nm) + d)
def scale(nm, s):
    return lambda sp: setattr(sp, nm, getattr(sp, nm) * s)


tt = times[t0:]
qo = qo_all[:len(tt)]
cm_obs = cm_series(qo, tt)
base = run(None); cm_base = cm_series(base, tt)
ok = np.isfinite(cm_obs) & np.isfinite(cm_base)
print(f"=== {REG} | {os.path.basename(CKPT)} | {ok.sum()} stations ===", flush=True)
print(f"CM observé  méd j{np.nanmedian(cm_obs[ok]):.1f} | CM simulé méd j{np.nanmedian(cm_base[ok]):.1f} "
      f"| BIAIS {np.nanmedian(cm_base[ok]-cm_obs[ok]):+.1f} j", flush=True)
import json
json.dump(dict(region=REG, ckpt=os.path.basename(CKPT), n=int(ok.sum()),
               cm_obs=round(float(np.nanmedian(cm_obs[ok])), 1),
               cm_sim=round(float(np.nanmedian(cm_base[ok])), 1),
               biais=round(float(np.nanmedian(cm_base[ok]-cm_obs[ok])), 1)),
          open(f"reports/freshet_biais_{REG}.json", "w"))
if os.environ.get("FRESHET_BASE_ONLY", "0") == "1":
    sys.exit(0)
for nm, fn in [("T_melt -1C", shift("T_melt", -1.0)), ("T_melt +1C", shift("T_melt", 1.0)),
               ("C_f x0.7", scale("C_f", 0.7)), ("C_f x1.4", scale("C_f", 1.4))]:
    cm = cm_series(run(fn), tt)
    d = np.nanmedian(cm[ok]-cm_base[ok])
    print(f"  {nm:14s} -> dCM {d:+5.2f} j", flush=True)
