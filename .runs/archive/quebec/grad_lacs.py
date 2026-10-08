"""Le détachement du stockage des lacs éteint-il le gradient des paramètres de lac ?
`message_passing.py` fait `lake_storage_new[lake_mask] = S_lake.detach()` à CHAQUE pas de
temps : le gradient ne voit que l'effet instantané de k_lake sur le débit du jour, jamais
son effet sur l'accumulation du stockage, qui est toute la physique d'un lac. Symptôme
observé le 4 août : k_lake vaut 1e-4 et beta 1.5 dans les 6 régions, soit exactement
l'initialisation, avec 1-5 % de dispersion pour des bornes couvrant 4 ordres de grandeur.

On compare la norme du gradient de la tête de lac AVEC et SANS le détachement, sur une
fenêtre courte, tous les autres paramètres étant identiques.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/grad_lacs.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch
import meandre.routing.message_passing as mp
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
NJ = int(os.environ.get("GRAD_JOURS", "400"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)[:NJ]
qo = td.q_obs[:NJ]

def essai(detacher):
    torch.manual_seed(0)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
        use_latent_codes=False, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(".runs/quebec/checkpoints/best-outv-etl-qc.pt" if REG == "outv"
           else f".runs/quebec/checkpoints/best-{REG}-etl-ds.pt")
    m.train(); m.vertical_column.etp_channel = 6
    mp.LACS_DETACHER = detacher
    Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                      graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                      withdrawals=td.withdrawals, day_of_year=td.day_of_year[:NJ])
    Qs = Q[:, td.station_idx]
    v = torch.isfinite(qo)
    perte = ((Qs[v] - qo[v]) ** 2).mean()
    m.zero_grad(); perte.backward()
    g_lac = m.spatial_encoder.fc_lake.weight.grad
    g_sol = m.spatial_encoder.fc_out.weight.grad
    return (float(perte), float(g_lac.norm()) if g_lac is not None else 0.0,
            float(g_sol.norm()) if g_sol is not None else 0.0)

for det in (True, False):
    p, gl, gs = essai(det)
    print(f"detach={det!s:5s} | perte {p:.4f} | ||grad tete de lac|| {gl:.4e} | "
          f"||grad tete principale|| {gs:.4e} | rapport {gl/max(gs,1e-12):.3e}", flush=True)
