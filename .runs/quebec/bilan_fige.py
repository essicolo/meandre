"""BILAN DE MASSE de la colonne FIGÉE (fidélité) : où disparaît la moitié de l'eau ?
Constat v1 : états du sol à 0.92-0.995 d'Hydrotel, pluie équivalente (+2 %), mais apport
latéral à 13 % en août et volume réseau à 58 %. Ici : mêmes réglages figés, et on SAUVE
les séries (ETP, ETR, couvert nival, apport latéral) pour le bilan et la confrontation
avec la réexécution instrumentée d'Hydrotel (sorties ETP/ETR_TOTAL/THETA/APPORT).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/bilan_fige.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hydrotel_calib import load_calibrated_soil, load_linacre_nodes, load_melt_nodes
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])

m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    use_latent_codes=False, latent_mode="additive", spatial_melt=False,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=False).to(DEVICE)
m.eval(); m.spatial_encoder.init_from_literature({})
m.vertical_column.set_calibrated_soil(load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE))
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))

with torch.no_grad():
    _, _, diag = m.simulate(forcing=td.forcing[:, :, :6], initial_state=HydroState.zeros(n, device=DEVICE),
                            graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                            withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                            return_diagnostics=True)
np.savez_compressed(f"D:/meandre-data/quebec/bilan_fige_{REG}.npz",
                    etp=diag.etp.cpu().numpy().astype(np.float32),
                    etr=diag.etr.cpu().numpy().astype(np.float32),
                    swe=diag.swe.cpu().numpy().astype(np.float32),
                    lateral_mm=diag.lateral_mm.cpu().numpy().astype(np.float32),
                    apport=diag.snowmelt.cpu().numpy().astype(np.float32),
                    theta1=diag.theta1.cpu().numpy().astype(np.float32),
                    theta2=diag.theta2.cpu().numpy().astype(np.float32),
                    theta3=diag.theta3.cpu().numpy().astype(np.float32))

sel = (tt >= "2021-01-01") & (tt <= "2024-12-31")
P = float(td.forcing[torch.tensor(np.asarray(sel), device=DEVICE), :, 0].mean()) * 365.25
etp = float(diag.etp[np.asarray(sel)].mean()) * 365.25
etr = float(diag.etr[np.asarray(sel)].mean()) * 365.25
lat = float(diag.lateral_mm[np.asarray(sel)].mean()) * 365.25
print(f"=== BILAN colonne figée {REG} (2021-2024, mm/an, médane spatiale implicite=moyenne) ===")
print(f"P {P:.0f} | ETP {etp:.0f} | ETR {etr:.0f} | apport latéral {lat:.0f} | "
      f"résidu P-ETR-latéral {P - etr - lat:+.0f}")
print(f"coefficient d'écoulement : {lat / max(P, 1e-9):.3f} (attendu Hydrotel ~0.5-0.6)")
app = float(diag.snowmelt[np.asarray(sel)].mean()) * 365.25
print(f"BILAN NEIGE : P {P:.0f} -> apport au sol {app:.0f} (fuite neige {P - app:+.0f} mm/an)")
print(f"BILAN SOL   : apport {app:.0f} -> ETR {etr:.0f} + latéral {lat:.0f} (fuite sol {app - etr - lat:+.0f} mm/an)")
