"""MEILLEUR CANDIDAT CUMULATIF sur OUTV, en inférence pure : noyau HGM du versant
(cache .hgm d'Hydrotel, +0.027 seul) + structure texturale Saxton-Rawls a mi-intensité
(+0.065 seule, vit dans la colonne donc hors banc rapide). Les deux corrections sont
orthogonales par construction : l'une étale DANS LE TEMPS, l'autre redistribue DANS
L'ESPACE la génération.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/combo_final.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hgm_loader import lire_hgm
from meandre.data.pedotransfert import saxton_rawls
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
CK = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
      "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PF = float(os.environ.get("PEDO_F", "0.5"))
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))

r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]
ck = f".runs/quebec/checkpoints/{CK[REG]}.pt"
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
    use_latent_codes=a_des_latents(ck, n), latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=True).to(DEVICE)
m.load(ck); m.eval(); m.vertical_column.etp_channel = 6

# pedotransfert : motif spatial normalise a mediane 1, mi-intensite
raw = pd.read_parquet("D:/meandre-data/quebec/territorial-raw-QC.parquet")
raw = raw[raw.region == REG]
assert len(raw) == n
p = saxton_rawls(raw.f_sand.values, raw.f_clay.values)
T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=DEVICE)
MOT = {"K_sat": T(p["k_sat"] / np.median(p["k_sat"])),
       "porosity": T(p["theta_s"] / np.median(p["theta_s"])),
       "theta_fc": T(p["theta_fc"] / np.median(p["theta_fc"])),
       "theta_wp": T(p["theta_wp"] / np.median(p["theta_wp"]))}
_orig = m.spatial_encoder.forward
def fwd(*a, _o=_orig, **k):
    sp = _o(*a, **k)
    for base, mot in MOT.items():
        fac = 1.0 + PF * (mot - 1.0)
        for i in (1, 2, 3):
            nm = f"{base}_{i}"
            if hasattr(sp, nm):
                setattr(sp, nm, getattr(sp, nm) * fac)
    return sp
m.spatial_encoder.forward = fwd

# noyau HGM
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, r["node_ids"]), device=DEVICE))

demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
    * AD.get(REG, {}).get("debias_et", 1.0)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
with torch.no_grad():
    Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                      graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                      withdrawals=td.withdrawals, day_of_year=td.day_of_year)
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
ks, rs, bs, gs = [], [], [], []
for s in range(Qs.shape[1]):
    o, si = qo[:, s], Qs[:, s]
    v = np.isfinite(o) & np.isfinite(si)
    if v.sum() < 60: continue
    rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean() / o[v].mean()
    g = (si[v].std() / si[v].mean()) / (o[v].std() / o[v].mean())
    ks.append(1 - np.sqrt((rr - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))
    rs.append(rr); bs.append(b); gs.append(g)
print(f"[{REG}] HGM + pedotransfert({PF}) : KGE {np.median(ks):.4f} | r {np.median(rs):.3f} "
      f"| beta {np.median(bs):.3f} | gamma {np.median(gs):.3f} (n={len(ks)})", flush=True)
