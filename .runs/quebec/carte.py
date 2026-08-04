"""CARTE PROVINCIALE : un champion LOCAL là où la région en a un (avec ses codes latents,
qui sont un effet aléatoire par nœud et n'existent que chez elle), sinon TRANSFERT du
champion global. Règle fixée a priori le 2 août : la calibration locale gagne ~+0.10 de
KGE au-dessus d'une dizaine de jauges et perd au-dessous ; sélectionner un champion sur
la validation avec 1-4 stations fait PIRE que ne pas sélectionner (0.627 vs 0.653).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/carte.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, importlib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from ckpt_util import a_des_latents

GLOBAL = "best-gasp-etl-ds"
LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
REGIONS = ["gasp", "sagu", "mont", "labi", "abit", "cnda", "cndb", "cndc", "cndd", "cnde",
           "outm", "outv", "slno", "slso", "vaud"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
rows = []
import joint_data
from et_module import compute_demand

for REG in REGIONS:
    ck = f".runs/quebec/checkpoints/{LOCAUX.get(REG, GLOBAL)}.pt"
    origine = "local" if REG in LOCAUX else "transfert"
    try:
        r = joint_data.load_region(REG, dict(cfg["loss"]), device=DEVICE)
    except Exception as ex:
        print(f"[{REG}] chargement impossible : {type(ex).__name__} {ex}", flush=True); continue
    td = r["train_data"]; n = r["n_nodes"]
    lat = a_des_latents(ck, n)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
        * AD.get(REG, {}).get("debias_et", 1.0)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
        use_latent_codes=lat, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
    m.vertical_column.compile_column = False
    with torch.no_grad():
        Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
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
        rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
        g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
        ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2)); rs.append(rr); bs.append(b); gs.append(g)
    med = float(np.median(ks)) if ks else float("nan")
    rows.append(dict(region=REG, origine=origine, champion=os.path.basename(ck)[:-3], latents=lat,
                     n=len(ks), kge=round(med, 4), r=round(float(np.median(rs)), 3) if rs else None,
                     beta=round(float(np.median(bs)), 3) if bs else None,
                     gamma=round(float(np.median(gs)), 3) if gs else None))
    print(f"[{REG}] {origine:9s} z_n={'oui' if lat else 'non'} | KGE {med:.4f} (n={len(ks)})", flush=True)
    del m, Q, f7, demand; torch.cuda.empty_cache()

df = pd.DataFrame(rows)
sfx = os.environ.get("JOINT_FX_SUFFIX", "-budyko")
df.to_csv(f"reports/carte{sfx}.csv", index=False)
print(f"\nmédiane {df.kge.median():.4f} | moyenne {df.kge.mean():.4f} | {len(df)} régions | forçage {sfx}")
print(df.to_string(index=False))
