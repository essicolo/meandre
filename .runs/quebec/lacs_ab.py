"""A/B LACS en inférence pure. Le déficit contre l'ensemble Hydrotel est LACUSTRE :
-0.03 sous 5 % de nœuds-lacs en amont (86 stations), -0.22 au-dessus (46 stations), avec
une marche nette et un plateau — signature d'un défaut de traitement, pas d'un manque de
calibration. On neutralise le masque de lacs du graphe (les nœuds redeviennent des
tronçons de rivière ordinaires) et on mesure, sans réentraîner.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/lacs_ab.py outv slno sagu gasp
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
rows = []

for REG in [a.lower() for a in sys.argv[1:]]:
    ck = f".runs/quebec/checkpoints/{LOCAUX[REG]}.pt"
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
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
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
    lac0 = td.graph.is_lake.clone()
    print(f"[{REG}] {int(lac0.sum())} nœuds-lacs sur {n} ({100*float(lac0.float().mean()):.1f} %)", flush=True)

    def run(actifs):
        td.graph.is_lake = lac0 if actifs else torch.zeros_like(lac0)
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: ks.append(np.nan); continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        del Q; torch.cuda.empty_cache()
        return np.array(ks)

    on, off = run(True), run(False)
    td.graph.is_lake = lac0
    v = np.isfinite(on) & np.isfinite(off)
    rows.append(dict(region=REG, n=int(v.sum()), f_lac=round(float(lac0.float().mean()), 3),
                     kge_lacs_on=round(float(np.median(on[v])), 4),
                     kge_lacs_off=round(float(np.median(off[v])), 4),
                     gain=round(float(np.median(off[v] - on[v])), 4),
                     stations_ameliorees=int((off[v] > on[v]).sum())))
    print(f"[{REG}] lacs ON {np.median(on[v]):.4f} -> OFF {np.median(off[v]):.4f} "
          f"(gain médian {np.median(off[v]-on[v]):+.4f}, {int((off[v]>on[v]).sum())}/{int(v.sum())} stations améliorées)", flush=True)
    del m; torch.cuda.empty_cache()

df = pd.DataFrame(rows); df.to_csv("reports/lacs_ab.csv", index=False)
print(df.to_string(index=False))
