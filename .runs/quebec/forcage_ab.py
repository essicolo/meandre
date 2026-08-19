"""A/B DE FORÇAGE en inférence pure : le champion régional, inchangé, évalué sur chaque
variante de forçage disponible. Répond à « corriger CaSR en vaut-il la peine ? ».
RÉSERVE : les champions ont été ENTRAÎNÉS sur -budyko, donc le test avantage cette
variante. Un écart en faveur du brut serait d'autant plus significatif ; un écart en
faveur de -budyko doit être confirmé par un réentraînement avant d'être cru.

  PYTHONIOENCODING=utf-8 python .runs/quebec/forcage_ab.py gasp sagu mont
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, importlib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState

VARIANTES = [("brut", "-none"), ("budyko", "-budyko"), ("hyb", "-hyb"), ("lin", "-lin")]
CKPTS = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
         "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
rows = []

for REG in [a.lower() for a in sys.argv[1:]]:
    ck = f".runs/quebec/checkpoints/{CKPTS[REG]}.pt"
    for nom, sfx in VARIANTES:
        os.environ["JOINT_FX_SUFFIX"] = sfx
        import joint_data; importlib.reload(joint_data)
        from et_module import compute_demand
        try:
            r = joint_data.load_region(REG, dict(cfg["loss"]), device=DEVICE)
        except FileNotFoundError:
            print(f"[{REG}] {nom}: fichier absent", flush=True); continue
        td = r["train_data"]; n = r["n_nodes"]
        ds = AD.get(REG, {}).get("debias_et", 1.0)
        demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) * ds
        f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
        m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
            use_temporal=False, use_residual=False, use_travel_time_attn=False,
            use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
            column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
            use_latent_codes=False, latent_mode="additive", spatial_melt=True,
            routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
            use_aquifer=True).to(DEVICE)
        m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
        hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
        qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks, bs, gs, rs = [], [], [], []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2)); rs.append(rr); bs.append(b); gs.append(g)
        P = td.forcing[:, :, 0].mean().item() * 365.25
        rows.append(dict(region=REG, forcage=nom, kge=round(float(np.median(ks)), 4),
                         r=round(float(np.median(rs)), 3), beta=round(float(np.median(bs)), 3),
                         gamma=round(float(np.median(gs)), 3), P_mm_an=round(P, 0), n=len(ks)))
        print(f"[{REG}] {nom:7s} KGE {np.median(ks):.4f} | r {np.median(rs):.3f} "
              f"| beta {np.median(bs):.3f} | gamma {np.median(gs):.3f} | P {P:.0f} mm/an", flush=True)
        del m, Q, f7, demand; torch.cuda.empty_cache()

df = pd.DataFrame(rows); df.to_csv("reports/forcage_ab.csv", index=False)
print(df.pivot(index="region", columns="forcage", values="kge").to_string())
