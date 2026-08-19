"""SELECTION DU CHAMPION A TRANSFERER, par région pauvre en jauges.
Mesure 2026-08-02 : la calibration LOCALE gagne ~+0.10 de KGE tenu de côté avec 30+
stations (slso), +0.03 avec 16 (outv), et PERD 0.22 avec 3 (abit 0.478 vs 0.697 en
transfert). Sous ~10 stations, la bonne question n'est donc pas comment entraîner
localement mais QUEL champion transférer. Sélection sur la VALIDATION 2019-2021
uniquement ; le tenu de côté 2022-2024 est reporté mais ne choisit jamais.

  PYTHONIOENCODING=utf-8 python .runs/quebec/choix_champion.py abit outm labi cnda
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand

CANDIDATS = ["gasp-etl-ds", "sagu-etl-ds", "mont-etl-ds", "slso-etl-canon",
             "slno-etl-canon", "outv-etl-qc"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("JOINT_FX_SUFFIX", "-none")
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
rows = []

for REG in [a.lower() for a in sys.argv[1:]]:
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
    td = r["train_data"]; n = r["n_nodes"]
    ds = AD.get(REG, {}).get("debias_et", 1.0)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) * ds
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    val = np.asarray((tt >= "2019-01-01") & (tt <= "2021-12-31"))
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)]

    def kge_med(Qs, o):
        ks = []
        for s in range(Qs.shape[1]):
            a, b_ = o[:, s], Qs[:, s]
            v = np.isfinite(a) & np.isfinite(b_)
            if v.sum() < 60: continue
            rr = np.corrcoef(a[v], b_[v])[0, 1]; be = b_[v].mean()/a[v].mean()
            ga = (b_[v].std()/b_[v].mean())/(a[v].std()/a[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (be-1)**2 + (ga-1)**2))
        return float(np.median(ks)) if ks else float("nan"), len(ks)

    for cand in CANDIDATS:
        p = f".runs/quebec/checkpoints/best-{cand}.pt"
        if not os.path.exists(p): continue
        m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
            use_temporal=False, use_residual=False, use_travel_time_attn=False,
            use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
            column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
            use_latent_codes=False, latent_mode="additive", spatial_melt=True,
            routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
            use_aquifer=True).to(DEVICE)
        try:
            m.load(p)
        except Exception as ex:
            print(f"[{REG}] {cand} : chargement impossible ({type(ex).__name__})", flush=True)
            del m; torch.cuda.empty_cache(); continue
        m.eval(); m.vertical_column.etp_channel = 6
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qa = Q[:, td.station_idx].cpu().numpy()
        kv, nv = kge_med(Qa[val], qo[val]); kh, _ = kge_med(Qa[hd], qo[hd])
        rows.append(dict(region=REG, champion=cand, n=nv, val=round(kv, 4), heldout=round(kh, 4)))
        print(f"[{REG}] {cand:16s} val {kv:.4f} | tenu {kh:.4f} (n={nv})", flush=True)
        del m, Q; torch.cuda.empty_cache()
    d = pd.DataFrame([x for x in rows if x["region"] == REG])
    if len(d):
        best = d.loc[d.val.idxmax()]
        print(f"  -> {REG} : {best.champion} (val {best.val:.4f}, tenu {best.heldout:.4f})", flush=True)

pd.DataFrame(rows).to_csv("reports/choix_champion.csv", index=False)
print("-> reports/choix_champion.csv")
