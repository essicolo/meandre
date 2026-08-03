"""CALIBRATION DU TIMING DE FONTE PAR LE CHAMP KRIGE (dérivée des données, pas du TOML).
Par région : (1) simule le champion, (2) mesure le centre de masse du freshet SIMULE
par nœud, (3) le compare au champ OBSERVE krigé (126 stations, GP), (4) convertit
l'écart en décalage du seuil de fonte via la sensibilité mesurée au banc (3 j / °C),
(5) ré-évalue le KGE held-out avec et sans. Produit un parquet dT par nœud + provenance.

  PYTHONIOENCODING=utf-8 python .runs/quebec/freshet_calib.py mont slso sagu
"""
import os, sys, json
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand

SENS = float(os.environ.get("FRESHET_SENS", "3.0"))  # j de CM par +1 °C (banc gasp)
CKPT = os.environ.get("FRESHET_CKPT", ".runs/quebec/checkpoints/best-gasp-etl-ds.pt")
CLAMP = float(os.environ.get("FRESHET_CLAMP", "2.0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ.setdefault("JOINT_FX_SUFFIX", "-none")
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
CHAMP = pd.read_parquet("D:/meandre-data/quebec/champ_freshet_QC.parquet")
rows, dts = [], []

for REG in [a.lower() for a in sys.argv[1:]]:
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
    td = r["train_data"]; n = r["n_nodes"]
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False, use_frost_rankinen=True,
        column_theta_init_frac=0.9, param_mode="nerf", column_mode="hydrotel", et_mode="mcguinness",
        use_temperature=False, use_latent_codes=False, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m.load(CKPT); m.eval(); m.vertical_column.etp_channel = 6
    m.vertical_column.compile_column = False
    _orig = m.spatial_encoder.forward
    times = pd.to_datetime(r["times"]); t0 = td.train_slice.start
    tt = pd.DatetimeIndex(times[t0:])
    sel_y = [(y, np.asarray((tt >= f"{y}-03-01") & (tt <= f"{y}-06-30"))) for y in range(2001, 2025)]
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)][hd]

    def sim(applyfn):
        m.spatial_encoder.forward = _orig if applyfn is None else \
            (lambda *a, **k: (lambda sp: (applyfn(sp), sp)[1])(_orig(*a, **k)))
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        return Q

    def cm_nodes(Q):
        """CM du freshet par NŒUD (médiane inter-annuelle), sur GPU."""
        acc = []
        for y, s in sel_y:
            w = Q[torch.tensor(s, device=DEVICE)]
            doy = torch.tensor(tt[s].dayofyear.values, dtype=torch.float32, device=DEVICE)[:, None]
            tot = w.sum(0).clamp_min(1e-6)
            acc.append(((w * doy).sum(0) / tot))
        return torch.stack(acc).median(0).values

    def kge_med(Q):
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        return float(np.median(ks)) if ks else float("nan")

    Q0 = sim(None)
    cm_sim = cm_nodes(Q0)
    cf = CHAMP[CHAMP.region == REG].sort_values("node_idx")
    assert len(cf) == n, f"{REG}: {len(cf)} vs {n}"
    cm_obs = torch.tensor(cf.cm_freshet.values, dtype=torch.float32, device=DEVICE)
    dT = ((cm_obs - cm_sim) / SENS).clamp(-CLAMP, CLAMP)
    k0 = kge_med(Q0)
    del Q0; torch.cuda.empty_cache()
    Q1 = sim(lambda sp: setattr(sp, "T_melt", sp.T_melt + dT))
    k1 = kge_med(Q1)
    del Q1; torch.cuda.empty_cache()
    rows.append(dict(region=REG, n_nodes=n, cm_sim_med=round(float(cm_sim.median()), 1),
                     cm_champ_med=round(float(cm_obs.median()), 1),
                     dT_med=round(float(dT.median()), 2), dT_p10=round(float(dT.quantile(0.1)), 2),
                     dT_p90=round(float(dT.quantile(0.9)), 2),
                     kge_base=round(k0, 4), kge_freshet=round(k1, 4), gain=round(k1-k0, 4)))
    dts.append(pd.DataFrame(dict(region=REG, node_idx=np.arange(n), dT=dT.cpu().numpy())))
    print(f"[{REG}] CM sim j{float(cm_sim.median()):.1f} vs champ j{float(cm_obs.median()):.1f} "
          f"| dT med {float(dT.median()):+.2f} ({float(dT.quantile(0.1)):+.2f}/{float(dT.quantile(0.9)):+.2f}) "
          f"| KGE {k0:.4f} -> {k1:.4f} ({k1-k0:+.4f})", flush=True)
    del m; torch.cuda.empty_cache()

pd.concat(dts).to_parquet("D:/meandre-data/quebec/champ_freshet_dT.parquet")
df = pd.DataFrame(rows); df.to_csv("reports/freshet_calib.csv", index=False)
print(df.to_string(index=False))
json.dump(dict(sensibilite_j_par_C=SENS, clamp_C=CLAMP, checkpoint=os.path.basename(CKPT),
               champ="champ_freshet_QC.parquet (GP 126 stations, R2 blocs 0.62)"),
          open("reports/freshet_calib_provenance.json", "w"), indent=1)
