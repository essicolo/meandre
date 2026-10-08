"""SENSIBILITÉ des paramètres de lac, en inférence pure. La tête de lac est restée à son
initialisation dans les 6 régions (k_lake 1e-4, beta 1.5, dispersion 1-5 % pour des bornes
sur 4 ordres de grandeur) et le déficit contre Hydrotel est lacustre. Avant de lui donner
un LR dédié et de réentraîner, on vérifie que ces paramètres DÉPLACENT le score : un LR
plus élevé sur un levier inerte ne sert à rien (règle Essi 2026-07-24).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/lacs_sweep.py outv slno
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
# Un facteur uniforme ne teste PAS l'hypothese : elle dit qu'un etang et le lac
# Saint-Jean ne devraient pas avoir le meme coefficient. La physique donne la forme :
# la loi implementee est Q = k*(S/A)^beta*A, un exutoire en seuil donne Q = C*L*h^1.5 ;
# en egalant avec beta=1.5 il vient k = C*L/A, et avec L proportionnel a la racine de A,
# k doit decroitre comme A^-0.5. Aujourd'hui k est constant : les grands lacs ont un
# exutoire proportionnellement bien trop large. ALPHA teste cette differenciation.
ESSAIS = [("reference", 1.0, 0.0), ("k x10", 10.0, 0.0), ("k x100", 100.0, 0.0),
          ("k /10", 0.1, 0.0), ("beta +0.5", 1.0, 0.5), ("beta -0.4", 1.0, -0.4)]
ALPHA = [float(x) for x in os.environ.get("LACS_ALPHA", "0.5,1.0").split(",") if x]
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
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
    _orig = m.spatial_encoder.lake_params

    def essai(fk, db):
        def lp(*a, _o=_orig, **k):
            kk, bb = _o(*a, **k)
            return torch.clamp(kk * fk, 1e-6, 1e-2), torch.clamp(bb + db, 1.0, 2.5)
        m.spatial_encoder.lake_params = lp
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s in range(Qs.shape[1]):
            o, si = qo[:, s], Qs[:, s]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        del Q; torch.cuda.empty_cache()
        return float(np.median(ks))

    # differenciation par la taille : k_i = k0 * (A_ref/A_i)^alpha
    _aire = r["territorial"].get_physical("area_km2_local")
    _aire = _aire.to(DEVICE) if _aire is not None else None
    _lac = td.graph.is_lake.bool()

    def essai_alpha(alpha):
        A = torch.clamp(_aire, min=0.1)
        Aref = A[_lac].median() if _lac.any() else A.median()
        fac = (Aref / A) ** alpha
        def lp(*a, _o=_orig, _f=fac, **k):
            kk, bb = _o(*a, **k)
            return torch.clamp(kk * _f, 1e-6, 1e-2), bb
        m.spatial_encoder.lake_params = lp
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                              withdrawals=td.withdrawals, day_of_year=td.day_of_year)
        Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
        ks = []
        for s2 in range(Qs.shape[1]):
            o, si = qo[:, s2], Qs[:, s2]
            v = np.isfinite(o) & np.isfinite(si)
            if v.sum() < 60: continue
            rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
            g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
            ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
        del Q; torch.cuda.empty_cache()
        return float(np.median(ks))

    base = None
    for nom, fk, db in ESSAIS:
        v = essai(fk, db)
        if base is None: base = v
        rows.append(dict(region=REG, essai=nom, kge=round(v, 4), delta=round(v - base, 4)))
        print(f"[{REG}] {nom:10s} KGE {v:.4f} ({v-base:+.4f})", flush=True)
    if _aire is not None:
        for al in ALPHA:
            v = essai_alpha(al)
            rows.append(dict(region=REG, essai=f"k ~ A^-{al}", kge=round(v, 4), delta=round(v - base, 4)))
            print(f"[{REG}] k ~ A^-{al:<4} KGE {v:.4f} ({v-base:+.4f})", flush=True)
    m.spatial_encoder.lake_params = _orig
    del m; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv("reports/lacs_sweep.csv", index=False)
