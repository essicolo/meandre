"""A/B PÉDOTRANSFERT en inférence pure. Saxton & Rawls (2006) relie la texture (déjà dans
les attributs, sable 0.34 en Abitibi contre 0.92 sur la Côte-Nord) aux propriétés
hydrauliques, alors que `init_from_literature` applique un unique loam moyen à toute la
province. On n'importe PAS les niveaux : la conductivité de Saxton-Rawls est une valeur de
matrice au point, 40-80x au-dessus de la conductivité effective journalière du modèle, et
cette dernière a été MESURÉE (recalage K_sat_1 = 0.04). On importe seulement la STRUCTURE
SPATIALE, en préservant la médiane de chaque champ.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/pedo_ab.py gasp mont sagu outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.pedotransfert import saxton_rawls
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))
RAW = pd.read_parquet("D:/meandre-data/quebec/territorial-raw-QC.parquet")
# force du transfert : 0 = aucun, 1 = structure pédotransfert complète
FORCES = [float(x) for x in os.environ.get("PEDO_F", "0.5,1.0").split(",")]
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
    _orig = m.spatial_encoder.forward

    rw = RAW[RAW.region == REG]
    assert len(rw) == n, f"{REG}: {len(rw)} vs {n}"
    pd_ = saxton_rawls(rw.f_sand.values, rw.f_clay.values)
    T = lambda x: torch.tensor(np.asarray(x, dtype=np.float32), device=DEVICE)
    # motifs normalisés à médiane 1 : on garde le NIVEAU mesuré du modèle
    MOT = {"K_sat": T(pd_["k_sat"] / np.median(pd_["k_sat"])),
           "porosity": T(pd_["theta_s"] / np.median(pd_["theta_s"])),
           "theta_fc": T(pd_["theta_fc"] / np.median(pd_["theta_fc"])),
           "theta_wp": T(pd_["theta_wp"] / np.median(pd_["theta_wp"]))}
    print(f"[{REG}] motifs (q10-q90) : K_sat {float(MOT['K_sat'].quantile(.1)):.2f}-{float(MOT['K_sat'].quantile(.9)):.2f} "
          f"| theta_fc {float(MOT['theta_fc'].quantile(.1)):.2f}-{float(MOT['theta_fc'].quantile(.9)):.2f}", flush=True)

    # ancrage d'exutoire optionnel, pour mesurer la COMBINAISON des deux lois
    _lac_fac = None
    if os.environ.get("PEDO_LAC", "0") == "1":
        _A = r["territorial"].get_physical("area_km2_local").to(DEVICE)
        _Al = torch.clamp(_A * T(np.clip(rw["lake_fraction"].values, 0, 1)), min=1e-3)
        _lac_fac = torch.clamp(float(os.environ.get("PEDO_LAC_AREF", "20")) / _Al, max=1.0)
        _o_lp = m.spatial_encoder.lake_params
        def _lp(*a, _o=_o_lp, _f=_lac_fac, **k):
            kk, bb = _o(*a, **k)
            return torch.clamp(kk * _f, 1e-6, 1e-2), bb
        m.spatial_encoder.lake_params = _lp
        print(f"[{REG}] ancrage d'exutoire actif (A_ref=20)", flush=True)

    def essai(force):
        def fwd(*a, _o=_orig, _f=force, **k):
            sp = _o(*a, **k)
            if _f > 0:
                for base, mot in MOT.items():
                    fac = 1.0 + _f * (mot - 1.0)
                    for i in (1, 2, 3):
                        nm = f"{base}_{i}"
                        if hasattr(sp, nm):
                            setattr(sp, nm, getattr(sp, nm) * fac)
            return sp
        m.spatial_encoder.forward = fwd if force > 0 else _orig
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

    base = essai(0.0)
    print(f"[{REG}] reference {base:.4f}", flush=True)
    for fo in FORCES:
        v = essai(fo)
        rows.append(dict(region=REG, force=fo, kge=round(v, 4), delta=round(v-base, 4)))
        print(f"[{REG}] force {fo:<4g} KGE {v:.4f} ({v-base:+.4f})", flush=True)
    m.spatial_encoder.forward = _orig
    del m; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv("reports/pedo_ab.csv", index=False)
