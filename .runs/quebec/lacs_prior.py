"""PRIOR PHYSIQUE D'EXUTOIRE, formulation affinée. Le banc du 5 août montre que
k_i = k0*(A_ref/A_i)^alpha gagne +0.027 sur OUTV (alpha=1, optimum net) mais rien sur SLNO
et un léger recul sur SAGU. Deux défauts de cette première version, corrigés ici :
  (1) la surface utilisée était celle du TRONÇON ; on prend la vraie surface lacustre
      (aire du tronçon x lake_fraction) ;
  (2) le facteur agissait dans les DEUX sens, or la réponse SATURE vers le haut (k x10 et
      k x100 donnent exactement le même KGE) : augmenter k pour les petits lacs ne fait
      rien et diluer l'effet. On ne réduit donc que les grands, sans jamais augmenter.
alpha = 1 correspond à une largeur d'exutoire INDÉPENDANTE de la taille du lac (fixée par
le chenal de sortie), ce que la mesure a désigné contre ma prédiction initiale de 0.5.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/lacs_prior.py outv slno sagu gasp
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

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
AREFS = [float(x) for x in os.environ.get("LACS_AREF", "1,5,20").split(",")]  # km2 de lac
ALPHA = float(os.environ.get("LACS_ALPHA", "1.0"))
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
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
    qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
    _orig = m.spatial_encoder.lake_params
    gp = r["territorial"].get_physical
    A = gp("area_km2_local").to(DEVICE)
    # lake_fraction n'est disponible que NORMALISEE dans le cache ; les valeurs
    # physiques sont dans le parquet d'attributs bruts provincial.
    _raw = pd.read_parquet(f"{_DATA_ROOT}/quebec/territorial-raw-QC.parquet")
    _raw = _raw[_raw.region == REG]
    assert len(_raw) == n, f"{REG}: {len(_raw)} lignes brutes pour {n} noeuds"
    fl = torch.tensor(_raw["lake_fraction"].values, dtype=torch.float32, device=DEVICE)
    A_lac = torch.clamp(A * torch.clamp(fl, 0.0, 1.0), min=1e-3)
    lac = td.graph.is_lake.bool()
    print(f"[{REG}] surface lacustre des nœuds-lacs : q10 {A_lac[lac].quantile(.1):.2f} "
          f"méd {A_lac[lac].median():.2f} q90 {A_lac[lac].quantile(.9):.2f} max {A_lac[lac].max():.1f} km²", flush=True)

    def essai(aref):
        fac = torch.clamp((aref / A_lac) ** ALPHA, max=1.0)   # ne réduit jamais en dessous, n'augmente jamais
        def lp(*a, _o=_orig, _f=fac, **k):
            kk, bb = _o(*a, **k)
            return torch.clamp(kk * _f, 1e-6, 1e-2), bb
        m.spatial_encoder.lake_params = lp if aref is not None else _orig
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

    m.spatial_encoder.lake_params = _orig
    base = essai(1e9)   # facteur = 1 partout
    print(f"[{REG}] reference {base:.4f}", flush=True)
    for aref in AREFS:
        v = essai(aref)
        rows.append(dict(region=REG, aref_km2=aref, alpha=ALPHA, kge=round(v, 4), delta=round(v-base, 4)))
        print(f"[{REG}] A_ref={aref:<5g} km²  KGE {v:.4f} ({v-base:+.4f})", flush=True)
    m.spatial_encoder.lake_params = _orig
    del m; torch.cuda.empty_cache()
pd.DataFrame(rows).to_csv("reports/lacs_prior.csv", index=False)
