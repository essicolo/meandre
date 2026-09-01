"""DUEL station par station contre l'ENSEMBLE COMPLET des 6 calages Hydrotel.
Comparer à un seul membre (LN24HA) est trop favorable ou trop défavorable selon la
région : Hydrotel est un ensemble de 6 calages équifinaux. On rapporte donc, pour chaque
station, le KGE de méandre, celui de chaque membre, celui de la médiane d'ensemble et
celui du MEILLEUR membre. Held-out 2022-2024, jours communs uniquement.

Évaluation GROUPÉE (demande d'Essi, 3 août) : le comptage se fait sur les stations, pas
sur les régions ; une médiane régionale calculée sur une jauge ne pèse plus autant qu'une
calculée sur trente.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-none python .runs/quebec/duel_hydrotel.py gasp sagu mont slso slno outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, json, numpy as np, pandas as pd, torch, xarray as xr, duckdb
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_RQH_ROOT = _osp.environ.get("MEANDRE_RQH", "C:/Users/parse01/documents-locaux/rqh-local")
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

MEMBRES = ["LN24HA", "MG24HA", "MG24HI", "MG24HK", "MG24HQ", "MG24HS"]
PT = f"{_RQH_ROOT}/rqh_2026-04/data/06_posttraitement/posttraitement_{m}.zarr"
LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
T0, T1 = "2022-01-01", "2024-12-31"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))


def kge(qs, qo):
    m = np.isfinite(qs) & np.isfinite(qo)
    if m.sum() < 60: return np.nan
    qs, qo = qs[m], qo[m]
    if qo.std() < 1e-9 or qs.std() < 1e-9: return np.nan
    r = np.corrcoef(qs, qo)[0, 1]; b = qs.mean()/qo.mean()
    g = (qs.std()/qs.mean())/(qo.std()/qo.mean())
    return 1 - np.sqrt((r-1)**2 + (b-1)**2 + (g-1)**2)


ZS = {}
for m in MEMBRES:
    z = xr.open_zarr(PT.format(m=m))
    ZS[m] = (z, z["troncon_id"].values.astype(str), pd.to_datetime(z["time"].values))
print(f"[hydrotel] {len(ZS)} membres ouverts", flush=True)

rows = []
for REG in [a.lower() for a in sys.argv[1:]]:
    ck = f".runs/quebec/checkpoints/{LOCAUX[REG]}.pt"
    r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
    td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
    lat = a_des_latents(ck, n)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
        * AD.get(REG, {}).get("debias_et", 1.0)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m_ = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
        use_latent_codes=lat, latent_mode="additive", spatial_melt=True,
        routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
        use_aquifer=True).to(DEVICE)
    m_.load(ck); m_.eval(); m_.vertical_column.etp_channel = 6
    with torch.no_grad():
        Q, _ = m_.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                           graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                           withdrawals=td.withdrawals, day_of_year=td.day_of_year)
    tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    Qs = Q[:, td.station_idx].cpu().numpy()
    del m_, Q; torch.cuda.empty_cache()

    db = ".runs/slso/data/slso.duckdb" if REG == "slso" else f"{_DATA_ROOT}/quebec/{REG}.duckdb"
    con = duckdb.connect(db, read_only=True)
    st = con.execute("select station_id, node_idx from stations").fetchdf()
    idx_map = {int(v): i for i, v in enumerate(td.station_idx.cpu().numpy())}
    for _, s in st.iterrows():
        ni = int(s.node_idx)
        if ni not in idx_map: continue
        o = con.execute(f"select date, discharge from observations where station_id={s.station_id} order by date").fetchdf()
        if len(o) < 60: continue
        o["date"] = pd.to_datetime(o["date"])
        obs = o.set_index("date")["discharge"]
        tid = f"{REG.upper()}{int(node_ids[ni]):05d}"
        me = pd.Series(Qs[:, idx_map[ni]], index=tt)
        rec = dict(region=REG, station=int(s.station_id), troncon=tid)
        idx = pd.date_range(T0, T1, freq="D")
        obs_c = obs.reindex(idx)
        rec["meandre"] = kge(me.reindex(idx).values, obs_c.values)
        mem = {}
        for mm, (z, tids, ztime) in ZS.items():
            w = np.flatnonzero(tids == tid)
            if not len(w): continue
            sr = pd.Series(z["Dis"].values[int(w[0]), :], index=ztime).reindex(idx)
            mem[mm] = sr.values
            rec[mm] = kge(sr.values, obs_c.values)
        if mem:
            med = np.nanmedian(np.vstack(list(mem.values())), axis=0)
            rec["ens_med"] = kge(med, obs_c.values)
            rec["meilleur_membre"] = np.nanmax([rec.get(mm, np.nan) for mm in mem])
        rows.append(rec)
    con.close()
    print(f"[{REG}] {sum(1 for x in rows if x['region']==REG)} stations", flush=True)

df = pd.DataFrame(rows)
df.to_csv("reports/duel_hydrotel.csv", index=False)
d = df.dropna(subset=["meandre", "ens_med"])
print(f"\n=== {len(d)} stations comparables (held-out {T0} -> {T1}) ===")
print(f"méandre           médiane {d.meandre.median():.3f} | moyenne {d.meandre.mean():.3f}")
print(f"ensemble (médiane) médiane {d.ens_med.median():.3f} | moyenne {d.ens_med.mean():.3f}")
print(f"meilleur membre    médiane {d.meilleur_membre.median():.3f}")
print(f"\nméandre > médiane d'ensemble : {(d.meandre > d.ens_med).sum()}/{len(d)} stations "
      f"({(d.meandre > d.ens_med).mean()*100:.0f} %)")
print(f"méandre > MEILLEUR membre    : {(d.meandre > d.meilleur_membre).sum()}/{len(d)} stations "
      f"({(d.meandre > d.meilleur_membre).mean()*100:.0f} %)")
print("\npar région :")
print(d.groupby("region").agg(n=("station","size"), meandre=("meandre","median"),
      ens=("ens_med","median"), meilleur=("meilleur_membre","median")).round(3).to_string())
