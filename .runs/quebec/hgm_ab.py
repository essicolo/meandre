"""BANC COMBINÉ : hydrogramme géomorphologique d'Hydrotel (noyau .hgm) et lacs de
troncon.trl, en inférence pure sur le champion. Mesure double : KGE aux jauges (tenu de
côté 2022-2024) ET corrélation quotidienne contre Hydrotel sur TOUS les tronçons (dont
les têtes < 50 km², là où la décorrélation était massive : r 0.27-0.31).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/hgm_ab.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, json, numpy as np, pandas as pd, torch, xarray as xr
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
CK = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
      "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))

r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
K = lire_hgm(PROJ, node_ids)

def lire_lacs_trl():
    p = Path(f"{PROJ}/physitel/troncon.trl")
    lignes = [l.strip() for l in p.read_text(encoding="latin-1").splitlines() if l.strip()]
    d = {}
    for l in lignes[3:]:
        t = l.split()
        if int(t[1]) != 1:
            ptr = 4 + int(t[3])
            d[int(t[0])] = (float(t[ptr+1]), float(t[ptr+2]), float(t[ptr+3]))
    idx = {int(i): j for j, i in enumerate(node_ids)}
    surf = np.full(n, np.nan); c = np.full(n, np.nan); kk = np.full(n, np.nan)
    for tid, (s_, c_, k_) in d.items():
        if tid in idx:
            surf[idx[tid]], c[idx[tid]], kk[idx[tid]] = s_, c_, k_
    return surf, c, kk

surf, c_trl, k_trl_exp = lire_lacs_trl()
lac = td.graph.is_lake.bool().cpu().numpy()
couv = np.isfinite(surf) & lac & (surf > 0)
k_trl = np.where(couv, c_trl / np.clip(surf * 1e6, 1.0, None), np.nan)
couv_t = torch.tensor(couv, device=DEVICE)
kt = torch.tensor(np.nan_to_num(k_trl, nan=1e-4), dtype=torch.float32, device=DEVICE)
bt = torch.tensor(np.nan_to_num(k_trl_exp, nan=1.5), dtype=torch.float32, device=DEVICE)

dht = xr.open_dataset(f"{PROJ}/simulation/simulation/resultat/debit_aval.nc")
tht = pd.to_datetime(dht["time"].values); mht = (tht >= T0) & (tht <= T1)
QH = dht["debit_aval"].values[mht]; ids = dht["idtroncon"].values; dht.close()
pos = {int(i): j for j, i in enumerate(ids)}
QH = QH[:, [pos[int(i)] for i in node_ids]]

ck = f".runs/quebec/checkpoints/{CK[REG]}.pt"
lat_ok = a_des_latents(ck, n)
demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
    * AD.get(REG, {}).get("debias_et", 1.0)
f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
    use_latent_codes=lat_ok, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=True).to(DEVICE)
m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
hd = np.asarray((tt >= T0) & (tt <= T1))
qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
_orig = m.spatial_encoder.lake_params

import duckdb, collections
con = duckdb.connect(f"D:/meandre-data/quebec/{REG}.duckdb", read_only=True)
e = con.execute("select src, dst from edges").fetchdf(); con.close()
A = r["territorial"].get_physical("area_km2_local").cpu().numpy()
Acum = A.copy()
enfants = collections.defaultdict(list)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    enfants[int(s_)].append(int(d_))
for u in td.graph.topo_order.cpu().numpy():
    for v in enfants.get(int(u), []):
        Acum[v] += Acum[u]
tete = Acum < 50

def run(mode):
    m.set_hgm_kernel(K if "hgm" in mode else None)
    if "trl" in mode:
        m.set_lake_area(torch.tensor(np.where(couv, surf, 1.0), dtype=torch.float32))
        def lp(*a, _o=_orig, **kw):
            kk2, bb = _o(*a, **kw)
            return torch.where(couv_t, kt, kk2), torch.where(couv_t, bt, bb)
        m.spatial_encoder.lake_params = lp
    else:
        m.set_lake_area(None); m.spatial_encoder.lake_params = _orig
    with torch.no_grad():
        Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
    QM = Q[torch.tensor(hd, device=DEVICE)].cpu().numpy()
    Qs = QM[:, td.station_idx.cpu().numpy()]
    del Q; torch.cuda.empty_cache()
    ks = []
    for s in range(Qs.shape[1]):
        o, si = qo[:, s], Qs[:, s]
        v = np.isfinite(o) & np.isfinite(si)
        if v.sum() < 60: continue
        rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean()/o[v].mean()
        g = (si[v].std()/si[v].mean())/(o[v].std()/o[v].mean())
        ks.append(1 - np.sqrt((rr-1)**2 + (b-1)**2 + (g-1)**2))
    nT = min(QM.shape[0], QH.shape[0])
    rs = np.full(n, np.nan)
    for j in range(n):
        h_, m_ = QH[:nT, j], QM[:nT, j]
        if h_.std() > 1e-9 and m_.std() > 1e-9:
            rs[j] = np.corrcoef(h_, m_)[0, 1]
    return float(np.median(ks)), float(np.nanmedian(rs)), float(np.nanmedian(rs[tete])), float(np.nanmedian(rs[lac]))

print(f"[{REG}] noyau hgm : jour0 méd {float(np.median(K[:,0])):.2f} | lacs trl {int(couv.sum())}/{int(lac.sum())}", flush=True)
for mode in ("ref", "hgm", "hgm+trl"):
    kge, rall, rtete, rlac = run(mode)
    print(f"[{REG}] {mode:8s} KGE jauges {kge:.4f} | r réseau {rall:.3f} | r têtes<50km² {rtete:.3f} | r lacs {rlac:.3f}", flush=True)
