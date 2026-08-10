"""BANC DE ROUTAGE SUR PRODUCTION EN CACHE (accélération ~x20, demande d'Essi).

Tout ce qu'on teste en ce moment (noyau HGM du versant, lois de lac, Muskingum) est en
AVAL de la colonne verticale, qui coûte ~20 min par simulation. Ici : la colonne est
simulée UNE fois et sa production latérale (mm/j par nœud) est mise en cache ; chaque
variante ne rejoue ensuite que le routage (~1-2 min). Mesure double : KGE aux jauges
(tenu de côté) et r quotidien contre Hydrotel (réseau, têtes <50 km², lacs).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/banc_routage.py outv ref hgm trl hgm+trl
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, json, numpy as np, pandas as pd, torch, xarray as xr
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.temporal.ring_buffer import OutflowRingBuffer
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region
from et_module import compute_demand
from ckpt_util import a_des_latents

REG = sys.argv[1].lower()
MODES = sys.argv[2:] or ["ref"]
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
CK = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
      "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"
CACHE = f"D:/meandre-data/quebec/cache_lateral_{REG}.npz"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))

r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
ck = f".runs/quebec/checkpoints/{CK[REG]}.pt"
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
    use_latent_codes=a_des_latents(ck, n), latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=True).to(DEVICE)
m.load(ck); m.eval(); m.vertical_column.etp_channel = 6
m.vertical_column.compile_column = False

_cache_ok = os.path.exists(CACHE) and "q_sim" in np.load(CACHE).files
if not _cache_ok:
    print("[cache] simulation unique de la colonne (une vingtaine de minutes)...", flush=True)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords, r["territorial"], DEVICE) \
        * AD.get(REG, {}).get("debias_et", 1.0)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    with torch.no_grad():
        Qref, _, diag = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                                   graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                                   withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                                   return_diagnostics=True)
    np.savez_compressed(CACHE, lateral_mm=diag.lateral_mm.cpu().numpy().astype(np.float32),
                        q_sim=Qref.cpu().numpy().astype(np.float32))
    del Qref
    del diag; torch.cuda.empty_cache()
    print(f"[cache] écrit -> {CACHE}", flush=True)
_cz = np.load(CACHE)
lat_mm = torch.tensor(_cz["lateral_mm"], device=DEVICE)
Q_vrai = _cz["q_sim"]
T = lat_mm.shape[0]
print(f"[cache] production latérale {tuple(lat_mm.shape)}", flush=True)

with torch.no_grad():
    sp = m.spatial_encoder(td.node_coords, r["territorial"].data)
    m.routing._lake_k, m.routing._lake_beta = m.spatial_encoder.lake_params(
        td.node_coords, r["territorial"].data)
K_musk = sp.K_musk_hours * 3600.0; x_musk = sp.x_musk   # simulate convertit en SECONDES (model.py:371)
# BANC_K : force le temps de transfert (heures) pour tester la plage PHYSIQUE (~0.3 h,
# Manning sur troncon.trl) contre la plage du modèle (bornes [4,48], appris ~24 h).
if os.environ.get("BANC_K"):
    _kh = float(os.environ["BANC_K"])
    K_musk = torch.full_like(K_musk, _kh * 3600.0)
    print(f"[banc] K_musk FORCÉ à {_kh} h (physique Manning ~0.3 h ; appris ~23.7 h)", flush=True)
area_local = r["territorial"].get_physical("area_km2_local").to(DEVICE)
area_cum = (r["territorial"].area_km2_physical.to(DEVICE)
            if r["territorial"].area_km2_physical is not None else area_local)

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

surf, c_trl, k_exp = lire_lacs_trl()
lacm = td.graph.is_lake.bool().cpu().numpy()
couv = np.isfinite(surf) & lacm & (surf > 0)
couv_t = torch.tensor(couv, device=DEVICE)
kt = torch.tensor(np.nan_to_num(np.where(couv, c_trl / np.clip(surf * 1e6, 1, None), np.nan),
                                nan=1e-4), dtype=torch.float32, device=DEVICE)
bt = torch.tensor(np.nan_to_num(k_exp, nan=1.5), dtype=torch.float32, device=DEVICE)
k0, b0 = m.routing._lake_k.clone(), m.routing._lake_beta.clone()

dht = xr.open_dataset(f"{PROJ}/simulation/simulation/resultat/debit_aval.nc")
tht = pd.to_datetime(dht["time"].values); mh = (tht >= T0) & (tht <= T1)
QH = dht["debit_aval"].values[mh]; ids = dht["idtroncon"].values; dht.close()
pos = {int(i): j for j, i in enumerate(ids)}
QH = QH[:, [pos[int(i)] for i in node_ids]]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
hd = np.asarray((tt >= T0) & (tt <= T1))
qo = td.q_obs.cpu().numpy()[:len(tt)][hd]

import duckdb, collections
con = duckdb.connect(f"D:/meandre-data/quebec/{REG}.duckdb", read_only=True)
e = con.execute("select src, dst from edges").fetchdf(); con.close()
Acum = area_local.cpu().numpy().copy()
enfants = collections.defaultdict(list)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    enfants[int(s_)].append(int(d_))
for u in td.graph.topo_order.cpu().numpy():
    for v in enfants.get(int(u), []):
        Acum[v] += Acum[u]
tete = Acum < 50

def route(mode):
    hgm = None
    if "hgm" in mode:
        hgm = torch.tensor(K, device=DEVICE)
        hgm = hgm / hgm.sum(1, keepdim=True).clamp(min=1e-12)
    if "trl" in mode:
        m.routing._lake_k = torch.where(couv_t, kt, k0)
        m.routing._lake_beta = torch.where(couv_t, bt, b0)
        m.routing._lake_area_km2 = torch.tensor(np.where(couv, surf, 1.0),
                                                dtype=torch.float32, device=DEVICE)
    else:
        m.routing._lake_k, m.routing._lake_beta = k0, b0
        m.routing._lake_area_km2 = None
    Q_prev = torch.zeros(n, device=DEVICE)
    S_lake = torch.zeros(n, device=DEVICE)
    buf = OutflowRingBuffer(n, m.max_travel_time, device=DEVICE)
    m.routing._op_state = None
    file_q = torch.zeros(n, hgm.shape[1], device=DEVICE) if hgm is not None else None
    out = []
    with torch.no_grad():
        for t in range(T):
            lat = lat_mm[t]
            if hgm is not None:
                file_q = file_q + hgm * lat.unsqueeze(1)
                lat = file_q[:, 0]
                file_q = torch.cat([file_q[:, 1:], torch.zeros_like(file_q[:, :1])], dim=1)
            Q, S_lake, _ = m.routing(lat, td.graph, Q_prev, buf, td.withdrawals, t,
                                     K_musk, x_musk, lake_storage=S_lake,
                                     area_km2=area_cum, area_km2_local=area_local)
            buf.push(Q); Q_prev = Q; out.append(Q)
    return torch.stack(out).cpu().numpy()

# VALIDATION DU BANC : le mode ref doit reproduire la simulation complete.
_Qr = route("ref")
_d = np.abs(_Qr - Q_vrai) / np.clip(np.abs(Q_vrai), 1e-6, None)
print(f"[banc] ecart routage rejoue vs simulate : rel median {np.median(_d):.4f} | "
      f"rel p95 {np.quantile(_d, 0.95):.4f}", flush=True)
_dn = _d.mean(axis=0)
aval_lac = np.zeros(n, bool)
pile = [int(i) for i in np.flatnonzero(lacm)]
vus = set(pile)
while pile:
    u = pile.pop()
    for v in enfants.get(u, []):
        if v not in vus:
            vus.add(v); aval_lac[v] = True; pile.append(v)
for nom, msk in [("lacs", lacm), ("aval de lac", aval_lac & ~lacm),
                 ("rivieres hors influence", ~lacm & ~aval_lac)]:
    print(f"[banc]   erreur moyenne {nom:22s} : med {np.median(_dn[msk]):.4f} | "
          f"p95 {np.quantile(_dn[msk], 0.95):.4f} (n={int(msk.sum())})", flush=True)

for mode in MODES:
    QM = route(mode)[hd]
    Qs = QM[:, td.station_idx.cpu().numpy()]
    ks = []
    for s in range(Qs.shape[1]):
        o, si = qo[:, s], Qs[:, s]
        v = np.isfinite(o) & np.isfinite(si)
        if v.sum() < 60: continue
        rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean() / o[v].mean()
        g = (si[v].std() / si[v].mean()) / (o[v].std() / o[v].mean())
        ks.append(1 - np.sqrt((rr - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))
    nT = min(QM.shape[0], QH.shape[0])
    rs = np.full(n, np.nan)
    for j in range(n):
        h_, m_ = QH[:nT, j], QM[:nT, j]
        if h_.std() > 1e-9 and m_.std() > 1e-9:
            rs[j] = np.corrcoef(h_, m_)[0, 1]
    print(f"[{REG}] {mode:8s} KGE jauges {float(np.median(ks)):.4f} | r réseau {np.nanmedian(rs):.3f} "
          f"| r têtes {np.nanmedian(rs[tete]):.3f} | r lacs {np.nanmedian(rs[lacm]):.3f}", flush=True)
