"""FIDÉLITÉ TOTALE v2 (questions d'Essi) : colonne FIGÉE sur le calage Hydrotel + météo
du PROJET (Thiessen sur OUTV.nc, comme simulation.csv le déclare) + routage par le CLONE
de l'onde cinématique modifiée (hydrotel_clone/network_routing_torch, porté ligne à
ligne du C++ et validé sur Delisle) au lieu du Muskingum. Zéro entraînement.

Chaîne : colonne (sol bv3c complet + Linacre calé + fonte calée + noyau HGM) -> apport
latéral m³/s -> routeur fidèle (rivières onde cinématique, lacs Q = c·h^k de troncon.trl).
Comparé à debit_aval.nc tronçon par tronçon sur 2022-2024. Ce qui reste d'écart ne peut
venir que du Thiessen approximé (plus proche voisin) ou d'une divergence réelle de clone.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/fidelite2.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, numpy as np, pandas as pd, torch, xarray as xr
from scipy.spatial import cKDTree
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hydrotel_calib import load_calibrated_soil, load_linacre_nodes, load_melt_nodes
from meandre.data.hgm_loader import lire_hgm
from hydrotel_clone.network_routing_torch import route_network_torch
from joint_data import load_region

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_PLAT_ROOT = _osp.environ.get("MEANDRE_PLATFORMS", "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel")
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"{_PLAT_ROOT}/LN24HA/{REG.upper()}_LN24HA_2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))

r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])

# ── météo du PROJET : Thiessen (plus proche station) sur OUTV.nc, à partir de 2020 ──
dm = xr.open_dataset(f"{PROJ}/meteo/{REG.upper()}.nc")
tm = pd.to_datetime(dm["time"].values)
lat_col = 0 if 40 < float(td.node_coords[:, 0].mean()) < 62 else 1
ncrd = td.node_coords.cpu().numpy()
lat0 = float(ncrd[:, lat_col].mean())
def proj(lon, lat):
    return np.c_[np.asarray(lon) * 111.32 * np.cos(np.radians(lat0)), np.asarray(lat) * 110.57]
# x,y du fichier meteo : verifier si degres ou metres
sx, sy = dm["x"].values, dm["y"].values
if np.nanmax(np.abs(sx)) > 360:   # coordonnees projetees -> approx : normaliser par kd sur (x,y) directement
    st_xy = np.c_[sx, sy]
    raise SystemExit("meteo en coordonnees projetees : adapter la projection des noeuds")
st_xy = proj(sx, sy)
tree = cKDTree(st_xy)
_, jn = tree.query(proj(ncrd[:, 1 - lat_col], ncrd[:, lat_col]), k=1)
forc = td.forcing[:, :, :6].clone()
comm = tt.intersection(tm)
i_f = tt.get_indexer(comm); i_m = tm.get_indexer(comm)
for canal, var in [(0, "pr"), (1, "tasmin"), (2, "tasmax")]:
    V = dm[var].values          # (T_m, stations)
    forc[torch.tensor(i_f, device=DEVICE), :, canal] = torch.tensor(
        V[i_m][:, jn], dtype=torch.float32, device=DEVICE)
dm.close()
print(f"[meteo] Thiessen projet : {len(comm)} jours remplacés (dès {comm[0].date()})", flush=True)

# ── colonne figée ────────────────────────────────────────────────────────────
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    use_latent_codes=False, latent_mode="additive", spatial_melt=False,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=False).to(DEVICE)
m.eval(); m.spatial_encoder.init_from_literature({})
m.vertical_column.set_calibrated_soil(load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE))
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))
print("[colonne] sol bv3c COMPLET + Linacre calé + fonte calée + noyau HGM", flush=True)

with torch.no_grad():
    _, _, diag = m.simulate(forcing=forc, initial_state=HydroState.zeros(n, device=DEVICE),
                            graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                            withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                            return_diagnostics=True)
apl = diag.q_lateral            # (T, n) m3/s, POST-noyau HGM
del diag; torch.cuda.empty_cache()

# ── réseau pour le routeur fidèle : géométrie troncon.trl + coef d'optimisation ──
lignes = [l.strip() for l in (Path(PROJ) / "physitel" / "troncon.trl").read_text(encoding="latin-1").splitlines() if l.strip()]
geo = {}
for l in lignes[3:]:
    t = l.split()
    tid = int(t[0]); typ = int(t[1])
    if typ == 1:
        geo[tid] = dict(riv=True, lng=float(t[4]), lrg=float(t[5]), pte=max(float(t[6]), 0.0025))
    else:
        ptr = 4 + int(t[3])
        geo[tid] = dict(riv=False, surf=float(t[ptr + 1]) * 1e6, c=float(t[ptr + 2]), k=float(t[ptr + 3]))
# coefficients d'optimisation rugosité/largeur (onde_cinematique_modifiee.csv)
rugo = {}; larg = {}
ocm = Path(PROJ) / "simulation/simulation/onde_cinematique_modifiee.csv"
for l in ocm.read_text(encoding="latin-1").splitlines():
    t = [x.strip() for x in l.split(";")]
    if len(t) >= 3 and t[0].isdigit():
        rugo[int(t[0])] = float(t[1]); larg[int(t[0])] = float(t[2])
MAN_DEF = 0.04
isr = np.zeros(n, bool); lng = np.ones(n); lrg = np.ones(n); pte = np.full(n, 0.0025)
man = np.full(n, MAN_DEF); surf = np.ones(n); c_l = np.full(n, 1.0); k_l = np.full(n, 1.5)
for j, nid in enumerate(node_ids):
    g = geo.get(int(nid))
    if g is None: continue
    if g["riv"]:
        isr[j] = True; lng[j] = max(g["lng"], 1.0)
        lrg[j] = max(g["lrg"] * larg.get(int(nid), 1.0), 0.1)
        pte[j] = g["pte"]; man[j] = max(MAN_DEF * rugo.get(int(nid), 1.0), 0.02)
    else:
        surf[j] = max(g["surf"], 1.0); c_l[j] = g["c"]; k_l[j] = g["k"]
P = {k2: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k2, v in
     dict(lng=lng, lrg=lrg, pte=pte, man=man, surface_m2=surf, c=c_l, k=k_l).items()}
P["is_river"] = torch.tensor(isr, device=DEVICE)

# aval et niveaux topologiques
import duckdb, collections
con = duckdb.connect(f"{_DATA_ROOT}/quebec/{REG}.duckdb", read_only=True)
e = con.execute("select src, dst from edges").fetchdf(); con.close()
down = np.full(n, -1, dtype=np.int64)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    down[int(s_)] = int(d_)
niveau = np.zeros(n, dtype=np.int64)
for u in td.graph.topo_order.cpu().numpy():
    d_ = down[int(u)]
    if d_ >= 0:
        niveau[d_] = max(niveau[d_], niveau[int(u)] + 1)
groups = [torch.tensor(np.flatnonzero(niveau == lv), dtype=torch.long, device=DEVICE)
          for lv in range(int(niveau.max()) + 1)]
downstream = torch.tensor(down, device=DEVICE)
print(f"[reseau] {int(isr.sum())} rivières, {int((~isr).sum())} lacs, {len(groups)} niveaux", flush=True)

# routage fidèle sur 2021-2024 (1 an de chauffe routeur)
i21 = int(np.flatnonzero(tt >= "2021-01-01")[0])
with torch.no_grad():
    QF = route_network_torch(P, downstream, groups, apl[i21:], pdts=86400)
QF = QF.cpu().numpy()
ttr = tt[i21:]
hdr = np.asarray((ttr >= T0) & (ttr <= T1))
QM = QF[hdr]

dht = xr.open_dataset(f"{PROJ}/simulation/simulation/resultat/debit_aval.nc")
tht = pd.to_datetime(dht["time"].values); mh = (tht >= T0) & (tht <= T1)
QH = dht["debit_aval"].values[mh]; ids = dht["idtroncon"].values; dht.close()
pos = {int(i2): j2 for j2, i2 in enumerate(ids)}
QH = QH[:, [pos[int(i2)] for i2 in node_ids]]
nT = min(QM.shape[0], QH.shape[0]); QM, QH = QM[:nT], QH[:nT]
rs = np.full(n, np.nan); be = np.full(n, np.nan)
for j in range(n):
    h_, m_ = QH[:, j], QM[:, j]
    if h_.std() > 1e-9 and m_.std() > 1e-9:
        rs[j] = np.corrcoef(h_, m_)[0, 1]; be[j] = m_.mean() / max(h_.mean(), 1e-9)
A = r["territorial"].get_physical("area_km2_local").cpu().numpy()
Acum = A.copy(); enf = collections.defaultdict(list)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    enf[int(s_)].append(int(d_))
for u in td.graph.topo_order.cpu().numpy():
    for v2 in enf.get(int(u), []):
        Acum[v2] += Acum[u]
print(f"\n=== FIDÉLITÉ v2 {REG} (2022-2024, {int(np.isfinite(rs).sum())} tronçons) ===")
print(f"r médian {np.nanmedian(rs):.3f} | beta médian {np.nanmedian(be):.3f}")
for lo, hi, lib in [(0, 50, 'têtes <50'), (50, 1000, '50-1000'), (1000, 1e9, '>1000 km²')]:
    msk = (Acum >= lo) & (Acum < hi)
    print(f"  {lib:12s} r {np.nanmedian(rs[msk]):.3f} | beta {np.nanmedian(be[msk]):.3f} (n={int(msk.sum())})")
lacm = td.graph.is_lake.bool().cpu().numpy()
print(f"  {'lacs':12s} r {np.nanmedian(rs[lacm]):.3f} | beta {np.nanmedian(be[lacm]):.3f} (n={int(lacm.sum())})")
np.savez_compressed(f"{_DATA_ROOT}/quebec/fidelite2_{REG}.npz", r=rs, beta=be, acum=Acum, lac=lacm)
