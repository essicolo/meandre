"""TEST DE FIDÉLITÉ TOTAL (proposition d'Essi) : obtient-on les débits d'Hydrotel en
FIGEANT tous les paramètres sur ses fichiers de projet, sans aucun entraînement ?

Tout est pris du projet Hydrotel LN24HA de la région :
- sol COMPLET : bv3c.csv (y compris épaisseurs 3 m, krec ~1e-7, coef_recharge 0 — c'est
  ce qu'Hydrotel exécute réellement), via load_calibrated_soil ;
- ETP : Linacre + coefficient d'optimisation régional (linacre.csv), clone validé à la
  décimale, via load_linacre_nodes/set_linacre_params ;
- fonte : taux et seuils calés (degre_jour_modifie.csv) via load_melt_nodes ;
- lacs : surface + loi de tarage Q = c·h^k de troncon.trl (k_lake = c/A, beta = k) ;
- versant : noyau HGM du cache .hgm (convolution <=10 j, comme le C++).
Pas d'aquifère (Hydrotel n'en a pas), pas de canal ETP appris, pas de codes latents.
Routage rivière : Muskingum init littérature (K=24 h) — seule pièce non fidèle restante,
avec le forçage (-hyb ~ SIMAT contre l'interpolation de stations d'Hydrotel).

On compare étage par étage à Hydrotel : teneurs en eau et apport latéral au 2023-08-01
(fichiers d'état), débit aval en série complète 2022-2024 (debit_aval.nc, 3412 tronçons).
Là où ça colle, le clone est fidèle ; là où ça dérive, on tient le module divergent.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/fidelite_hydrotel.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, numpy as np, pandas as pd, torch, xarray as xr
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hydrotel_calib import (load_calibrated_soil, load_linacre_nodes,
                                         load_melt_nodes, load_passage_pluie_neige)
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"
DATE_ETAT = "2023-08-01"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))

r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]

# ── MÊMES INTRANTS DES DEUX CÔTÉS (exigence d'Essi) : la météo du PROJET Hydrotel
# (Thiessen sur <REG>.nc, ce que simulation.csv déclare) remplace CaSR. Sans ça on
# compare deux forçages en croyant comparer deux codes. FIDELITE_METEO=casr pour
# revenir au forçage du run.
if os.environ.get("FIDELITE_METEO", "projet") == "projet":
    import xarray as _xr
    from scipy.spatial import cKDTree as _KD
    _tt0 = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
    _dm = _xr.open_dataset(f"{PROJ}/meteo/{REG.upper()}.nc")
    _tm = pd.to_datetime(_dm["time"].values)
    _ncrd = td.node_coords.cpu().numpy()
    _latcol = 0 if 40 < float(_ncrd[:, 0].mean()) < 62 else 1
    _lat0 = float(_ncrd[:, _latcol].mean())
    def _proj(lon, lat):
        return np.c_[np.asarray(lon) * 111.32 * np.cos(np.radians(_lat0)), np.asarray(lat) * 110.57]
    _sx, _sy = _dm["x"].values, _dm["y"].values
    if np.nanmax(np.abs(_sx)) > 360:
        raise SystemExit("meteo du projet en coordonnees projetees : adapter")
    _, _jn = _KD(_proj(_sx, _sy)).query(_proj(_ncrd[:, 1 - _latcol], _ncrd[:, _latcol]), k=1)
    _comm = _tt0.intersection(_tm)
    _if = _tt0.get_indexer(_comm); _im = _tm.get_indexer(_comm)
    _forc = td.forcing[:, :, :6].clone()
    for _c, _v in [(0, "pr"), (1, "tasmin"), (2, "tasmax")]:
        _V = _dm[_v].values
        _forc[torch.tensor(_if, device=DEVICE), :, _c] = torch.tensor(
            _V[_im][:, _jn], dtype=torch.float32, device=DEVICE)
    _dm.close()
    _pm = float(_forc[:, :, 0].mean()) * 365.25
    _pc = float(td.forcing[:, :, 0].mean()) * 365.25
    print(f"[meteo] PROJET Hydrotel (Thiessen) : {len(_comm)} jours | P {_pm:.0f} mm/an "
          f"(CaSR du run : {_pc:.0f})", flush=True)
    FORC = _forc
else:
    FORC = td.forcing[:, :, :6]
    print(f"[meteo] forçage du run (CaSR)", flush=True)

m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    use_latent_codes=False, latent_mode="additive", spatial_melt=False,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=False).to(DEVICE)
m.eval()
m.spatial_encoder.init_from_literature({})
m.vertical_column.compile_column = False

# 1. sol COMPLET du calage (épaisseurs, textures, krec, coef_recharge)
soil = load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE)
m.vertical_column.set_calibrated_soil(soil)
# 2. ETP Linacre + coefficient régional
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
# 3. fonte calée
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
# 4. lacs troncon.trl
lignes = [l.strip() for l in (Path(PROJ) / "physitel" / "troncon.trl").read_text(encoding="latin-1").splitlines() if l.strip()]
dl = {}
for l in lignes[3:]:
    t = l.split()
    if int(t[1]) != 1:
        ptr = 4 + int(t[3])
        dl[int(t[0])] = (float(t[ptr+1]), float(t[ptr+2]), float(t[ptr+3]))
idx = {int(i): j for j, i in enumerate(node_ids)}
surf = np.full(n, np.nan); cc = np.full(n, np.nan); kk = np.full(n, np.nan)
for tid, (s_, c_, k_) in dl.items():
    if tid in idx:
        surf[idx[tid]], cc[idx[tid]], kk[idx[tid]] = s_, c_, k_
lacm = td.graph.is_lake.bool().cpu().numpy()
couv = np.isfinite(surf) & lacm & (surf > 0)
couv_t = torch.tensor(couv, device=DEVICE)
kt = torch.tensor(np.nan_to_num(np.where(couv, cc / np.clip(surf * 1e6, 1, None), np.nan), nan=1e-4),
                  dtype=torch.float32, device=DEVICE)
bt = torch.tensor(np.nan_to_num(kk, nan=1.5), dtype=torch.float32, device=DEVICE)
_olp = m.spatial_encoder.lake_params
def lp(*a, _o=_olp, **kw):
    k2, b2 = _o(*a, **kw)
    return torch.where(couv_t, kt, k2), torch.where(couv_t, bt, b2)
m.spatial_encoder.lake_params = lp
m.set_lake_area(torch.tensor(np.where(couv, surf, 1.0), dtype=torch.float32))
# 4b. seuil de partition pluie/neige calibré (thiessen.csv) — jamais lu jusqu'au 10 août
_seuil = load_passage_pluie_neige(PROJ)
m.vertical_column.t_neige_seuil = _seuil
print(f"[fidelite] seuil pluie/neige du projet : {_seuil:+.4f} °C (méandre codait 0.0)")
# 5. noyau de versant
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))
print(f"[fidelite] {REG} : sol+linacre+fonte+lacs+hgm figés depuis {PROJ}", flush=True)

with torch.no_grad():
    Q, _, diag = m.simulate(forcing=FORC, initial_state=HydroState.zeros(n, device=DEVICE),
                            graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                            withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                            return_diagnostics=True)
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])

# ETAGE 1 : etats internes au 2023-08-01
tag = DATE_ETAT.replace("-", "") + "00"
bv = pd.read_csv(f"{PROJ}/etat/bilan_vertical_{tag}.csv", sep=";", skiprows=3)
ach = pd.read_csv(f"{PROJ}/etat/acheminement_riviere_{tag}.csv", sep=";", skiprows=3)
ach.columns = [c.strip() for c in ach.columns]
i = int(np.flatnonzero(tt == pd.Timestamp(DATE_ETAT))[0])
print(f"\n=== {REG} {DATE_ETAT} (paramètres FIGÉS, zéro entraînement) ===")
for j, nm in enumerate(["theta1", "theta2", "theta3"], start=1):
    hv = np.median(bv[f"THETA {j}"].values)
    mv = float(getattr(diag, nm)[i].median())
    print(f"  {nm:14s} hydrotel {hv:.4f} | meandre {mv:.4f} | rapport {mv/max(hv,1e-9):.3f}")
al_h = ach["APPORT"].values
al_m = diag.q_lateral[i].cpu().numpy()
v = np.isfinite(al_h) & np.isfinite(al_m)
print(f"  apport latéral  hydrotel méd {np.median(al_h[v]):.4f} | meandre {np.median(al_m[v]):.4f} "
      f"| rapport {np.median(al_m[v])/max(np.median(al_h[v]),1e-9):.3f} | corr {np.corrcoef(al_h[v], al_m[v])[0,1]:+.3f}")

# ETAGE 2 : debit aval, series completes
dht = xr.open_dataset(f"{PROJ}/simulation/simulation/resultat/debit_aval.nc")
tht = pd.to_datetime(dht["time"].values); mh = (tht >= T0) & (tht <= T1)
QH = dht["debit_aval"].values[mh]; ids = dht["idtroncon"].values; dht.close()
pos = {int(i2): j2 for j2, i2 in enumerate(ids)}
QH = QH[:, [pos[int(i2)] for i2 in node_ids]]
hd = np.asarray((tt >= T0) & (tt <= T1))
QM = Q[torch.tensor(hd, device=DEVICE)].cpu().numpy()
nT = min(QM.shape[0], QH.shape[0]); QM, QH = QM[:nT], QH[:nT]
rs = np.full(n, np.nan); be = np.full(n, np.nan)
for j in range(n):
    h_, m_ = QH[:, j], QM[:, j]
    if h_.std() > 1e-9 and m_.std() > 1e-9:
        rs[j] = np.corrcoef(h_, m_)[0, 1]
        be[j] = m_.mean() / max(h_.mean(), 1e-9)
A = r["territorial"].get_physical("area_km2_local").cpu().numpy()
import duckdb, collections
con = duckdb.connect(f"D:/meandre-data/quebec/{REG}.duckdb", read_only=True)
e = con.execute("select src, dst from edges").fetchdf(); con.close()
Acum = A.copy(); enf = collections.defaultdict(list)
for s_, d_ in zip(e["src"].values, e["dst"].values):
    enf[int(s_)].append(int(d_))
for u in td.graph.topo_order.cpu().numpy():
    for v2 in enf.get(int(u), []):
        Acum[v2] += Acum[u]
# ── OÙ ET QUAND ÇA BIFURQUE : décomposition des 3 flux + cycle saisonnier ──
print("\n=== DÉCOMPOSITION de la production (mm/an, moyennes réseau) ===")
_hdt = torch.tensor(hd, device=DEVICE)
for _nm in ("prod_surf", "prod_hypo", "prod_base"):
    _v = getattr(diag, _nm, None)
    if _v is None:
        print(f"  {_nm:10s} absent")
        continue
    print(f"  {_nm:10s} {_v[_hdt].mean().item() * 365.25:7.1f} mm/an")
print(f"  {'total':10s} {diag.lateral_mm[_hdt].mean().item() * 365.25:7.1f} mm/an")

print("\n=== CYCLE SAISONNIER du débit réseau : méandre / Hydrotel ===")
_mois = pd.DatetimeIndex(tt[hd][:nT]).month
print(f"  {'mois':>5s} {'méandre':>9s} {'hydrotel':>9s} {'rapport':>8s}")
for _m in range(1, 13):
    _k = _mois == _m
    if _k.sum() < 10:
        continue
    _qm = float(np.nanmean(QM[_k])); _qh = float(np.nanmean(QH[_k]))
    print(f"  {_m:5d} {_qm:9.2f} {_qh:9.2f} {_qm / max(_qh, 1e-9):8.3f}")

print(f"\n  débit aval vs Hydrotel (2022-2024, {int(np.isfinite(rs).sum())} tronçons) :")
print(f"  r médian {np.nanmedian(rs):.3f} | beta médian {np.nanmedian(be):.3f}")
for lo, hi, lib in [(0, 50, 'têtes <50'), (50, 1000, '50-1000'), (1000, 1e9, '>1000 km²')]:
    msk = (Acum >= lo) & (Acum < hi)
    print(f"    {lib:12s} r {np.nanmedian(rs[msk]):.3f} | beta {np.nanmedian(be[msk]):.3f} (n={int(msk.sum())})")
print(f"    {'lacs':12s} r {np.nanmedian(rs[lacm]):.3f} | beta {np.nanmedian(be[lacm]):.3f} (n={int(lacm.sum())})")

# repère : KGE aux jauges de ce clone fige
qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
Qs = QM[:, td.station_idx.cpu().numpy()]
ks = []
for s in range(Qs.shape[1]):
    o, si = qo[:nT, s], Qs[:, s]
    v = np.isfinite(o) & np.isfinite(si)
    if v.sum() < 60: continue
    rr = np.corrcoef(o[v], si[v])[0, 1]; b = si[v].mean() / o[v].mean()
    g = (si[v].std() / si[v].mean()) / (o[v].std() / o[v].mean())
    ks.append(1 - np.sqrt((rr - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))
print(f"\n  KGE aux jauges (repère, obs) : {np.median(ks):.4f} (Hydrotel ~0.82, champion méandre 0.4992)")
