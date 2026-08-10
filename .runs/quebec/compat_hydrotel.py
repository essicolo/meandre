"""BANC DE COMPATIBILITÉ méandre / Hydrotel, étage par étage (demande d'Essi :
établir la compatibilité AVANT toute amélioration).

Tout vient du projet Hydrotel : météo (Thiessen sur <REG>.nc, fenêtre 2020-2026, donc
on atteint la date d'état ENNEIGÉE que notre forçage à nous ne couvrait pas), sol bv3c,
ETP Linacre calée, fonte calée, seuil pluie/neige de thiessen.csv, occupation du sol,
milieux humides isolés, lacs du troncon.trl, noyau de versant du cache .hgm. Zéro
entraînement, zéro paramètre appris : ce qui diverge ne peut venir que du CODE.

Étages comparés, du haut vers le bas de la colonne :
  1. NEIGE      stock par couvert (fonte_neige_*.csv) — jamais comparé jusqu'ici
  2. SOL        theta 1/2/3 (bilan_vertical_*.csv)
  3. PRODUCTION apport latéral par tronçon (acheminement_riviere_*.csv)
  4. ROUTAGE    débit AMONT et débit AVAL par tronçon (état + debit_aval.nc)
  5. SAISON     cycle mensuel du débit réseau

  PYTHONIOENCODING=utf-8 python .runs/quebec/compat_hydrotel.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, numpy as np, pandas as pd, torch, xarray as xr
from scipy.spatial import cKDTree
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.routing.withdrawals import WithdrawalData
from meandre.data.hydrotel_calib import (load_calibrated_soil, load_linacre_nodes, load_melt_nodes,
                                         load_passage_pluie_neige, load_occupation_sol,
                                         load_milieux_humides)
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))

r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]

# ── forçage ENTIÈREMENT issu du projet (fenêtre complète, on atteint février 2026) ──
dm = xr.open_dataset(f"{PROJ}/meteo/{REG.upper()}.nc")
tm = pd.DatetimeIndex(pd.to_datetime(dm["time"].values))
ncrd = td.node_coords.cpu().numpy()
latcol = 0 if 40 < float(ncrd[:, 0].mean()) < 62 else 1
lat0 = float(ncrd[:, latcol].mean())
proj = lambda lon, lat: np.c_[np.asarray(lon) * 111.32 * np.cos(np.radians(lat0)),
                              np.asarray(lat) * 110.57]
_, jn = cKDTree(proj(dm["x"].values, dm["y"].values)).query(
    proj(ncrd[:, 1 - latcol], ncrd[:, latcol]), k=1)
T = len(tm)
forc = torch.zeros(T, n, 6, device=DEVICE)
for c, v in [(0, "pr"), (1, "tasmin"), (2, "tasmax")]:
    forc[:, :, c] = torch.tensor(dm[v].values[:, jn], dtype=torch.float32, device=DEVICE)
dm.close()
doy = torch.tensor(tm.dayofyear.values, dtype=torch.long, device=DEVICE)
wdr = WithdrawalData.zeros(T, n, device=DEVICE)
print(f"[compat] {REG} : météo du PROJET {tm[0].date()} -> {tm[-1].date()} ({T} j), "
      f"P {float(forc[:, :, 0].mean()) * 365.25:.0f} mm/an | prélèvements NULS des deux côtés",
      flush=True)

# ── modèle : tout figé depuis le projet ──────────────────────────────────────
m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    use_latent_codes=False, latent_mode="additive", spatial_melt=False,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=False).to(DEVICE)
m.eval(); m.spatial_encoder.init_from_literature({})
m.vertical_column.compile_column = False
m.vertical_column.set_calibrated_soil(load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE))
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.t_neige_seuil = load_passage_pluie_neige(PROJ)
lc = load_occupation_sol(PROJ, node_ids, device=DEVICE)
lc.update(load_milieux_humides(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_land_cover(lc)
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))

lignes = [l.split() for l in (Path(PROJ) / "physitel" / "troncon.trl").read_text(encoding="latin-1").splitlines()[3:] if l.strip()]
dl = {int(t[0]): (float(t[4 + int(t[3]) + 1]), float(t[4 + int(t[3]) + 2]), float(t[4 + int(t[3]) + 3]))
      for t in lignes if int(t[1]) != 1}
idx = {int(i): j for j, i in enumerate(node_ids)}
surf = np.full(n, np.nan); cc = np.full(n, np.nan); kk = np.full(n, np.nan)
for tid, (s_, c_, k_) in dl.items():
    if tid in idx:
        surf[idx[tid]], cc[idx[tid]], kk[idx[tid]] = s_, c_, k_
lacm = td.graph.is_lake.bool().cpu().numpy()
couv = np.isfinite(surf) & lacm & (surf > 0)
cvt = torch.tensor(couv, device=DEVICE)
kt = torch.tensor(np.nan_to_num(np.where(couv, cc / np.clip(surf * 1e6, 1, None), np.nan), nan=1e-4),
                  dtype=torch.float32, device=DEVICE)
bt = torch.tensor(np.nan_to_num(kk, nan=1.5), dtype=torch.float32, device=DEVICE)
_o = m.spatial_encoder.lake_params
m.spatial_encoder.lake_params = lambda *a, **kw: (
    lambda k2, b2: (torch.where(cvt, kt, k2), torch.where(cvt, bt, b2)))(*_o(*a, **kw))
m.set_lake_area(torch.tensor(np.where(couv, surf, 1.0), dtype=torch.float32))
print(f"[compat] figé : sol+linacre+fonte+seuil({m.vertical_column.t_neige_seuil:+.2f}°C)"
      f"+occupation(forêt {float(lc['f_forest_raw'].mean()):.2f})+MH+lacs+hgm", flush=True)

with torch.no_grad():
    Q, _, diag = m.simulate(forcing=forc, initial_state=HydroState.zeros(n, device=DEVICE),
                            graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                            withdrawals=wdr, day_of_year=doy, return_diagnostics=True)

# ── outils de comparaison ────────────────────────────────────────────────────
def lire_etat(nom, date):
    f = Path(PROJ) / "etat" / f"{nom}_{date.replace('-', '')}00.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, sep=";", skiprows=3)
    d.columns = [c.strip() for c in d.columns]
    return d

def par_troncon(vals_uhrh, ids_uhrh):
    """Agrège une variable par UHRH vers les tronçons, au prorata des aires."""
    from meandre.data.physitel_loader import _parse_troncon
    tr = {t["id"]: t for t in _parse_troncon(Path(PROJ) / "physitel" / "troncon.trl")}
    uh = pd.read_csv(Path(PROJ) / "physitel" / "uhrh.csv", sep=";", skiprows=1)
    ca = next(c for c in uh.columns if "superficie" in c.lower() or "aire" in c.lower())
    aires = dict(zip(uh[uh.columns[0]].astype(int), uh[ca].astype(float)))
    par = dict(zip(ids_uhrh.astype(int), vals_uhrh))
    out = np.full(n, np.nan)
    for j, nid in enumerate(node_ids):
        t = tr.get(int(nid))
        if t is None: continue
        vs, ws = [], []
        for u in t["uhrh_ids"]:
            u = abs(int(u))
            if u in par:
                vs.append(par[u]); ws.append(aires.get(u, 1.0))
        if vs:
            out[j] = float(np.average(vs, weights=ws))
    return out

def score(sim, obs, nom, unite=""):
    v = np.isfinite(sim) & np.isfinite(obs)
    if v.sum() < 10:
        print(f"  {nom:26s} : pas assez de données"); return
    rr = np.corrcoef(sim[v], obs[v])[0, 1] if sim[v].std() > 1e-12 and obs[v].std() > 1e-12 else np.nan
    print(f"  {nom:26s} hydrotel {np.median(obs[v]):9.4f} | méandre {np.median(sim[v]):9.4f} "
          f"| rapport {np.median(sim[v]) / max(abs(np.median(obs[v])), 1e-9):6.3f} | corr {rr:+.3f} {unite}")

ti = pd.DatetimeIndex(tm)
def idx_date(d):
    k = np.flatnonzero(ti == pd.Timestamp(d))
    return int(k[0]) if len(k) else None

# ── 1. NEIGE ────────────────────────────────────────────────────────────────
print("\n=== ÉTAGE 1 : NEIGE (stock, mm d'équivalent en eau) ===")
for date in ("2026-02-19", "2023-08-01"):
    i = idx_date(date)
    fn_ = lire_etat("fonte_neige", date)
    if fn_ is None or i is None:
        print(f"  {date} : indisponible"); continue
    ids_u = fn_[fn_.columns[0]].values
    st = {c: pd.to_numeric(fn_[c], errors="coerce").values * 1000.0
          for c in fn_.columns if c.upper().startswith("STOCK")}
    pc = {"CONIFERS": lc["f_forest_conifer_raw"].cpu().numpy(),
          "FEUILLUS": lc["f_forest_deciduous_raw"].cpu().numpy()}
    pc["DECOUVERT"] = np.clip(1.0 - pc["CONIFERS"] - pc["FEUILLUS"], 0.0, 1.0)
    tot = np.zeros(n)
    for c, v in st.items():
        cl = c.replace("STOCK", "").strip().upper()
        w = pc.get(cl)
        if w is None: continue
        tot += w * np.nan_to_num(par_troncon(v, ids_u))
    score(diag.swe[i].cpu().numpy(), tot, f"stock pondéré {date}", "mm")

# ── 2. SOL ──────────────────────────────────────────────────────────────────
print("\n=== ÉTAGE 2 : SOL (teneurs en eau) ===")
for date in ("2023-08-01", "2026-02-19"):
    i = idx_date(date); bv = lire_etat("bilan_vertical", date)
    if bv is None or i is None:
        print(f"  {date} : indisponible"); continue
    ids_u = bv[bv.columns[0]].values
    for j, nm in enumerate(["theta1", "theta2", "theta3"], start=1):
        obs = par_troncon(pd.to_numeric(bv[f"THETA {j}"], errors="coerce").values, ids_u)
        score(getattr(diag, nm)[i].cpu().numpy(), obs, f"{nm} {date}")

# ── 3. PRODUCTION et 4. ROUTAGE ────────────────────────────────────────────
print("\n=== ÉTAGES 3-4 : PRODUCTION et ROUTAGE (par tronçon) ===")
for date in ("2023-08-01", "2026-02-19"):
    i = idx_date(date); ach = lire_etat("acheminement_riviere", date)
    if ach is None or i is None:
        print(f"  {date} : indisponible"); continue
    pos = {int(v): k for k, v in enumerate(ach[ach.columns[0]].values)}
    ordre = [pos.get(int(x), -1) for x in node_ids]
    ok = np.array([o >= 0 for o in ordre])
    pr = lambda col: np.where(ok, pd.to_numeric(ach[col], errors="coerce").values[np.array(ordre).clip(0)], np.nan)
    score(diag.q_lateral[i].cpu().numpy(), pr("APPORT"), f"apport latéral {date}", "m³/s")
    if "DEBIT AMONT" in ach.columns:
        score(diag.q_upstream[i].cpu().numpy(), pr("DEBIT AMONT"), f"débit amont {date}", "m³/s")
    score(Q[i].cpu().numpy(), pr("DEBIT AVAL"), f"débit aval {date}", "m³/s")

# ── 5. SÉRIE COMPLÈTE + SAISON ─────────────────────────────────────────────
dht = xr.open_dataset(f"{PROJ}/simulation/simulation/resultat/debit_aval.nc")
tht = pd.to_datetime(dht["time"].values); mh = (tht >= T0) & (tht <= T1)
QH = dht["debit_aval"].values[mh]; ids = dht["idtroncon"].values; dht.close()
pos = {int(i2): j2 for j2, i2 in enumerate(ids)}
QH = QH[:, [pos[int(i2)] for i2 in node_ids]]
hd = np.asarray((ti >= T0) & (ti <= T1))
QM = Q[torch.tensor(hd, device=DEVICE)].cpu().numpy()
nT = min(QM.shape[0], QH.shape[0]); QM, QH = QM[:nT], QH[:nT]
rs = np.full(n, np.nan); be = np.full(n, np.nan)
for j in range(n):
    h_, m_ = QH[:, j], QM[:, j]
    if h_.std() > 1e-9 and m_.std() > 1e-9:
        rs[j] = np.corrcoef(h_, m_)[0, 1]; be[j] = m_.mean() / max(h_.mean(), 1e-9)
print(f"\n=== ÉTAGE 5 : SÉRIE 2022-2024 ({int(np.isfinite(rs).sum())} tronçons) ===")
print(f"  r médian {np.nanmedian(rs):.3f} | beta médian {np.nanmedian(be):.3f} | "
      f"lacs r {np.nanmedian(rs[lacm]):.3f}")
mois = pd.DatetimeIndex(ti[hd][:nT]).month
print("  cycle mensuel méandre/hydrotel : " +
      " ".join(f"{mo}:{np.nanmean(QM[mois == mo]) / max(np.nanmean(QH[mois == mo]), 1e-9):.2f}"
               for mo in range(1, 13)))
np.savez_compressed(f"D:/meandre-data/quebec/compat_{REG}.npz", r=rs, beta=be, lac=lacm)
