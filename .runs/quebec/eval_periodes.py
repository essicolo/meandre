"""L'ENTRAÎNEMENT sur-ajuste-t-il, ou la PÉRIODE de test est-elle simplement différente ?

Question d'Essi (2026-08-12) : « si l'apprentissage détériore, c'est que la fonction de
perte est mauvaise ». Peut-être, mais trois causes donnent la même signature (validation
0.834, tenu de côté 0.605) et appellent des remèdes opposés :
  a) la perte vise la mauvaise cible,
  b) le modèle sur-ajuste sa période de calage,
  c) la période 2022-2024 est climatiquement différente (mesuré en juin : pluie estivale
     +28 %, hivers +1.5 °C), et TOUT modèle y perd.

Un seul test les sépare : évaluer le modèle ANCRÉ (zéro paramètre appris, donc incapable
de sur-ajuster quoi que ce soit) sur LES DEUX périodes, et comparer sa chute à celle de
l'entraîné.
  - si l'ancré chute autant : cause (c), la perte est innocente, il faut une évaluation
    multi-périodes et non une nouvelle perte ;
  - si l'ancré reste stable : cause (a) ou (b), et le chantier est bien l'apprentissage.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/eval_periodes.py outv [checkpoint]
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hydrotel_calib import (load_calibrated_soil, load_linacre_nodes, load_melt_nodes,
                                         load_passage_pluie_neige, load_occupation_sol,
                                         load_milieux_humides, load_phenologie)
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
CKPT = sys.argv[2] if len(sys.argv) > 2 else None      # None = modèle ANCRÉ
MEMBRE = os.environ.get("FIDELITE_MEMBRE", "LN24HA")
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/{MEMBRE}/{REG.upper()}_{MEMBRE}_2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PERIODES = [("validation", "2019-01-01", "2021-12-31"), ("tenu de côté", "2022-01-01", "2024-12-31")]

cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])

# Le modèle ENTRAÎNÉ consomme la demande évaporative APPRISE comme 7e canal
# (etp_channel=6, et_mode="mcguinness"), là où l'ancré utilise Linacre calée. Pour
# comparer les deux sur les mêmes périodes il faut reconstruire ce canal à l'identique
# d'etl_run, sinon la colonne réclame set_linacre_params et l'évaluation échoue.
FORC = td.forcing[:, :, :6]
if CKPT:
    import torch.nn as _nn
    _ETB = "D:/meandre-data/quebec/checkpoints-etbench"
    _nrm = torch.load(f"{_ETB}/norm.pt", weights_only=False)
    _HH, _HC = _nrm["h_hist"], _nrm["h_comp"]
    _FS = r["territorial"].n_features
    _mlp = _nn.Sequential(_nn.Linear(12 + _FS + 3, 64), _nn.ReLU(), _nn.Linear(64, 1),
                          _nn.Softplus()).to(DEVICE)
    _sd = torch.load(f"{_ETB}/mlp.pt", weights_only=True)
    _mlp.load_state_dict({k.replace("head.", ""): v for k, v in _sd.items()}); _mlp.eval()
    with torch.no_grad():
        _F = td.forcing; _T = _F.shape[0]
        _mu, _sg = _nrm["mean"].to(DEVICE), _nrm["std"].to(DEVICE)
        _C = torch.cat([torch.zeros(1, n, 6, device=DEVICE), _F[:, :, :6].cumsum(0)], dim=0)
        _t = torch.arange(_T, device=DEVICE)
        _lo8 = torch.clamp(_t - (_HC - 1), min=0)
        _a8 = (_C[_t + 1] - _C[_lo8]) / (_t + 1 - _lo8).reshape(-1, 1, 1)
        _hi90 = torch.clamp(_t - (_HC - 1), min=1); _lo90 = torch.clamp(_t - (_HC - 1) - _HH, min=0)
        _a90 = (_C[_hi90] - _C[_lo90]) / torch.clamp(_hi90 - _lo90, min=1).reshape(-1, 1, 1)
        _doy = td.day_of_year
        _sc = torch.stack([torch.sin(2 * np.pi * _doy / 365.25),
                           torch.cos(2 * np.pi * _doy / 365.25)], dim=1)
        _lc0 = 0 if 40 < float(td.node_coords[:, 0].mean()) < 62 else 1
        _lat = td.node_coords[:, _lc0].float() / 50.0
        _st = torch.cat([r["territorial"].data, _lat[:, None]], dim=1)
        _dem = torch.empty(_T, n, device=DEVICE)
        for _l in range(0, _T, 365):
            _h = min(_l + 365, _T)
            _x = torch.cat([(_a8[_l:_h] - _mu) / _sg, (_a90[_l:_h] - _mu) / _sg,
                            _st[None, :, :-1].expand(_h - _l, -1, -1),
                            _sc[_l:_h, None, :].expand(_h - _l, n, 2),
                            _st[None, :, -1:].expand(_h - _l, -1, -1)], dim=2)
            _dem[_l:_h] = _mlp(_x.reshape(-1, _x.shape[-1])).reshape(_h - _l, n)
    _dem = _dem * float(os.environ.get("ETL_DEMAND_SCALE", "0.963"))
    FORC = torch.cat([td.forcing[:, :, :6], _dem[:, :, None]], dim=2)
    print(f"[periodes] demande ET apprise reconstruite : "
          f"{float(_dem.mean()) * 365.25:.0f} mm/an", flush=True)

m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode=("mcguinness" if CKPT else "linacre"),
    use_temperature=False,
    use_latent_codes=bool(CKPT), latent_mode="additive", spatial_melt=bool(CKPT),
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    use_aquifer=bool(CKPT)).to(DEVICE)
m.eval(); m.spatial_encoder.init_from_literature({})
if CKPT:
    m.vertical_column.etp_channel = 6
if CKPT:
    m.load(CKPT)
    # PIÈGE (rencontré le 2026-08-13) : l'occupation du sol, les milieux humides, la
    # phénologie, le noyau de versant et les lacs du trl sont des réglages d'EXÉCUTION,
    # posés par etl_run à chaque run et ABSENTS du point de reprise. Les omettre à
    # l'évaluation fait chuter le même checkpoint de 0.6051 à 0.4449 : on mesure alors
    # un modèle amputé, pas celui qui a été entraîné.
    _lc = load_occupation_sol(PROJ, node_ids, device=DEVICE)
    _lc.update(load_milieux_humides(PROJ, node_ids, device=DEVICE))
    m.vertical_column.set_land_cover(_lc)
    m.vertical_column.set_phenology(load_phenologie(PROJ) or None)
    m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))
    import numpy as _np2
    _lg = [l.split() for l in (Path(PROJ) / "physitel" / "troncon.trl").read_text(
        encoding="latin-1").splitlines()[3:] if l.strip()]
    _dl = {int(t[0]): (float(t[4 + int(t[3]) + 1]), float(t[4 + int(t[3]) + 2]),
                       float(t[4 + int(t[3]) + 3])) for t in _lg if int(t[1]) != 1}
    _ix = {int(i): j for j, i in enumerate(node_ids)}
    _su = _np2.full(n, _np2.nan); _cc = _np2.full(n, _np2.nan); _kk = _np2.full(n, _np2.nan)
    for _t2, (_s2, _c2, _k2) in _dl.items():
        if _t2 in _ix:
            _su[_ix[_t2]], _cc[_ix[_t2]], _kk[_ix[_t2]] = _s2, _c2, _k2
    _lac = td.graph.is_lake.bool().cpu().numpy()
    _cv = _np2.isfinite(_su) & _lac & (_su > 0)
    _cvt = torch.tensor(_cv, device=DEVICE)
    _kt = torch.tensor(_np2.nan_to_num(_np2.where(_cv, _cc / _np2.clip(_su * 1e6, 1, None),
                                                  _np2.nan), nan=1e-4),
                       dtype=torch.float32, device=DEVICE)
    _bt = torch.tensor(_np2.nan_to_num(_kk, nan=1.5), dtype=torch.float32, device=DEVICE)
    _o0 = m.spatial_encoder.lake_params
    m.spatial_encoder.lake_params = lambda *a_, **k_: (
        lambda k2, b2: (torch.where(_cvt, _kt, k2), torch.where(_cvt, _bt, b2)))(*_o0(*a_, **k_))
    m.set_lake_area(torch.tensor(_np2.where(_cv, _su, 1.0), dtype=torch.float32))
    print(f"[periodes] checkpoint {Path(CKPT).name} + réglages d'exécution d'etl_run "
          f"(occupation forêt {float(_lc['f_forest_raw'].mean()):.2f}, MH, phéno, HGM, lacs trl)",
          flush=True)
else:
    m.vertical_column.set_calibrated_soil(load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE))
    m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
    m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
    m.vertical_column.t_neige_seuil = load_passage_pluie_neige(PROJ)
    _lc = load_occupation_sol(PROJ, node_ids, device=DEVICE)
    _lc.update(load_milieux_humides(PROJ, node_ids, device=DEVICE))
    m.vertical_column.set_land_cover(_lc)
    m.vertical_column.set_phenology(load_phenologie(PROJ) or None)
    m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))
    print(f"[periodes] modèle ANCRÉ sur {MEMBRE} (zéro paramètre appris)", flush=True)

with torch.no_grad():
    Q, _ = m.simulate(forcing=FORC, initial_state=HydroState.zeros(n, device=DEVICE),
                         graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                         withdrawals=td.withdrawals, day_of_year=td.day_of_year)

def kge(o, s):
    if o.std() < 1e-9 or s.std() < 1e-9 or o.mean() <= 0 or s.mean() <= 0:
        return np.nan
    rr = np.corrcoef(o, s)[0, 1]
    b = s.mean() / o.mean()
    g = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1.0 - np.sqrt((rr - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

sid = td.station_idx.cpu().numpy()
qo_all = td.q_obs.cpu().numpy()
Qs_all = Q[:, td.station_idx].cpu().numpy()
nom = Path(CKPT).name if CKPT else f"ANCRÉ({MEMBRE})"
print(f"\n=== {nom} sur {REG.upper()}, mêmes stations, même formule ===")
res = {}
for lib, a, b in PERIODES:
    msk = np.asarray((tt >= a) & (tt <= b))
    ks = []
    for k in range(len(sid)):
        o, s = qo_all[msk, k], Qs_all[msk, k]
        v = np.isfinite(o) & np.isfinite(s)
        if v.sum() >= 60:
            ks.append(kge(o[v], s[v]))
    ks = np.array([x for x in ks if np.isfinite(x)])
    res[lib] = np.median(ks)
    print(f"  {lib:14s} ({a[:4]}-{b[:4]}) : n={len(ks)} | médian {np.median(ks):.4f}")
# FENÊTRE GLISSANTE (idée d'Essi : « est-ce qu'on pourrait prendre un autre hold-out
# que 2022-2024 ? »). Le modèle ANCRÉ n'a rien appris, donc son score par fenêtre de
# 3 ans mesure la DIFFICULTÉ INTRINSÈQUE de chaque période, sans aucun sur-ajustement
# possible. Si 2022-2024 y ressort comme une période anormalement dure, le tenu de côté
# actuel est un juge biaisé et il faut en changer.
print(f"\n=== difficulté par fenêtre de 3 ans ({nom}) ===")
for a0 in range(2001, 2023, 3):
    a1 = a0 + 2
    msk = np.asarray((tt >= f"{a0}-01-01") & (tt <= f"{a1}-12-31"))
    if msk.sum() < 300:
        continue
    ks = []
    for k in range(len(sid)):
        o, s_ = qo_all[msk, k], Qs_all[msk, k]
        v = np.isfinite(o) & np.isfinite(s_)
        if v.sum() >= 60:
            ks.append(kge(o[v], s_[v]))
    ks = np.array([x for x in ks if np.isfinite(x)])
    if len(ks):
        print(f"  {a0}-{a1} : n={len(ks):2d} | médian {np.median(ks):.4f}")

if len(res) == 2:
    d = res["tenu de côté"] - res["validation"]
    print(f"\n  CHUTE validation -> tenu de côté : {d:+.4f}")
    print("  repère : le modèle ENTRAÎNÉ chute de 0.834 (val, métrique d'entraînement) "
          "à 0.6051. Si l'ANCRÉ chute autant, la période est en cause et la perte est "
          "innocente ; s'il reste stable, le chantier est bien l'apprentissage.")
