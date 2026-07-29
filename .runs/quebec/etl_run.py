"""Étape 2 du design ET appris : run mono-région avec la demande évaporative du
module MLP (banc et_bench, supervisé MOD16) injectée comme 7e canal de forçage
(etp_channel), à la place de formule ETP × K_c. Un seul changement vs la recette
v4 de la région ; baselines GASP : v4 0.489 / v7 (ancrages) 0.577 held-out.

  ETL_REGION=gasp ETL_EPOCHS=12 python .runs/quebec/etl_run.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".runs/quebec"))
try:
    sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tomllib
import numpy as np
import torch
import torch.nn as nn
from meandre.model import HydroModel
from meandre.training.trainer import Trainer, TrainingConfig, TrainingData
from meandre.utils.metrics import kge as kge_fn
from meandre.utils.state import HydroState
from joint_data import load_region

REG = os.environ.get("ETL_REGION", "gasp").lower()
N_EPOCHS = int(os.environ.get("ETL_EPOCHS", "12"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_CFG = ".runs/quebec/config/gasp-v4.toml"
CKPT = f".runs/quebec/checkpoints/best-{REG}-etl{os.environ.get('ETL_TAG', '')}.pt"   # ETL_TAG évite d'écraser les checkpoints de diagnostic
ETB = "D:/meandre-data/quebec/checkpoints-etbench"

cfg = tomllib.load(open(BASE_CFG, "rb"))
lcfg = dict(cfg["loss"]); tcfg = cfg["training"]; mcfg = cfg["model"]
if "ETL_WSNOW" in os.environ:
    # seuils de fonte appris contre MOD10 (fonte à 0 jusqu'à Tmax+5.5 au banc freshet
    # = 2 semaines de retard ; la donnée entre par la loss, leçon pilote4b/4c)
    lcfg["w_snow"] = float(os.environ["ETL_WSNOW"])
    print(f"[etl] w_snow = {lcfg['w_snow']} (fonte supervisée MOD10)")
if "ETL_WET" in os.environ:
    # mode appris : w_et(MOD16) est un DOUBLE ancrage (le module encode déjà MOD16,
    # biaisé +15-30 % à l'est vs bilan) — il poussait K_c à 1.07 malgré beta 0.78 (etl2)
    lcfg["w_et"] = float(os.environ["ETL_WET"])
    print(f"[etl] w_et override = {lcfg['w_et']}")

r = load_region(REG, lcfg, device=DEVICE)
td, vd = r["train_data"], r["val_data"]
n_nodes = r["n_nodes"]
print(f"[etl] {REG}: {n_nodes} nœuds, {r['n_gauges']} jauges")

# ── demande évaporative du module appris (MLP du banc, gelé) ────────────────
norm = torch.load(f"{ETB}/norm.pt", weights_only=False)
H_HIST, H_COMP = norm["h_hist"], norm["h_comp"]
F_STATIC = r["territorial"].n_features
mlp = nn.Sequential(nn.Linear(12 + F_STATIC + 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()).to(DEVICE)
sd = torch.load(f"{ETB}/mlp.pt", weights_only=True)
mlp.load_state_dict({k.replace("head.", ""): v for k, v in sd.items()})
mlp.eval()

with torch.no_grad():
    F = td.forcing            # (T, N, 6+) sur device
    T = F.shape[0]
    mean, std = norm["mean"].to(DEVICE), norm["std"].to(DEVICE)
    C = torch.cat([torch.zeros(1, n_nodes, 6, device=DEVICE), F[:, :, :6].cumsum(0)], dim=0)
    t_ar = torch.arange(T, device=DEVICE)
    lo8 = torch.clamp(t_ar - (H_COMP - 1), min=0)
    a8 = (C[t_ar + 1] - C[lo8]) / (t_ar + 1 - lo8).reshape(-1, 1, 1)
    hi90, lo90 = torch.clamp(t_ar - (H_COMP - 1), min=1), torch.clamp(t_ar - (H_COMP - 1) - H_HIST, min=0)
    a90 = (C[hi90] - C[lo90]) / torch.clamp(hi90 - lo90, min=1).reshape(-1, 1, 1)
    # NB : fenêtres TRAÎNANTES (8 j finissant à t, 90 j avant) — au banc la fenêtre 8 j
    # était le composite [t, t+8) ; décalage ~4 j << cycle saisonnier de l'ET.
    doy = td.day_of_year
    sc = torch.stack([torch.sin(2 * np.pi * doy / 365.25), torch.cos(2 * np.pi * doy / 365.25)], dim=1)
    lat_col = 0 if 40 < float(td.node_coords[:, 0].mean()) < 62 else 1
    lat = td.node_coords[:, lat_col].float() / 50.0
    stat = torch.cat([r["territorial"].data, lat[:, None]], dim=1)   # (N, F+1)
    demand = torch.empty(T, n_nodes, device=DEVICE)
    for lo in range(0, T, 365):
        hi = min(lo + 365, T)
        a8n = (a8[lo:hi] - mean) / std
        a90n = (a90[lo:hi] - mean) / std
        scb = sc[lo:hi, None, :].expand(hi - lo, n_nodes, 2)
        x = torch.cat([a8n, a90n, stat[None, :, :-1].expand(hi - lo, -1, -1), scb, stat[None, :, -1:].expand(hi - lo, -1, -1)], dim=2)
        demand[lo:hi] = mlp(x.reshape(-1, x.shape[-1])).reshape(hi - lo, n_nodes)
print(f"[etl] demande ET apprise : {float(demand.mean()) * 365.25:.0f} mm/an moyen | max {float(demand.max()):.1f} mm/j")

_ds = float(os.environ.get("ETL_DEMAND_SCALE", "1.0"))
if _ds != 1.0:
    # débiaisage RÉGIONAL structurel de la demande (ratio bilan P-Q / MOD16) : appliqué
    # au canal, le gradient ne peut pas le défaire (le prior K_c doux était re-défait
    # à l'entraînement : mont-kc 0.583 < 0.617 inférence)
    demand = demand * _ds
    print(f"[etl] demande ET débiaisée × {_ds} (bilan/MOD16 régional)")
f7 = torch.cat([F[:, :, :6], demand[:, :, None]], dim=2)


def with_forcing(d):
    return TrainingData(forcing=f7, q_obs=d.q_obs, station_mask=d.station_mask,
                        station_idx=d.station_idx, graph=d.graph, node_coords=d.node_coords,
                        territorial=d.territorial, withdrawals=d.withdrawals,
                        day_of_year=d.day_of_year, train_slice=d.train_slice, val_slice=d.val_slice,
                        et_obs=d.et_obs, tws_obs=d.tws_obs)


td, vd = with_forcing(td), with_forcing(vd)

model = HydroModel(
    n_nodes=n_nodes,
    n_territorial=F_STATIC,
    n_forcing=6,
    use_temporal=False,
    use_residual=False,
    use_travel_time_attn=False,
    use_frost_rankinen=bool(mcfg.get("use_frost_rankinen", True)),
    column_theta_init_frac=float(mcfg.get("column_theta_init_frac", 0.9)),
    param_mode="nerf",
    column_mode="hydrotel",
    et_mode="mcguinness",   # court-circuité par etp_channel
    use_temperature=False,
    use_latent_codes=bool(mcfg.get("use_latent_codes", True)),
    latent_mode="additive",
    spatial_melt=bool(mcfg.get("spatial_melt", True)),
    routing_mode=mcfg.get("routing_mode", "operator-lagged"),
    predict_lake_params=bool(mcfg.get("predict_lake_params", True)),
    compile_soil=bool(mcfg.get("compile_soil", True)),
    # AQUIFÈRE RESTITUANT (spec Essi 2026-07-28) : recharge -> réservoir lent par nœud,
    # vidange k_gw NeRF (prior = récessions MESURÉES des jauges). Banc partition :
    # krec 5e-5 + k_gw 0.068 = baseflow 24%, +0.07 KGE en inférence pure.
    use_aquifer=os.environ.get("ETL_AQUIFER", "0") == "1",
    # LEVIERS PICS (r vs Hydrotel) : advection pure (onde cinématique sans diffusion),
    # célérité dépendante du débit, UH de versant. Activés par env pour le banc GASP.
    pure_advection=os.environ.get("ETL_ADVECTION", "0") == "1",
    discharge_dependent_celerity=os.environ.get("ETL_DQCEL", "0") == "1",
    use_hillslope_uh=os.environ.get("ETL_HILLSLOPE", "0") == "1",
).to(DEVICE)
lp = dict(cfg.get("literature_prior") or {})
lp["K_c"] = float(os.environ.get("ETL_KC", "1.0"))   # autour de la demande apprise : 1.0 neutre,
# ou PRIOR MESURÉ = ratio bilan P-Q / MOD16 par région (MOD16 sur-évapore le sud +17-25%,
# et_bilan_check 2026-07-21 ; K_c×0.8 en inférence : MONT test 0.544->0.617, beta 0.73->0.91)
# K_sat_1 (surface) : l'init/prior littérature à 0.080 m/j est 6× trop perméable
# (diag GASP : le sol absorbe 83% de l'orage, coeff ruiss 17% vs 30-50% réel).
# Recaler l'ancre du prior plus bas re-génère la crue (banc d'impulsion), K_sat_3
# intact => baseflow préservé. Cible via env (défaut = valeur actuelle).
if "ETL_KSAT1" in os.environ:
    lp["K_sat_1"] = float(os.environ["ETL_KSAT1"])
    print(f"[etl] K_sat_1 prior recalé -> {lp['K_sat_1']} m/j (génération de crue)")
if "ETL_KGW" in os.environ:
    lp["k_gw"] = float(os.environ["ETL_KGW"])   # prior mesuré (récessions jauges)
    print(f"[etl] k_gw prior -> {lp['k_gw']} /j (récessions mesurées)")
if "ETL_TMELT" in os.environ:
    # seuil de fonte NeRF : cible init+prior Hydrotel-comme-littérature (QC ~ +2°C),
    # le champ reste libre par nœud (PAS un delta autour d'un squelette figé)
    lp["T_melt"] = float(os.environ["ETL_TMELT"])
    print(f"[etl] T_melt prior -> {lp['T_melt']} °C (seuil de fonte NeRF)")
model.spatial_encoder.init_from_literature(lp)
if os.environ.get("ETL_FONTE_LIT", "0") == "1":
    # taux de fonte par couvert : init Hydrotel-littérature (4.5/9/18 mm/j/°C au lieu
    # de 12/14/16) — apprenables comme avant, C_f NeRF = leur variation spatiale
    import math as _mth
    with torch.no_grad():
        for nm, v in [("sp_fonte_conif", 4.5), ("sp_fonte_feu", 9.0), ("sp_fonte_dec", 18.0)]:
            getattr(model.vertical_column, nm).copy_(torch.tensor(_mth.log(_mth.expm1(v))))
    print("[etl] taux de fonte init littérature-Hydrotel : 4.5/9/18 mm/j/°C")
model.vertical_column.etp_channel = 6
if "ETL_KREC" in os.environ:
    import math as _mk
    _kv = float(os.environ["ETL_KREC"])
    _lo, _hi = model.vertical_column._krec_bounds
    _x = min(max((_kv - _lo) / (_hi - _lo), 1e-6), 1 - 1e-6)
    with torch.no_grad():
        model.vertical_column.krec_raw.copy_(torch.tensor(_mk.log(_x / (1 - _x))))
    print(f"[etl] krec init -> {_kv:.0e} (drainage profond, banc partition)")
if "ETL_MELT_DIR" in os.environ:
    # fonte RÉGIONALE calée (taux+seuils plateforme), NeRF mscale module autour.
    # A/B inférence 2026-07-25 : +0.149 KGE sur checkpoint gasp (v7 : +0.088 entraîné).
    from meandre.data.hydrotel_calib import load_melt_nodes
    _mp = load_melt_nodes(os.environ["ETL_MELT_DIR"], r["node_ids"], device=DEVICE)
    model.vertical_column.set_melt_params(_mp)
    print(f"[etl] fonte régionale ancrée ({os.environ['ETL_MELT_DIR'].split('/')[-1]}) | "
          f"taux méd {float(_mp['taux_c'].median()):.1f}/{float(_mp['taux_f'].median()):.1f}/{float(_mp['taux_d'].median()):.1f}")

# CÉLÉRITÉ : le K_musk appris collapse à 24h/tronçon (init) -> retard cumulé 6j du pic
# (diag GASP). Facteur d'échelle sur K_musk_hours (célérité de base plus rapide), en
# gardant le routage opérateur rapide (contrairement à dq_celerity qui le casse).
_kms = float(os.environ.get("ETL_KMUSK_SCALE", "1"))
if _kms != 1.0:
    _orig_fwd = model.spatial_encoder.forward
    def _fwd_kscale(*a, _o=_orig_fwd, _s=_kms, **k):
        sp = _o(*a, **k)
        sp.K_musk_hours = torch.clamp(sp.K_musk_hours * _s, min=1.0, max=48.0)
        return sp
    model.spatial_encoder.forward = _fwd_kscale
    print(f"[etl] K_musk × {_kms} (célérité accélérée, routage opérateur préservé)")
print(f"[etl] modèle {sum(p.numel() for p in model.parameters()):,} params | etp_channel=6 (demande apprise × K_c NeRF, init 1.0)")

tconf = TrainingConfig(
    n_epochs=N_EPOCHS,
    lr=float(tcfg.get("lr", 5e-4)),
    chunk_steps=int(tcfg.get("chunk_steps", 45)),
    tbptt_steps=int(tcfg.get("tbptt_steps", 365)),
    grad_clip=float(tcfg.get("clip_grad_norm", 1.0)),
    w_prior=float(tcfg.get("w_prior", 0.005)),
    best_metric="kge_median",
    # autopilot du TOML : LR plateau + garde-fou régression (sans lui, GASP/MONT-etl
    # divergeaient après le pic epoch ~7-12, val -0.15 non rattrapée — bug 2026-07-22)
    autopilot=bool(tcfg.get("autopilot", True)),
    autopilot_grace_epochs=int(tcfg.get("autopilot_grace_epochs", 8)),
    autopilot_lr_patience=int(tcfg.get("autopilot_lr_patience", 6)),
    autopilot_lr_factor=float(tcfg.get("autopilot_lr_factor", 0.5)),
    autopilot_lr_min=float(tcfg.get("autopilot_lr_min", 1e-5)),
    autopilot_beta_threshold=float(tcfg.get("autopilot_beta_threshold", 0.10)),
    autopilot_restart_regression=float(tcfg.get("autopilot_restart_regression", 0.05)),
    autopilot_restart_max=int(tcfg.get("autopilot_restart_max", 3)),
    val_every=1,
)
tr = Trainer(model=model, loss_fn=r["loss_fn"], train_data=td, val_data=vd,
             config=tconf, run_name=f"{REG}-etl", checkpoint_path=CKPT)
tr.fit()

# ── held-out 2022-2024 (best checkpoint) ─────────────────────────────────────
model.load(CKPT); model.eval()
with torch.no_grad():
    Q, _ = model.simulate(forcing=f7, initial_state=HydroState.zeros(n_nodes, device=DEVICE),
                          graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
                          withdrawals=td.withdrawals, day_of_year=td.day_of_year)
times = r["times"]
sl = (times >= "2022-01-01") & (times <= "2024-12-31")
slt = torch.tensor(sl.values if hasattr(sl, "values") else sl, device=DEVICE)
Qs = Q[slt][:, td.station_idx].cpu()
t0 = td.train_slice.start
qo_test = td.q_obs[np.flatnonzero(sl)[0] - t0 : np.flatnonzero(sl)[-1] - t0 + 1].cpu()
ks = []
for s in range(Qs.shape[1]):
    v = ~torch.isnan(qo_test[:, s]) & ~torch.isnan(Qs[:, s])
    if v.sum() < 60: continue
    ks.append(float(kge_fn(qo_test[v, s], Qs[v, s])))
ks = np.array(ks)
print(f"\n[etl] HELD-OUT 2022-2024 {REG}: n={len(ks)} | médian {np.median(ks):.4f} | mean {ks.mean():.4f}")
print("[etl] DONE")
