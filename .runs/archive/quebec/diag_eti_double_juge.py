"""L'ETI ameliore-t-il les DEUX juges a la fois, sans aucun entrainement ?

Motif (2026-08-27, R50-R51). Le degre-jour fond trop tot : GRACE montre un stockage qui
descend un a deux mois trop tot (R47-R48), CanSWE montre un manteau qui perd un quart de
sa masse en avril quand la realite n'en perd rien (R50), et l'ETI juge sur la MASSE
corrige ce calendrier sur SAGU et OUTV sans aucun calage, aux valeurs de LITTERATURE
(R51). Reste a savoir si le MEME changement ameliore aussi GRACE -- si la fonte trop
precoce est la cause commune aux deux symptomes, une seule correction doit regler les
deux juges a la fois, ce qui se verifie en INFERENCE, avant de payer un seul epoch.

melt_mode="eti" est deja cable dans HydrotelColumn : tf et srf sont des SCALAIRES
GLOBAUX APPRENABLES (nn.Parameter), initialises aux valeurs de litterature (1.2e-3,
9.4e-6). Ce script les laisse a leur valeur d'origine (aucun entrainement) et ne change
que le MODE de fonte, pour isoler l'effet du mecanisme de la moindre optimisation.

    python .runs/quebec/diag_eti_double_juge.py sagu
    python .runs/quebec/diag_eti_double_juge.py outv
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), ".runs/quebec"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import tomllib
import torch
import xarray as xr

from domain_data import load_domain
from meandre.model import HydroModel
from meandre.utils import paths as _paths
from meandre.utils.state import HydroState

REG = (sys.argv[1] if len(sys.argv) > 1 else "sagu").lower()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
dom = load_domain([REG], dict(cfg["loss"]), device=DEV)
td = dom["train_data"]

# ── canal SW_in, ajoute au forcage a 6 canaux ────────────────────────────────
fsw = _paths.data_path("quebec", f"forcing-{REG}-swin.nc")
if not os.path.exists(fsw):
    print(f"[diag] pas de cache SW_in pour {REG} : ETI impossible ici"); sys.exit(1)
dsw = xr.open_dataset(fsw)
SW = torch.tensor(dsw["forcing"].values[:, :, 0], dtype=torch.float32, device=DEV)
dsw.close()
F6 = td.forcing[:]                              # (T, N, 6), materialise une fois
F7 = torch.cat([F6, SW[:, :, None]], dim=2)      # (T, N, 7)
print(f"[diag] {REG} : forcage etendu a 7 canaux (SW_in ajoute)", flush=True)


def construire(melt_mode):
    m = HydroModel(
        n_nodes=dom["n_nodes"], n_territorial=dom["territorial"].n_features, n_forcing=7,
        use_temporal=False, use_residual=False, use_travel_time_attn=False,
        use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
        column_mode="hydrotel", et_mode="linacre", use_temperature=False,
        use_latent_codes=False, spatial_melt=True,
        routing_mode=cfg["model"].get("routing_mode", "operator-lagged"),
        predict_lake_params=True, compile_soil=False, use_aquifer=True,
        melt_mode=melt_mode).to(DEV)
    if dom.get("land_cover"):
        m.vertical_column.set_land_cover(dom["land_cover"])
    if dom.get("melt_params"):
        m.vertical_column.set_melt_params(dom["melt_params"])
    if dom.get("linacre"):
        m.vertical_column.set_linacre_params(*dom["linacre"])
        m.vertical_column.etp_channel = None
    if dom.get("phenology"):
        m.vertical_column.set_phenology(dom["phenology"])
    if dom.get("soil"):
        from meandre.data.hydrotel_calib import imposed_retention_curve
        m.vertical_column.set_calibrated_soil(imposed_retention_curve(dom["soil"], True))
    m.vertical_column.split_mode = "wet_bulb"
    m.vertical_column.t_neige_seuil = -0.8
    if melt_mode == "degree_day":
        m.vertical_column.melt_seasonal_amp = 0.5   # recette 1.0 telle quelle
    _lp = dict(cfg.get("literature_prior") or {})
    _lp["K_sat_1"] = 0.04; _lp["K_c"] = 1.0; _lp["k_gw"] = 0.07; _lp.setdefault("krec", 5e-5)
    m.spatial_encoder.init_from_literature(_lp)
    m.spatial_encoder.prior_on_krec = True
    return m


def simuler(m):
    with torch.no_grad():
        _, _, diag = m.simulate(
            forcing=F7, initial_state=HydroState.zeros(dom["n_nodes"], device=DEV),
            graph=td.graph, node_coords=td.node_coords, territorial=td.territorial,
            withdrawals=td.withdrawals, day_of_year=td.day_of_year, return_diagnostics=True)
    return diag


def juger_grace(m, diag):
    sp = m.spatial_encoder(td.node_coords, td.territorial.to_tensor())
    z1 = getattr(m.vertical_column, "z1", 0.15)
    stor = ((diag.theta1 * z1 + diag.theta2 * sp.Z2 + diag.theta3 * sp.Z3) * 1000.0
            + diag.swe + diag.s_gw + diag.canopy
            + (diag.wet_vol if getattr(diag, "wet_vol", None) is not None
               else torch.zeros_like(diag.swe))).mean(dim=1).detach().cpu().numpy()
    obs = td.tws_obs[:, 0].cpu().numpy()          # un seul bassin ici
    mois = np.array([t.month for t in dom["times"]])
    ok = np.isfinite(obs)
    cs = np.array([stor[(mois == m2) & ok].mean() for m2 in range(1, 13)])
    co = np.array([obs[(mois == m2) & ok].mean() for m2 in range(1, 13)])
    cs -= cs.mean(); co -= co.mean()
    r = float(np.corrcoef(cs, co)[0, 1])
    print(f"  GRACE  : correlation climatologie = {r:.3f} | amplitude sim/obs = "
          f"{(cs.max() - cs.min()) / max(co.max() - co.min(), 1e-9):.2f} | pics sim/obs "
          f"= mois {1 + int(np.argmax(cs))}/{1 + int(np.argmax(co))}")


def juger_canswe(m, diag):
    from meandre.data.basin_cache import BasinCache
    bc = BasinCache(_paths.data_path("quebec", f"{REG}.duckdb"))
    mes, sites = bc.load_canswe("2000-01-01", "2024-12-31")
    if mes is None or len(mes) == 0:
        print("  CanSWE : aucune donnee"); return
    times = dom["times"]
    pos = pd.Series(np.arange(len(times)), index=pd.DatetimeIndex(times).normalize())
    mes = mes.copy()
    mes["t"] = pos.reindex(pd.DatetimeIndex(mes.date).normalize()).to_numpy()
    mes = mes[np.isfinite(mes.t) & np.isfinite(mes.swe_mm) & (mes.swe_mm >= 0)]
    swe = diag.swe.detach().cpu().numpy()
    mes["sim"] = swe[mes.t.astype(int), mes.node_idx.astype(int)]
    mes["mois"] = pd.DatetimeIndex(mes.date).month
    print("  CanSWE, climatologie mensuelle (mm)")
    print("           " + "".join(f"{x:6d}" for x in (10, 11, 12, 1, 2, 3, 4, 5, 6)))
    sim_c, obs_c = [], []
    for x in (10, 11, 12, 1, 2, 3, 4, 5, 6):
        sub = mes[mes.mois == x]
        sim_c.append(sub.sim.mean()); obs_c.append(sub.swe_mm.mean())
    print("  simule   " + "".join(f"{v:6.0f}" for v in sim_c))
    print("  CanSWE   " + "".join(f"{v:6.0f}" for v in obs_c))


for mode in ("degree_day", "eti"):
    print(f"\n=== {mode} ===", flush=True)
    modele = construire(mode)
    diag = simuler(modele)
    juger_grace(modele, diag)
    juger_canswe(modele, diag)
