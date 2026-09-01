"""Passes du RAPPORT v1.0 : la machinerie de carte.py, plus la passe naturalisee et les dumps.

POURQUOI CE SCRIPT ET PAS etl_run. Les trente premieres passes ont fait tourner les
champions SOUS UNE AUTRE RECETTE que la leur (aucune des variables d'environnement de
leurs flottes n'etait posee ; ces checkpoints sont anterieurs a la fiche d'execution,
donc aucun avertissement). Resultat : OUTV a 0.546 de KGE median quand son champion
mesure vaut 0.796. La dette six, une fois de plus. carte.py, lui, porte le deploiement
VALIDE de chaque champion -- canal de demande d'evapotranspiration appris avec sa
correction par region, latents detectes, aquifere -- et a produit la carte provinciale
verifiee. On le prolonge donc : deux simulations par region, geree et naturalisee, et
les caches que le rapport lit.

  PYTHONIOENCODING=utf-8 python .runs/quebec/carte_rapport.py [regions...]
"""
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, ".runs/quebec")

import copy
import json
import tomllib

import numpy as np
import pandas as pd
import torch

from meandre.model import HydroModel
from meandre.utils.state import HydroState
from ckpt_util import a_des_latents
import joint_data
from et_module import compute_demand

# Racines portables (portage grappe, 2026-09-01) : les chemins absolus rendaient toute
# execution hors du poste d'origine impossible. Defauts inchanges.
import os as _osp
_DATA_ROOT = _osp.environ.get("MEANDRE_DATA", "D:/meandre-data")

GLOBAL = "best-gasp-etl-ds"
LOCAUX = {"gasp": "best-gasp-etl-ds", "sagu": "best-sagu-etl-ds", "mont": "best-mont-etl-ds",
          "outv": "best-outv-etl-qc", "slso": "best-slso-etl-canon", "slno": "best-slno-etl-canon"}
REGIONS = [a.lower() for a in sys.argv[1:]] or [
    "outv", "gasp", "sagu", "mont", "slno", "slso",
    "abit", "cnda", "cndb", "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SORTIE = os.environ.get("MEANDRE_RAPPORT", f"{_DATA_ROOT}/quebec/rapport")
os.makedirs(SORTIE, exist_ok=True)
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
AD = json.load(open("reports/deploy_adapters.json"))


def prelev_nul(w):
    """Prelevements a ZERO en gardant l'objet : simulate appelle gw_withdrawal sans garde."""
    z = copy.copy(w)
    for a in ("_vals", "_vals_gw"):
        if hasattr(z, a):
            setattr(z, a, getattr(z, a) * 0.0)
    if getattr(z, "net", None) is not None:
        z.net = z.net * 0.0
    return z


def dump_reach(chemin, Q, m, td, terr, n):
    Qr = Q.cpu().numpy()
    tt = pd.DatetimeIndex(pd.to_datetime(times))
    mois_cal = tt.month.to_numpy()
    qm = np.stack([Qr[mois_cal == k].mean(axis=0) for k in range(1, 13)])
    cle = tt.year * 12 + tt.month
    mois_u = np.unique(cle)
    qms = np.stack([Qr[cle == k].mean(axis=0) for k in mois_u])
    with torch.no_grad():
        sp = m.spatial_encoder(td.node_coords, terr.to_tensor())
    champs = {f"param_{k}": getattr(sp, k).detach().cpu().numpy()
              for k in sp.__dataclass_fields__
              if torch.is_tensor(getattr(sp, k)) and getattr(sp, k).shape[:1] == (n,)}
    w = td.withdrawals
    wnet = (w.net.abs().sum(dim=0).cpu().numpy()
            if getattr(w, "net", None) is not None else np.zeros(n, dtype=np.float32))
    np.savez_compressed(chemin,
                        q_mensuel=qm.astype(np.float32),
                        q_annuel=Qr.mean(axis=0).astype(np.float32),
                        q_mois_serie=qms.astype(np.float32),
                        mois_serie=mois_u.astype(np.int32),
                        coords=td.node_coords.cpu().numpy(),
                        prelev_net_abs=wnet.astype(np.float32),
                        **{k: v.astype(np.float32) for k, v in champs.items()})


for REG in REGIONS:
    ck = f".runs/quebec/checkpoints/{LOCAUX.get(REG, GLOBAL)}.pt"
    try:
        r = joint_data.load_region(REG, dict(cfg["loss"]), device=DEVICE)
    except Exception as ex:
        print(f"[{REG}] chargement impossible : {type(ex).__name__} {ex}", flush=True)
        continue
    td = r["train_data"]
    n = r["n_nodes"]
    times = pd.to_datetime(r["times"])[td.train_slice.start:]
    lat = a_des_latents(ck, n)
    demand = compute_demand(td.forcing, td.day_of_year, td.node_coords,
                            r["territorial"], DEVICE) * AD.get(REG, {}).get("debias_et", 1.0)
    f7 = torch.cat([td.forcing[:, :, :6], demand[:, :, None]], dim=2)
    m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
                   use_temporal=False, use_residual=False, use_travel_time_attn=False,
                   use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                   use_latent_codes=lat, latent_mode="additive", spatial_melt=True,
                   routing_mode="operator-lagged", predict_lake_params=True,
                   compile_soil=False, use_aquifer=True).to(DEVICE)
    m.load(ck)
    m.eval()
    m.vertical_column.etp_channel = 6
    for mode, w in (("avec", td.withdrawals), ("sans", prelev_nul(td.withdrawals))):
        with torch.no_grad():
            Q, _ = m.simulate(forcing=f7, initial_state=HydroState.zeros(n, device=DEVICE),
                              graph=td.graph, node_coords=td.node_coords,
                              territorial=td.territorial, withdrawals=w,
                              day_of_year=td.day_of_year)
        dump_reach(f"{SORTIE}/rap-{REG}-{mode}.npz", Q, m, td, r["territorial"], n)
        if mode == "avec":
            tt = pd.DatetimeIndex(times)
            hd = np.asarray((tt >= "2022-01-01") & (tt <= "2024-12-31"))
            qo = td.q_obs.cpu().numpy()[:len(tt)][hd]
            Qs = Q[torch.tensor(hd, device=DEVICE)][:, td.station_idx].cpu().numpy()
            np.savez_compressed(f"{SORTIE}/rap-{REG}-q.npz",
                                q_sim=Qs.astype(np.float32), q_obs=qo.astype(np.float32),
                                station_ids=np.array([str(x) for x in r["station_ids"]],
                                                     dtype=object),
                                dates=np.array([str(d)[:10] for d in tt[hd]]),
                                allow_pickle=True)
        del Q
        torch.cuda.empty_cache()
    print(f"[{REG}] passes ecrites ({'latents' if lat else 'sans latents'})", flush=True)
    del m, f7, demand
    torch.cuda.empty_cache()
print("PASSES RAPPORT v1.0 TERMINEES", flush=True)
