"""OÙ SE TROUVE L'ÉCART qui nous sépare du MEILLEUR membre d'Hydrotel ?

Objectif d'Essi (2026-08-15) : dépasser le meilleur d'Hydrotel (0.83-0.85 par station,
contre 0.7810 pour méandre sur OUTV). Avant de choisir un levier (GRU, ETP apprise,
données auxiliaires), on décompose l'écart : composantes du KGE par station, cycle
saisonnier du biais, et contribution de chaque saison au score.

Le principe : un KGE se décompose en corrélation, biais de volume et rapport de
variabilité. Savoir LEQUEL manque oriente le levier — une corrélation faible appelle du
timing (routage, fonte), un biais appelle du volume (ETP, forçage), un gamma appelle de
l'amplitude (génération de crue).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/diagnostic_ecart.py outv \
      .runs/quebec/checkpoints/best-outv-etl-socle.pt
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, numpy as np, pandas as pd, torch, xarray as xr
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hydrotel_calib import (load_calibrated_soil, load_linacre_nodes, load_melt_nodes,
                                         load_passage_pluie_neige, load_occupation_sol,
                                         load_milieux_humides, load_phenologie,
                                         appariement_provincial)
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
CKPT = sys.argv[2] if len(sys.argv) > 2 else f".runs/quebec/checkpoints/best-{REG}-etl-socle.pt"
MEMBRE_REF = os.environ.get("MEMBRE_REF", "MG24HK")   # le meilleur membre sur OUTV
PLAT = "C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA"
PROJ = f"{PLAT}/{REG.upper()}_LN24HA_2020"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"

cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEVICE)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])

m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    # spatial_melt=true dans la config, donc le point de reprise PORTE ces poids :
    # les omettre fait chuter le même checkpoint de 0.781 à 0.735 (4e occurrence de
    # « un point de reprise ne définit pas un modèle », dette #6 du registre).
    use_latent_codes=False, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=False,
    # DIAG_AQUIFER=1 pour les points de reprise entraînés AVEC aquifère (aq30 et
    # suivants). Cinquième occurrence évitée du piège « un point de reprise ne définit
    # pas un modèle » : ce script le construisait toujours sans.
    use_aquifer=os.environ.get("DIAG_AQUIFER", "0") == "1").to(DEVICE)
m.eval(); m.spatial_encoder.init_from_literature({})
# mêmes réglages d'exécution que l'entraînement du socle (un point de reprise ne définit
# pas un modèle : voir la dette #6 du registre)
_cs = load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE)
m.vertical_column.set_calibrated_soil({k: v for k, v in _cs.items()
                                       if not k.startswith(("ks", "thetas"))})
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.t_neige_seuil = load_passage_pluie_neige(PROJ)
_lc = load_occupation_sol(PROJ, node_ids, device=DEVICE)
_lc.update(load_milieux_humides(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_land_cover(_lc)
m.vertical_column.set_phenology(load_phenologie(PROJ) or None)
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))
m.load(CKPT)
print(f"[diag] {Path(CKPT).name} sur {REG.upper()}", flush=True)

with torch.no_grad():
    Q, _ = m.simulate(forcing=td.forcing[:, :, :6], initial_state=HydroState.zeros(n, device=DEVICE),
                      graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                      withdrawals=td.withdrawals, day_of_year=td.day_of_year)

msk = np.asarray((tt >= T0) & (tt <= T1))
sid = td.station_idx.cpu().numpy()
qo = td.q_obs.cpu().numpy()[msk]
qm = Q[torch.tensor(msk, device=DEVICE)][:, td.station_idx].cpu().numpy()
mois = tt[msk].month

# membre de référence d'Hydrotel, mêmes stations
z = xr.open_zarr(f"C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data/06_posttraitement/"
                 f"posttraitement_{MEMBRE_REF}.zarr")
cols = appariement_provincial(REG, [int(node_ids[i]) for i in sid],
                              np.asarray(z["troncon_id"].values).astype(str))
qh = z["Dis"].sel(time=slice(T0, T1)).transpose("time", "troncon_idx").values[
    :, [c if c is not None else 0 for c in cols]]
z.close()
nT = min(len(qo), len(qh)); qo, qm, qh, mois = qo[:nT], qm[:nT], qh[:nT], mois[:nT]

def comp(o, s):
    v = np.isfinite(o) & np.isfinite(s)
    if v.sum() < 60 or o[v].std() < 1e-9 or s[v].std() < 1e-9:
        return (np.nan,) * 4
    o, s = o[v], s[v]
    rr = np.corrcoef(o, s)[0, 1]; b = s.mean() / o.mean()
    g = (s.std() / s.mean()) / (o.std() / o.mean())
    return rr, b, g, 1 - np.sqrt((rr - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

print(f"\n=== COMPOSANTES DU KGE, médianes sur {len(sid)} stations ({T0[:4]}-{T1[:4]}) ===")
print(f"{'modèle':22s} {'r':>7s} {'beta':>7s} {'gamma':>7s} {'KGE':>7s}")
for nom, sim in [("méandre (socle)", qm), (f"Hydrotel {MEMBRE_REF}", qh)]:
    C = np.array([comp(qo[:, k], sim[:, k]) for k in range(len(sid))])
    C = C[np.isfinite(C[:, 3])]
    print(f"{nom:22s} {np.median(C[:,0]):7.3f} {np.median(C[:,1]):7.3f} "
          f"{np.median(C[:,2]):7.3f} {np.median(C[:,3]):7.3f}")

print(f"\n=== BIAIS MENSUEL (simulé / observé, médiane sur stations) ===")
print(f"{'mois':>5s} {'méandre':>9s} {'Hydrotel':>9s}")
for mo in range(1, 13):
    k = mois == mo
    if k.sum() < 10:
        continue
    rm = np.nanmedian(np.nansum(qm[k], 0) / np.clip(np.nansum(qo[k], 0), 1e-9, None))
    rh = np.nanmedian(np.nansum(qh[k], 0) / np.clip(np.nansum(qo[k], 0), 1e-9, None))
    print(f"{mo:5d} {rm:9.3f} {rh:9.3f}")

print(f"\n=== COMBIEN COÛTE CHAQUE SAISON ? (KGE recalculé en EXCLUANT une saison) ===")
sais = {"hiver (DJF)": [12, 1, 2], "printemps (MAM)": [3, 4, 5],
        "été (JJA)": [6, 7, 8], "automne (SON)": [9, 10, 11]}
base = np.nanmedian([comp(qo[:, k], qm[:, k])[3] for k in range(len(sid))])
print(f"  KGE complet : {base:.4f}")
for lib, mm in sais.items():
    keep = ~np.isin(mois, mm)
    v = np.nanmedian([comp(qo[keep, k], qm[keep, k])[3] for k in range(len(sid))])
    print(f"  sans {lib:18s} {v:.4f}   ({v - base:+.4f})")
