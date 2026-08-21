"""D'OÙ VIENT LE DÉFICIT DE FÉVRIER ? (chantier hiver, 2026-08-19)

Le plus gros écart mensuel du champion OUTV : février à 0,69 de l'observé, alors
qu'Hydrotel fait l'erreur INVERSE (1,24). Deux hypothèses restent au registre après
la réfutation de la recharge (R11) :

  H-neige   : l'eau de février est BLOQUÉE dans le manteau (fonte hivernale absente,
              porte de gel trop fermée) -> le SWE monte quand l'observé monte.
  H-nappe   : l'eau est DRAINÉE trop vite avant février (vidange k_gw trop rapide,
              ou nappe régionale absente) -> le réservoir est déjà vide en février.

Le discriminant est le BILAN par mois : si la production totale (surface + hypo +
base) suffit et que seul le débit manque, c'est le routage ; si la production
manque et que le SWE gonfle, c'est la neige ; si la production manque sans que le
SWE gonfle, c'est le stockage souterrain.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb DIAG_AQUIFER=1 python .runs/quebec/hiver_diag.py outv \
      .runs/quebec/checkpoints/best-outv-etl-aq30.pt
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
                                         imposed_retention_curve,
                                         appariement_provincial)
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region
from recipe import set_lake_area_from_hydrolakes

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
CKPT = sys.argv[2] if len(sys.argv) > 2 else ".runs/quebec/checkpoints/best-outv-etl-aq30.pt"
MEMBRE_REF = os.environ.get("MEMBRE_REF", "MG24HK")
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
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
    use_latent_codes=False, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True,
    # compile_soil comme a l'ENTRAINEMENT (x17,6) : sans lui ce diagnostic
    # demande plus d'une heure de GPU pour une seule trajectoire.
    compile_soil=os.environ.get("DIAG_COMPILE", "1") == "1",
    use_aquifer=os.environ.get("DIAG_AQUIFER", "1") == "1").to(DEVICE)
m.eval(); m.spatial_encoder.init_from_literature({})
_cs = load_calibrated_soil(PROJ, node_ids, 0.15, device=DEVICE)
m.vertical_column.set_calibrated_soil(imposed_retention_curve(_cs, os.environ.get("DIAG_AQUIFER", "1") == "1"))
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEVICE))
m.vertical_column.t_neige_seuil = load_passage_pluie_neige(PROJ)
_lc = load_occupation_sol(PROJ, node_ids, device=DEVICE)
_lc.update(load_milieux_humides(PROJ, node_ids, device=DEVICE))
m.vertical_column.set_land_cover(_lc)
m.vertical_column.set_phenology(load_phenologie(PROJ) or None)
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEVICE))
set_lake_area_from_hydrolakes(m, REG, r["territorial"].get_physical("area_km2_local"), n)
m.load(CKPT)
print(f"[hiver] {Path(CKPT).name} sur {REG.upper()}", flush=True)

with torch.no_grad():
    Q, _, D = m.simulate(forcing=td.forcing[:, :, :6], initial_state=HydroState.zeros(n, device=DEVICE),
                         graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                         withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                         return_diagnostics=True)

msk = np.asarray((tt >= T0) & (tt <= T1))
sid = td.station_idx.cpu().numpy()
mois = tt[msk].month
qo = td.q_obs.cpu().numpy()[msk]
qm = Q[torch.tensor(msk, device=DEVICE)][:, td.station_idx].cpu().numpy()

def npy(x):
    return x[torch.tensor(msk, device=DEVICE)].cpu().numpy() if torch.is_tensor(x) else np.asarray(x)[msk]

swe = npy(D.swe) if hasattr(D, "swe") else None
fonte = npy(D.snowmelt); qbase = npy(D.q_baseflow); rech = npy(D.recharge)
lat_mm = npy(D.lateral_mm); etr = npy(D.etr)

# membre Hydrotel de référence
z = xr.open_zarr(f"C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data/06_posttraitement/"
                 f"posttraitement_{MEMBRE_REF}.zarr")
cols = appariement_provincial(REG, [int(node_ids[i]) for i in sid],
                              np.asarray(z["troncon_id"].values).astype(str))
qh = z["Dis"].sel(time=slice(T0, T1)).transpose("time", "troncon_idx").values[
    :, [c if c is not None else 0 for c in cols]]
z.close()
nT = min(len(qo), len(qh))
qo, qm, qh, mois = qo[:nT], qm[:nT], qh[:nT], mois[:nT]
for nom in ("swe", "fonte", "qbase", "rech", "lat_mm", "etr"):
    v = locals()[nom]
    if v is not None:
        locals()[nom] = v[:nT]

print(f"\n=== BILAN PAR MOIS, moyennes sur le domaine ({T0[:4]}-{T1[:4]}) ===")
print(f"{'mois':>4s} {'Q/Qobs':>7s} {'Hyd/Qobs':>9s} | {'SWE mm':>8s} {'fonte':>7s} "
      f"{'prod mm/j':>10s} {'base mm/j':>10s} {'rech mm/j':>10s} {'ETR mm/j':>9s}")
for mo in list(range(11, 13)) + list(range(1, 6)):
    k = mois == mo
    if k.sum() < 10:
        continue
    rm = np.nanmedian(np.nansum(qm[k], 0) / np.clip(np.nansum(qo[k], 0), 1e-9, None))
    rh = np.nanmedian(np.nansum(qh[k], 0) / np.clip(np.nansum(qo[k], 0), 1e-9, None))
    print(f"{mo:4d} {rm:7.3f} {rh:9.3f} | {np.nanmean(swe[k]):8.1f} {np.nanmean(fonte[k]):7.3f} "
          f"{np.nanmean(lat_mm[k]):10.3f} {np.nanmean(qbase[k]):10.4f} "
          f"{np.nanmean(rech[k]):10.4f} {np.nanmean(etr[k]):9.3f}")

print(f"\n=== LE SWE MONTE-T-IL QUAND L'OBSERVÉ MONTE ? (février, par année) ===")
an = tt[msk][:nT].year
for a in sorted(set(an)):
    kf = (an == a) & (mois == 2)
    if kf.sum() < 10:
        continue
    swe_deb, swe_fin = np.nanmean(swe[kf][:3]), np.nanmean(swe[kf][-3:])
    qo_m = np.nanmean(np.nanmean(qo[kf], 1)); qm_m = np.nanmean(np.nanmean(qm[kf], 1))
    print(f"  {a} : SWE {swe_deb:6.1f} -> {swe_fin:6.1f} mm ({swe_fin-swe_deb:+6.1f}) | "
          f"fonte {np.nanmean(fonte[kf]):.3f} mm/j | Qobs {qo_m:7.1f} Qsim {qm_m:7.1f} m3/s "
          f"({qm_m/max(qo_m,1e-9):.2f})")

print(f"\n=== LA PRODUCTION SUFFIT-ELLE ? (février, conversion mm/j -> m3/s au domaine) ===")
aire = float(td.forcing.shape[1])
kf = mois == 2
prod_mm = np.nanmean(lat_mm[kf]); base_mm = np.nanmean(qbase[kf])
print(f"  production latérale totale {prod_mm:.3f} mm/j, dont base {base_mm:.4f} mm/j "
      f"({100*base_mm/max(prod_mm,1e-9):.1f} %)")
print(f"  déficit de débit en février : {1 - np.nanmedian(np.nansum(qm[kf],0)/np.clip(np.nansum(qo[kf],0),1e-9,None)):.1%}")
