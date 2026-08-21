"""BALAYAGE CROISE recharge x vidange (chantier hiver, 2026-08-19).

R11 avait libere la recharge en laissant la vidange a 0.0645 /j (residence 15 j) :
on remplissait un reservoir perce, l'eau ressortait avant fevrier, le modele
s'effondrait. Les recessions hivernales des jauges d'OUTV (1316 segments purs)
donnent un taux MESURE de 0.0273 /j (residence 37 j), 2.4x plus lent, avec une
composante lente a 0.0090 /j (111 j).

Ici on croise les deux, en INFERENCE PURE depuis le champion (aucune epoque) :
le but n'est pas un score mais de voir si le mecanisme repare FEVRIER, qui est le
plus gros ecart mensuel (0.688 de l'observe contre 1.235 pour Hydrotel).

On imprime aussi la recharge annuelle en mm/an : c'est un LIVRABLE du projet, la
realite quebecoise se compte en dizaines a centaines de mm/an, et le debit seul la
prefere nulle (note d'enjeu au registre).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb MEANDRE_NSUBSTEP=64 python .runs/quebec/aquifere_sweep.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
from pathlib import Path
import tomllib, math, numpy as np, pandas as pd, torch
from meandre.model import HydroModel
from meandre.utils.state import HydroState
from meandre.data.hydrotel_calib import (load_calibrated_soil, load_linacre_nodes, load_melt_nodes,
                                         load_passage_pluie_neige, load_occupation_sol,
                                         load_milieux_humides, load_phenologie,
                                         imposed_retention_curve)
from meandre.data.hgm_loader import lire_hgm
from joint_data import load_region
from recipe import set_lake_area_from_hydrolakes
from meandre.utils.metrics import kge as kge_fn

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
CKPT = os.environ.get("SWEEP_CKPT", ".runs/quebec/checkpoints/best-outv-etl-aq30.pt")
PROJ = f"C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA/{REG.upper()}_LN24HA_2020"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0, T1 = "2022-01-01", "2024-12-31"

cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device=DEV)
td = r["train_data"]; n = r["n_nodes"]; node_ids = r["node_ids"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
msk = np.asarray((tt >= T0) & (tt <= T1))
mskt = torch.tensor(msk, device=DEV)
mois = tt[msk].month
qo = td.q_obs.cpu().numpy()[msk]

m = HydroModel(n_nodes=n, n_territorial=r["territorial"].n_features, n_forcing=6,
    use_temporal=False, use_residual=False, use_travel_time_attn=False,
    use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
    column_mode="hydrotel", et_mode="linacre", use_temperature=False,
    use_latent_codes=False, latent_mode="additive", spatial_melt=True,
    routing_mode="operator-lagged", predict_lake_params=True, compile_soil=True,
    use_aquifer=True).to(DEV)
m.eval(); m.spatial_encoder.init_from_literature({})
_cs = load_calibrated_soil(PROJ, node_ids, 0.15, device=DEV)
_LIBRE = os.environ.get("SWEEP_KREC_LIBRE", "1") == "1"
# ATTENTION : le champion aq30 (0.7880) a tourne AVANT le correctif du 2026-08-17,
# donc avec krec IMPOSE (25 champs, regle sans aquifere) et un aquifere affame.
# SWEEP_KREC_LIBRE=0 reproduit cette recette-la ; =1 libere la recharge (recette
# actuelle du pilote, 23 champs).
m.vertical_column.set_calibrated_soil(imposed_retention_curve(_cs, _LIBRE))
print(f"[balayage] ancrage : krec {'LIBRE' if _LIBRE else 'IMPOSE (recette du champion)'} "
      f"-> {len(imposed_retention_curve(_cs, _LIBRE))} champs imposes", flush=True)
m.vertical_column.set_linacre_params(*load_linacre_nodes(PROJ, node_ids, device=DEV))
m.vertical_column.set_melt_params(load_melt_nodes(PROJ, node_ids, device=DEV))
m.vertical_column.t_neige_seuil = load_passage_pluie_neige(PROJ)
_lc = load_occupation_sol(PROJ, node_ids, device=DEV)
_lc.update(load_milieux_humides(PROJ, node_ids, device=DEV))
m.vertical_column.set_land_cover(_lc)
m.vertical_column.set_phenology(load_phenologie(PROJ) or None)
m.set_hgm_kernel(torch.tensor(lire_hgm(PROJ, node_ids), device=DEV))
set_lake_area_from_hydrolakes(m, REG, r["territorial"].get_physical("area_km2_local"), n)
m.load(CKPT)
vc = m.vertical_column
lo, hi = vc._krec_bounds
krec_champion = float(torch.sigmoid(vc.krec_raw) * (hi - lo) + lo)
print(f"[balayage] {Path(CKPT).name} | krec appris = {krec_champion:.3e} "
      f"(bornes {lo:.0e}-{hi:.0e})", flush=True)

def poser_krec(v):
    f = min(max((v - lo) / (hi - lo), 1e-6), 1 - 1e-6)
    with torch.no_grad():
        vc.krec_raw.copy_(torch.tensor(math.log(f / (1 - f)), device=vc.krec_raw.device))

def poser_kgw(v):
    """Cible de vidange, en MEDIANE. k_gw est un CHAMP appris par le NeRF : on le
    met a l'echelle au lieu de l'ecraser par une constante, sinon on ne teste plus
    la vidange mais la perte de toute la structure spatiale (et le controle du
    champion ne se reproduit plus : 0.5811 au lieu de 0.7880, mesure du 2026-08-19).
    v = None laisse le champ appris intact."""
    m._kgw_cible = None if v is None else float(v)
    m._kgw_facteur = None

# crochet : met k_gw a l'echelle juste avant l'appel a l'aquifere
_aq_brut = vc.aquifer.forward
_vu = {}
def _aq(recharge, S_gw, k_gw, gw_withdrawal=None):
    _vu["med"] = float(k_gw.median())
    cible = getattr(m, "_kgw_cible", None)
    if cible is not None:
        if getattr(m, "_kgw_facteur", None) is None:
            m._kgw_facteur = cible / max(_vu["med"], 1e-9)
        k_gw = k_gw * m._kgw_facteur
    return _aq_brut(recharge, S_gw, k_gw, gw_withdrawal=gw_withdrawal)
vc.aquifer.forward = _aq

def kge(o, s):
    """MEME fonction que le pilote d'entrainement (meandre.utils.metrics.kge) : une
    metrique maison qui differe de quelques millemes suffit a rendre un banc
    incomparable au journal du champion."""
    v = np.isfinite(o) & np.isfinite(s)
    if v.sum() < 60: return np.nan
    return float(kge_fn(torch.tensor(o[v]), torch.tensor(s[v])))

def essai(krec, kgw):
    if krec is not None:
        poser_krec(krec)
    poser_kgw(kgw)
    with torch.no_grad():
        Q, _, D = m.simulate(forcing=td.forcing[:, :, :6], initial_state=HydroState.zeros(n, device=DEV),
                             graph=td.graph, node_coords=td.node_coords, territorial=r["territorial"],
                             withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                             return_diagnostics=True)
    qm = Q[mskt][:, td.station_idx].cpu().numpy()
    nT = min(len(qo), len(qm))
    k = np.array([kge(qo[:nT, i], qm[:nT, i]) for i in range(qm.shape[1])])
    fev = mois[:nT] == 2
    rfev = np.nanmedian(np.nansum(qm[:nT][fev], 0) / np.clip(np.nansum(qo[:nT][fev], 0), 1e-9, None))
    rech_an = float(D.recharge[mskt].mean().cpu()) * 365.25
    base_part = float(D.q_baseflow[mskt].mean().cpu()) / max(float(D.lateral_mm[mskt].mean().cpu()), 1e-9)
    return np.nanmedian(k), rfev, rech_an, base_part

print(f"\n{'krec':>9s} {'k_gw':>8s} {'KGE med':>8s} {'fev/obs':>8s} {'recharge':>10s} {'part base':>10s}")
print(f"{'':>9s} {'/j':>8s} {'':>8s} {'(cible 1)':>8s} {'mm/an':>10s} {'%':>10s}")
# CONTROLE : on ne touche A RIEN (krec appris, champ k_gw appris). Si cette ligne
# ne rend pas 0.7880, la recette diverge encore et le reste ne veut rien dire.
ref = essai(None, None)
print(f"{krec_champion:9.1e} {_vu.get('med', float('nan')):8.4f} {ref[0]:8.4f} {ref[1]:8.3f} "
      f"{ref[2]:10.2f} {100*ref[3]:10.1f}   <- CHAMPION INTACT (cible 0.7880 / fev 0.688)")
for kgw in (0.0273,):
    for krec in (8e-6, 1.2e-5, 1.6e-5, 2e-5, 2.5e-5, 3e-5, 4e-5):
        v = essai(krec, kgw)
        print(f"{krec:9.1e} {kgw:8.4f} {v[0]:8.4f} {v[1]:8.3f} {v[2]:10.2f} {100*v[3]:10.1f}", flush=True)
