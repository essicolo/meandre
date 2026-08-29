"""BISSECTION de la regression provinciale : d'ou vient la chute de 0.62 a 0.45 ?

FAIT A EXPLIQUER. Le run provincial du 2026-08-26 rendait 0.6193 de KGE median tenu de
cote sur 141 stations, a trois epoques. Les runs du 2026-08-28, a trois epoques aussi,
rendent 0.4266 a 0.4513. Par plateforme :

              nos runs   run du 26 aout   modeles regionaux gen1
    outv        0.686        0.652              --
    slno        0.597        0.700            0.763
    gasp        0.420        0.736            0.713
    slso        0.234        0.538              --
    sagu        0.233        0.674            0.744
    mont        0.216        0.482            0.695

Nous sommes tres en dessous PARTOUT SAUF sur OUTV, le seul territoire ou la campagne de
physique a ete menee.

CE QUE CE SCRIPT NE FAIT PAS. Il n'entraine rien et n'optimise rien. Trois configurations
d'hyperparametres testees pendant la nuit donnent des scores par plateforme identiques au
troisieme chiffre : la cause n'est pas la.

CE QU'IL FAIT. Il prend le point de reprise du 26 aout, celui de l'epoque ou le modele
valait 0.62, et le fait tourner en INFERENCE PURE sur les caches d'aujourd'hui, avec et
sans prelevements. Deux lectures possibles.

  1. Sans prelevements il remonte vers 0.62 : la regression vient de la REINGESTION du
     26 aout (commit 8e02622), qui a multiplie la couverture par deux et demi. Les
     plateformes qui s'effondrent -- mont, sagu, slso -- sont precisement celles qui
     portent le plus de sites : slso 1087, slno 1017, sagu 234. OUTV, epargne, en porte
     peu. La correlation est trop forte pour etre ignoree.

  2. Il reste a 0.45 dans les deux cas : ce n'est pas la donnee, et le suspect suivant
     est le champ thermique du gel (48fb393, actif par defaut via use_frost_rankinen).

Un point de reprise ne definit pas un modele (dette #6) : la recette est donc posee ici
exactement comme le pilote la pose, ancrages compris.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".runs/quebec")
sys.path.insert(0, ".")

CKPT = os.environ.get("BIS_CKPT", "D:/meandre-data/quebec/runpod/best-province.pt")
PLATS = (os.environ.get("BIS_PLATEFORMES") or "outv,gasp,mont,sagu,slno,slso").split(",")


def kge(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    if m.sum() < 365:
        return np.nan
    o, s = o[m], s[m]
    if o.std() < 1e-9 or s.std() < 1e-9:
        return np.nan
    r = float(np.corrcoef(o, s)[0, 1])
    b = float(s.mean() / o.mean())
    g = float((s.std() / s.mean()) / (o.std() / o.mean()))
    return 1.0 - float(np.sqrt((r - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2))


def main():
    from domain_data import load_domain
    from meandre.model import HydroModel
    from meandre.utils.state import HydroState

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    noms = [p.strip() for p in PLATS if p.strip()]
    dom = load_domain(noms, {}, device=dev)
    td, terr = dom["train_data"], dom["territorial"]

    modele = HydroModel(
        n_territorial=int(terr.n_features), n_nodes=int(dom["n_nodes"]),
        n_forcing=6, use_temporal=False, use_residual=False,
        param_mode="nerf", column_mode="hydrotel", et_mode="linacre",
        use_latent_codes=False, spatial_melt=True, use_aquifer=True,
        use_temperature=False, predict_lake_params=True,
        routing_mode="operator-lagged").to(dev)
    if dom.get("land_cover"):
        modele.vertical_column.set_land_cover(dom["land_cover"])
    if dom.get("melt_params"):
        modele.vertical_column.set_melt_params(dom["melt_params"])
    if dom.get("phenology"):
        modele.vertical_column.set_phenology(dom["phenology"])
    if dom.get("linacre"):
        modele.vertical_column.set_linacre_params(*dom["linacre"])
        modele.vertical_column.etp_channel = None
    if dom.get("kgw") is not None:
        _o = modele.spatial_encoder.forward
        def _kgw(*a, _o=_o, _k=dom["kgw"], **kw):
            sp = _o(*a, **kw)
            sp.k_gw = _k
            return sp
        modele.spatial_encoder.forward = _kgw
    modele.vertical_column.split_mode = "wet_bulb"
    modele.vertical_column.t_neige_seuil = -0.8
    if dom.get("soil"):
        modele.vertical_column.set_calibrated_soil(dom["soil"])
    modele.load(CKPT)
    modele.eval()

    times = pd.DatetimeIndex(dom["times"])
    sl = (times >= "2022-01-01") & (times <= "2024-12-31")
    n = int(sl.sum())
    q_obs = dom["val_data"].q_obs

    res = {}
    for nom, w in (("avec prelevements", td.withdrawals), ("SANS prelevements", None)):
        with torch.no_grad():
            Q, _ = modele.simulate(
                forcing=td.forcing[:],
                initial_state=HydroState.zeros(int(dom["n_nodes"]), device=dev),
                graph=td.graph, node_coords=td.node_coords, territorial=terr,
                withdrawals=w, day_of_year=td.day_of_year)
        par = {}
        for j, brut in enumerate(dom["station_ids"]):
            plat = str(brut).split(":")[0]
            node = int(td.station_idx[j])
            k = kge(q_obs[-n:, j].detach().cpu().numpy(),
                    Q[sl][:, node].detach().cpu().numpy())
            if np.isfinite(k):
                par.setdefault(plat, []).append(k)
        res[nom] = par
        tous = [x for v in par.values() for x in v]
        print(f"[{nom}] mediane {np.median(tous):.4f} sur {len(tous)} stations", flush=True)
        del Q
        torch.cuda.empty_cache()

    print(f"\n{'plateforme':>12} | {'avec':>8} | {'sans':>8} | {'ecart':>8} | n")
    print("-" * 56)
    for plat in noms:
        a = res["avec prelevements"].get(plat, [])
        b = res["SANS prelevements"].get(plat, [])
        if not a or not b:
            continue
        ma, mb = float(np.median(a)), float(np.median(b))
        print(f"{plat:>12} | {ma:8.4f} | {mb:8.4f} | {mb - ma:+8.4f} | {len(a)}")

    print("\nLECTURE. Un ecart franchement POSITIF, surtout sur mont, sagu et slso, dit que")
    print("la reingestion des prelevements du 26 aout porte la regression. Un ecart nul")
    print("partout la disculpe, et le suspect suivant est le champ thermique du gel.")


if __name__ == "__main__":
    main()
