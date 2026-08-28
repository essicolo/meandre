"""JUGE NIVAL autonome : noter un checkpoint contre CanSWE, sans reentrainer.

Motif (2026-08-28). R56 pose que le KGE seul ne peut pas valider un changement du module
de fonte : un verrou qui bouge peut acheter du debit en deplacant de l'eau au mauvais
moment, ce qui serait une compensation de plus, exactement ce que le projet cherche a
eviter. Le juge annonce etait la neige. Or province.py ne l'imprime pas, et cabler le
diagnostic dans la boucle d'entrainement le ferait payer a tous les runs. On le sort donc
en script, applicable a n'importe quel checkpoint deja produit.

APPARIEMENT SITE ET JOUR, jamais de climatologie de reseau. Les sites qui rapportent
changent d'un jour a l'autre et le rapport pic sur neige tombee correle a +0.58 avec
l'altitude du site : une moyenne du reseau retient donc le jour ou les sites les plus
enneiges rapportent. C'est l'erreur R28, payee trois fois. Ici chaque mesure est comparee
au noeud ET au jour qui lui correspondent, et rien d'autre.

  python .runs/quebec/juge_neige.py <plateforme> <tag0> [tag1 ...]

Exemple : python .runs/quebec/juge_neige.py sagu canopee-sagu-0 canopee-sagu-1
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".runs/quebec")
sys.path.insert(0, ".")

MOIS = (10, 11, 12, 1, 2, 3, 4, 5)


def juger(reg: str, tag: str, dom, mes):
    from meandre.model import HydroModel

    ck_path = f".runs/quebec/checkpoints/best-{tag}.pt"
    if not os.path.exists(ck_path):
        print(f"[{tag}] checkpoint absent : {ck_path}")
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    td = dom["train_data"]
    modele = HydroModel(
        n_territorial=int(dom["territorial"].shape[1]), n_nodes=int(dom["n_nodes"]),
        use_latent_codes=True, spatial_melt=True,
        canopy_melt_lag=bool(ck.get("canopy_melt_lag", False))).to(dev)
    sd = ck.get("state_dict", ck)
    own = modele.state_dict()
    for k in [k for k, v in sd.items() if k in own and own[k].shape != v.shape]:
        sd.pop(k)
    modele.load_state_dict(sd, strict=False)
    modele.eval()
    with torch.no_grad():
        out = modele(td.forcing, td.graph, td.node_coords, td.territorial,
                     withdrawals=td.withdrawals, day_of_year=td.day_of_year,
                     return_diagnostics=True)
    diag = out[-1] if isinstance(out, (tuple, list)) else out
    swe = diag.swe.detach().cpu().numpy()

    m = mes.copy()
    m["sim"] = swe[m.t.astype(int), m.node_idx.astype(int)]
    lignes = {}
    for x in MOIS:
        sub = m[m.mois == x]
        if len(sub) < 30:
            continue
        lignes[x] = (sub.sim.mean(), sub.swe_mm.mean(), len(sub))
    return lignes


def main():
    from domain_data import load_domain
    from meandre.data.basin_cache import BasinCache
    from meandre.utils import paths as _paths

    reg = sys.argv[1].lower()
    tags = sys.argv[2:]
    if not tags:
        raise SystemExit("usage : juge_neige.py <plateforme> <tag> [tag ...]")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dom = load_domain([reg], {}, device=dev)

    bc = BasinCache(_paths.data_path("quebec", f"{reg}.duckdb"))
    mes, _ = bc.load_canswe("2000-01-01", "2024-12-31")
    if mes is None or len(mes) == 0:
        raise SystemExit(f"[{reg}] aucune mesure CanSWE")
    pos = pd.Series(np.arange(len(dom["times"])),
                    index=pd.DatetimeIndex(dom["times"]).normalize())
    mes = mes.copy()
    mes["t"] = pos.reindex(pd.DatetimeIndex(mes.date).normalize()).to_numpy()
    mes = mes[np.isfinite(mes.t) & np.isfinite(mes.swe_mm) & (mes.swe_mm >= 0)]
    mes["mois"] = pd.DatetimeIndex(mes.date).month
    print(f"[juge] {reg} : {len(mes):,} mesures CanSWE appariees site et jour")

    res = {t: juger(reg, t, dom, mes) for t in tags}
    res = {k: v for k, v in res.items() if v}
    if not res:
        raise SystemExit("aucun checkpoint lisible")

    entete = "".join(f"{x:7d}" for x in MOIS)
    ref = next(iter(res.values()))
    print(f"\n  mois          {entete}")
    print("  CanSWE (mm)   " + "".join(f"{ref[x][1]:7.0f}" if x in ref else "      ." for x in MOIS))
    for t, l in res.items():
        print(f"  {t[:12]:>12}  " + "".join(f"{l[x][0]:7.0f}" if x in l else "      ." for x in MOIS))
    print("\n  RAPPORT simule / mesure (1.00 = juste)")
    for t, l in res.items():
        print(f"  {t[:12]:>12}  " + "".join(
            f"{l[x][0] / l[x][1]:7.2f}" if x in l and l[x][1] > 1 else "      ." for x in MOIS))
    print("\n  n mesures     " + "".join(f"{ref[x][2]:7d}" if x in ref else "      ." for x in MOIS))
    print("\nLECTURE. Un rapport proche de 1 sur novembre-mars dit que la MASSE est juste ;")
    print("un rapport qui s'effondre en avril-mai dit que la fonte part trop tot. Le verrou")
    print("de canopee agit sur la DATE, donc c'est la marche entre mars et mai qu'il faut")
    print("regarder, pas la moyenne annuelle.")


if __name__ == "__main__":
    main()
