"""Controle des unites de l'ancrage geometrique sur une region reelle, sans entrainement.

Le 2026-09-15, la tache de controle sur Narval a refuse de poser l'ancre en annoncant une
longueur de troncon absente. La donnee etait pourtant presente : la table des aretes de la
base du Saguenay porte 2 211 troncons de 7 141 m de longueur mediane. C'est la lecture qui
echouait, et le message ne nommait pas la cause.

Ce script refait le calcul de l'ancre exactement comme le pilote, sur les fichiers de la
region, et annonce les grandeurs attendues : longueur mediane vers 7 km, ancre mediane vers
2 h. Il coute quelques secondes et se lance avant toute soumission.

    .venv/Scripts/python.exe .runs/quebec/verifier_ancrage_region.py sagu
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import duckdb
import pandas as pd
import torch

from meandre.spatial.field_network import SpatialFieldNetwork
from meandre.utils import paths as _paths


def main(reg):
    racine = _paths.DATA_ROOT
    base = f"{racine}/quebec/{reg}.duckdb"
    if not os.path.exists(base):
        print(f"base introuvable : {base}")
        return 1
    cx = duckdb.connect(base, read_only=True)
    n_nodes = cx.sql("select count(*) from nodes").fetchone()[0]
    ed = cx.sql("select src, edge_attr_0 from edges").fetchdf()
    lac = torch.tensor(cx.sql("select is_lake from nodes order by node_idx")
                       .fetchdf()["is_lake"].values.astype(bool))
    cx.close()

    lg = torch.zeros(n_nodes)
    lg[torch.tensor(ed["src"].values, dtype=torch.long)] = torch.tensor(
        ed["edge_attr_0"].values / 1000.0, dtype=torch.float32)
    nz = int((lg <= 0).sum())
    riv = (lg > 0) & (~lac)
    med = float(lg[riv].median())
    lg = torch.where(lg > 0, lg, torch.full_like(lg, med))
    lg = torch.where(lac, torch.full_like(lg, float("nan")), lg)
    print(f"{reg.upper()} : {n_nodes} nœuds, {len(ed)} arêtes, {nz} sans arête sortante, "
          f"{int(lac.sum())} lacs laissés libres")
    lr = lg[torch.isfinite(lg)]
    print(f"  longueur de rivière : méd {med:.2f} km | q10-q90 "
          f"{float(lr.quantile(0.1)):.2f}-{float(lr.quantile(0.9)):.2f} km "
          f"| max {float(lr.max()):.1f} km")

    pente = None
    rw = pd.read_parquet(f"{racine}/quebec/territorial-raw-QC.parquet")
    rw = rw[rw.region == reg]
    if len(rw) == n_nodes:
        pente = torch.tensor(rw["mean_slope_pct"].values / 100.0, dtype=torch.float32)
        print(f"  pente de versant : méd {float(pente.median()) * 100:.2f} % | q10-q90 "
              f"{float(pente.quantile(0.1)) * 100:.2f}-{float(pente.quantile(0.9)) * 100:.2f} %")
    else:
        print(f"  pente de versant ABSENTE : {len(rw)} lignes contre {n_nodes} nœuds")

    f = SpatialFieldNetwork(n_territorial=16, n_nodes=n_nodes)
    f.set_routing_anchor(lg, slope_frac=pente)
    ka = f._k_musk_anchor
    ka = ka[torch.isfinite(ka)]
    print(f"  ancre : méd {float(ka.median()):.2f} h | q10-q90 "
          f"{float(ka.quantile(0.1)):.2f}-{float(ka.quantile(0.9)):.2f} h "
          f"| bornes atteintes {int((ka <= 0.0501).sum())} bas, {int((ka >= 47.99).sum())} haut")
    ok = 0.3 < float(ka.median()) < 8.0 and 1.0 < med < 30.0
    ok = ok and int((ka <= 0.0501).sum()) < 0.05 * len(ka)
    print("\nCONTRÔLE RÉUSSI" if ok else "\nCONTRÔLE ÉCHOUÉ : unités invraisemblables")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu"))
