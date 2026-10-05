"""Trajectoire d'une sortie du champ au fil des époques, moyennée sur le bassin amont de stations.

Question (2026-10-05) : le champ régional part de K_c = 1,0 partout et finit à 0,60 au nord
alors qu'à la dernière époque toutes les cibles de la perte poussent K_c vers le haut dans
ces bassins (R307). Lire le champ de chaque point de reprise dit quand et comment il est
descendu, sans aucune simulation.

    python .runs/quebec/trajectoire_champ.py <base.duckdb> <sortie> <station,station,...> <point de reprise>...
"""
import sys
import os

sys.path.insert(0, ".runs/quebec")
sys.path.insert(0, ".")
import numpy as np
import torch

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel


def bassin_amont(edge_index, n, exutoire):
    """Nœuds dont l'eau passe par l'exutoire, exutoire compris."""
    amont = [[] for _ in range(n)]
    for s, t in edge_index.T.tolist():
        amont[t].append(s)
    vus, pile = {exutoire}, [exutoire]
    while pile:
        for s in amont[pile.pop()]:
            if s not in vus:
                vus.add(s)
                pile.append(s)
    return sorted(vus)


def main():
    base, sortie, stations = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    points = sys.argv[4:]
    d = BasinCache(base).load(device=torch.device("cpu"))
    terr, coords, graph = d["territorial"], d["node_coords"], d["graph"]
    n = coords.shape[0]
    import duckdb
    con = duckdb.connect(base, read_only=True)
    idx = dict(con.execute("SELECT station_id, node_idx FROM stations").fetchall())
    con.close()
    bassins = {s: bassin_amont(graph.edge_index, n, int(idx[s])) for s in stations}
    print(f"{n} noeuds, {terr.data.shape[1]} attributs ; bassins : "
          + ", ".join(f"{s} {len(b)} noeuds" for s, b in bassins.items()), flush=True)
    m = HydroModel(n_nodes=n, n_territorial=terr.data.shape[1], n_forcing=6,
                   use_temporal=False, use_residual=False, use_travel_time_attn=False,
                   use_frost_rankinen=True, column_theta_init_frac=0.9, param_mode="nerf",
                   column_mode="hydrotel", et_mode="mcguinness", use_temperature=False,
                   use_latent_codes=False, predict_lake_params=True)
    print("point de reprise".ljust(36) + "  ".join(f"{s:>8}" for s in stations) + "   domaine", flush=True)
    for ck in points:
        m.load(ck)
        m.eval()
        with torch.no_grad():
            v = getattr(m.spatial_encoder(coords, terr.to_tensor()), sortie).reshape(-1)[:n].double().numpy()
        print(os.path.basename(ck)[:34].ljust(36) + "  ".join(f"{v[b].mean():8.3f}" for b in bassins.values())
              + f"   {v.mean():7.3f}", flush=True)


if __name__ == "__main__":
    main()
