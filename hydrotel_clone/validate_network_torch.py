"""Équivalence du routeur réseau TORCH (différentiable, vectorisé par niveau)
contre la version numpy déjà validée vs C++, sur DELISLE. Même apport_lateral,
compare debit_aval à l'exutoire + gradient non nul (différentiabilité).

  python hydrotel_clone/validate_network_torch.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.network_routing import route_network, delisle_network, topo_levels
from hydrotel_clone.network_routing_torch import route_network_torch

DEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE"
W = DEL + "/simulation/simulation/resultat"


def rcols(name, h=2):
    L = open(f"{W}/{name}.csv", encoding="latin-1").read().splitlines()
    ids = [int(x) for x in L[1].split(";")[1:]]
    arr = np.array([[float(x) for x in ln.split(";")[1:]] for ln in L[h:] if len(ln.split(";")) > 1])
    return arr, ids


reaches, downstream, topo, idx_of_id = delisle_network(DEL)
nR = len(reaches)
levels = topo_levels(downstream, nR)

aplat, ids_a = rcols("apport_lateral")
T = min(500, len(aplat))
apl_reach = np.zeros((T, nR))
for col, tid in enumerate(ids_a):
    if tid in idx_of_id:
        apl_reach[:, idx_of_id[tid]] = aplat[:T, col]

# numpy de référence
out_np = route_network(reaches, downstream, topo, apl_reach)

# dict de tenseurs pour la version torch
def col(key, default=0.0):
    return torch.tensor([float(r.get(key, default)) for r in reaches], dtype=torch.float64)

P = {
    "is_river": torch.tensor([r["type"] == "river" for r in reaches]),
    "lng": col("lng", 1.0), "lrg": col("lrg", 1.0), "pte": col("pte", 0.0025),
    "man": col("man", 0.04), "surface_m2": col("surface_m2", 1.0),
    "c": col("c", 1.0), "k": col("k", 1.0),
}
ds_t = torch.tensor(downstream, dtype=torch.long)
lg_t = [torch.tensor(g, dtype=torch.long) for g in levels]
apl_t = torch.tensor(apl_reach, dtype=torch.float64, requires_grad=True)

out_t = route_network_torch(P, ds_t, lg_t, apl_t)
out_t_np = out_t.detach().numpy()

oidx = idx_of_id[1]
d = out_t_np[:, oidx] - out_np[:, oidx]
print(f"EXUTOIRE torch vs numpy : RMSE {np.sqrt(np.mean(d**2)):.6e}  max abs {np.max(np.abs(d)):.6e}")
dall = out_t_np - out_np
print(f"TOUS tronçons          : max abs {np.max(np.abs(dall)):.6e}")

# différentiabilité : gradient de la somme du débit exutoire vs apport_lateral
loss = out_t[:, oidx].sum()
loss.backward()
g = apl_t.grad
print(f"gradient : non-nuls {int((g != 0).sum())}/{g.numel()}  |g|max {g.abs().max():.4f}  any-nan {bool(torch.isnan(g).any())}")
print("DONE")
