"""Épreuves d'intégration des couches de l'IRDA au champ spatial, sans simulation.

Trois questions, dans l'ordre où une réponse négative arrête la suivante.

1. Construction. Les attributs de l'IRDA se rangent-ils par nœud, sous forme de composition
   (classes de drainage, matériau parental, affleurement, part non cartographiée) ?
2. Départ à chaud. Ajouter ces colonnes à l'entrée du champ perturbe-t-il les paramètres d'un
   modèle déjà entraîné ? Comparaison entre le rembourrage du chargeur, aléatoire à petite
   échelle, et un rembourrage à zéro.
3. Gradient. Les nouveaux poids reçoivent-ils un gradient non nul ?

    .venv/Scripts/python.exe .runs/quebec/test_irda_integration.py mont
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.basin_cache import BasinCache
from meandre.model import HydroModel
from meandre.utils import paths as _paths

MODELES = f"{_paths.DATA_ROOT}/quebec/checkpoints-ref-2026-09-15"
CLASSES = ["drainage_Excessif", "drainage_Bon", "drainage_Imparfait", "drainage_Mauvais", "drainage_Très mauvais", "materiau_ARGILE", "materiau_TILL", "materiau_SABLE", "materiau_LIMON", "materiau_GRAVIER", "materiau_ORGANIQUE", "roc_AFFLEUREMENT"]


def attributs_irda(reg):
    """Composition IRDA par nœud, dans l'ordre des nœuds de la base régionale."""
    ir = pd.read_parquet(f"{_paths.DATA_ROOT}/irda/irda-troncons.parquet")
    ir = ir[ir.region == reg].set_index("troncon")
    cx = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
    nd = cx.sql("select node_idx, node_id from nodes order by node_idx").fetchdf()
    cx.close()
    x = np.zeros((len(nd), len(CLASSES) + 1), dtype=np.float32)
    for i, tid in zip(nd.node_idx.values, nd.node_id.values):
        if tid in ir.index:
            ligne = ir.loc[tid]
            x[i, :-1] = [float(ligne.get(c, 0.0)) for c in CLASSES]
    # La part non cartographiée ferme la composition du drainage.
    x[:, -1] = np.clip(1.0 - x[:, :5].sum(axis=1), 0.0, 1.0)
    return x


def parametres(model, coords, territorial):
    with torch.no_grad():
        sp = model.spatial_encoder(coords, territorial)
    return torch.stack([getattr(sp, f) for f in sp.__dataclass_fields__ if torch.is_tensor(getattr(sp, f)) and getattr(sp, f).shape[:1] == coords.shape[:1]], dim=1)


def main(reg):
    ck = f"{MODELES}/best-{reg}-etl-ref-variations.pt"
    d = BasinCache(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb").load(torch.device("cpu"))
    coords, terr = d["node_coords"], d["territorial"].to_tensor()
    x = torch.tensor(attributs_irda(reg))
    print(f"1. construction : {x.shape[0]} nœuds, {x.shape[1]} colonnes IRDA")
    print(f"   somme du drainage : médiane {float(x[:, :5].sum(1).median()):.3f}, maximum {float(x[:, :5].sum(1).max()):.3f}")
    print(f"   nœuds couverts à plus de 50 % : {int((x[:, :5].sum(1) > 0.5).sum())}")

    base = HydroModel.from_checkpoint(ck, compile_soil=False)
    p0 = parametres(base, coords, terr)
    etendu = torch.cat([terr, x], dim=1)
    torch.manual_seed(0)
    alea = HydroModel.from_checkpoint(ck, compile_soil=False, n_territorial=terr.shape[1] + x.shape[1])
    p1 = parametres(alea, coords, etendu)
    zero = HydroModel.from_checkpoint(ck, compile_soil=False, n_territorial=terr.shape[1] + x.shape[1])
    n0 = terr.shape[1]
    with torch.no_grad():
        # Les colonnes territoriales suivent l'encodage de position dans l'entrée du tronc.
        debut = zero.spatial_encoder.fc1.in_features - x.shape[1]
        zero.spatial_encoder.fc1.weight[:, debut:] = 0.0
        zero.spatial_encoder.fc2.weight[:, -x.shape[1]:] = 0.0
    p2 = parametres(zero, coords, etendu)
    ech = p0.abs().clamp_min(1e-6)
    ecart_alea = ((p1 - p0).abs() / ech).median(dim=0).values
    ecart_zero = ((p2 - p0).abs() / ech).max()
    print("2. départ à chaud, écart relatif des paramètres au modèle d'origine")
    print(f"   rembourrage du chargeur : médiane sur paramètres {float(ecart_alea.median()):.2e}, pire paramètre {float(ecart_alea.max()):.2e}")
    print(f"   rembourrage à zéro : écart maximal {float(ecart_zero):.2e}")

    zero.spatial_encoder.zero_grad()
    sortie = zero.spatial_encoder(coords, etendu)
    cible = sum(getattr(sortie, f).sum() for f in ("K_sat_1", "porosity_1", "K_musk_hours"))
    cible.backward()
    g = zero.spatial_encoder.fc1.weight.grad
    print("3. gradient sur les poids des colonnes IRDA")
    print(f"   norme {float(g[:, debut:].norm()):.3e} contre {float(g[:, :debut].norm()):.3e} pour les autres entrées")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1].lower() if len(sys.argv) > 1 else "mont"))
