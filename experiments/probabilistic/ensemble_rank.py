"""Forme de l'enveloppe : ensemble de forçage PyGMET contre tête de quantiles, aux stations d'un
territoire, sur la période d'évaluation.

Trois enveloppes sont jugées par le diagramme de rangs (part des observations dans chaque classe,
attendue plate) et par la couverture de leur intervalle le plus large :
- la tête de quantiles seule, sur la passe de référence (intervalle 5-95 %) ;
- l'ensemble de forçage seul, N membres (intervalle min-max, nominal (N-1)/(N+1)) ;
- la combinaison, enveloppe relative de la tête appliquée à chaque membre, puis min des bas et max
  des hauts.

Usage : python ensemble_rank.py <q-ref.npz avec enveloppe> <q-ens*.npz ...>
"""
import sys

import numpy as np

ref = np.load(sys.argv[1], allow_pickle=True)
members = [np.load(f, allow_pickle=True) for f in sys.argv[2:]]
obs = ref["q_obs"]
sim = ref["q_sim"]
quantiles = sim[..., None] + ref["q_quantiles"]
ensemble = np.stack([m["q_sim"] for m in members])
assert all(list(m["station_ids"]) == list(ref["station_ids"]) for m in members)
valid = np.isfinite(obs) & (obs > 0) & np.isfinite(sim)
n = len(members)
print(f"{n} membres, {valid.sum()} jours-stations, {ref['dates'][0]} à {ref['dates'][-1]}")


def rank_histogram(label, lower_edges, upper_edge=None):
    """Part des observations sous chaque borne, puis classes du diagramme de rangs."""
    below = np.stack([(obs < b)[valid].mean() for b in lower_edges])
    classes = np.diff(np.concatenate([[0.0], below, [1.0]]))
    print(f"\n{label}")
    print("  part sous chaque borne :", np.round(below, 3))
    print("  classes du diagramme (%) :", np.round(100 * classes, 1), f"| attendu {100 / len(classes):.1f} chacune")
    print(f"  première classe {100 * classes[0]:.1f} %, dernière {100 * classes[-1]:.1f} %")


rank_histogram("tête de quantiles seule (bornes 5, 10, 25, 75, 90, 95 %)", [quantiles[..., k] for k in range(quantiles.shape[-1])])
inside_head = ((obs > quantiles[..., 0]) & (obs < quantiles[..., -1]))[valid].mean()
print(f"  couverture de l'intervalle 5-95 % : {100 * inside_head:.1f} % pour 90 attendus")

ordered = np.sort(ensemble, axis=0)
rank_histogram(f"ensemble de forçage seul ({n} membres ordonnés)", [ordered[k] for k in range(n)])
inside_ensemble = ((obs > ordered[0]) & (obs < ordered[-1]))[valid].mean()
print(f"  couverture min-max : {100 * inside_ensemble:.1f} % pour {100 * (n - 1) / (n + 1):.1f} attendus")
spread = ensemble.std(axis=0)[valid].mean()
error = np.abs(ensemble.mean(axis=0) - obs)[valid].mean()
print(f"  dispersion moyenne entre membres {spread:.2f} m³/s, erreur absolue moyenne de la moyenne d'ensemble {error:.2f} m³/s")

relative_low = quantiles[..., 0] / np.maximum(sim, 1e-6)
relative_high = quantiles[..., -1] / np.maximum(sim, 1e-6)
combined_low = (ordered[0] * relative_low)
combined_high = (ordered[-1] * relative_high)
inside_combined = ((obs > combined_low) & (obs < combined_high))[valid].mean()
print(f"\ncombinaison, tête relative sur chaque membre : couverture {100 * inside_combined:.1f} % ; part sous la borne basse {100 * (obs < combined_low)[valid].mean():.1f} %, au-dessus de la haute {100 * (obs > combined_high)[valid].mean():.1f} %")
