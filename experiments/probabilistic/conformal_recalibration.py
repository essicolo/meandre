"""Recalibration conforme de la tête de quantiles, par territoire et par niveau.

La tête rend six quantiles q_k = Q exp(c_k), où Q est le débit simulé (la médiane) et c_k un écart
logarithmique, négatif sous la médiane et positif au-dessus. Pour chaque territoire et chaque niveau
tau_k, on cherche le facteur s_k tel que la part des observations sous Q exp(s_k c_k) vaille tau_k sur
la période de calibration. La médiane ne bouge pas ; seules les largeurs s'étirent ou se resserrent,
et chaque côté de l'enveloppe a ses propres facteurs. Un territoire de moins de MIN_DAYS jours-stations
de calibration prend les facteurs du Québec entier.

Les facteurs sont ajustés sur une période et jugés sur une autre, jamais vue : la couverture est
imprimée avant et après, par territoire, sur la période d'évaluation.

Usage : python conformal_recalibration.py <q sur 2001-2024 avec enveloppe .npz> <sortie des facteurs .csv>
            [--calibration 2019-01-01,2021-12-31] [--evaluation 2022-01-01,2024-12-31]
"""
import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("series")
parser.add_argument("output")
parser.add_argument("--calibration", default="2019-01-01,2021-12-31")
parser.add_argument("--evaluation", default="2022-01-01,2024-12-31")
args = parser.parse_args()
MIN_DAYS = 3000

z = np.load(args.series, allow_pickle=True)
dates = pd.to_datetime([str(d) for d in z["dates"]])
taus = np.asarray(z["quantile_taus"], dtype=float)
simulated, observed = z["q_sim"], z["q_obs"]
offsets = np.log(np.maximum(simulated[..., None] + z["q_quantiles"], 1e-9) / np.maximum(simulated[..., None], 1e-9))
territory = np.array([str(s).split(":")[0] for s in z["station_ids"]])
log_ratio = np.log(np.maximum(observed, 1e-9) / np.maximum(simulated, 1e-9))
usable = np.isfinite(observed) & (observed > 0) & (simulated > 0) & np.isfinite(offsets).all(axis=-1)


def window(text):
    start, end = text.split(",")
    return (dates >= start) & (dates <= end)


calibration, evaluation = window(args.calibration), window(args.evaluation)


def scale_for(ratio, offset, tau):
    """Facteur s tel que la part de ratio < s * offset vaille tau ; recherche par dichotomie."""
    low, high = 0.05, 20.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        share = np.mean(ratio < middle * offset)
        # Sous la médiane (offset < 0), étirer baisse la part ; au-dessus, étirer l'augmente.
        too_low_share = share < tau
        if (offset.mean() < 0) == too_low_share:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


groups = ["quebec"] + sorted(set(territory))
factors = {}
for group in groups:
    stations = np.ones(len(territory), bool) if group == "quebec" else territory == group
    mask = usable[calibration][:, stations]
    if mask.sum() < MIN_DAYS and group != "quebec":
        factors[group] = factors["quebec"]
        continue
    ratio = log_ratio[calibration][:, stations][mask]
    factors[group] = [scale_for(ratio, offsets[calibration][:, stations][mask][:, k], tau) for k, tau in enumerate(taus)]
table = pd.DataFrame(factors, index=[f"tau_{t:.2f}" for t in taus]).T
table.to_csv(args.output)
print("facteurs d'étirement par territoire (1 = inchangé)")
print(table.round(2).to_string())


def coverage(stations, scale):
    mask = usable[evaluation][:, stations]
    ratio = log_ratio[evaluation][:, stations][mask]
    c = offsets[evaluation][:, stations][mask] * np.asarray(scale)[None, :]
    k05, k25, k75, k95 = list(taus).index(0.05), list(taus).index(0.25), list(taus).index(0.75), list(taus).index(0.95)
    return dict(n=int(mask.sum()), cov50=100 * np.mean((ratio > c[:, k25]) & (ratio < c[:, k75])), cov90=100 * np.mean((ratio > c[:, k05]) & (ratio < c[:, k95])), above=100 * np.mean(ratio > c[:, k95]), below=100 * np.mean(ratio < c[:, k05]))


rows = []
for group in groups:
    stations = np.ones(len(territory), bool) if group == "quebec" else territory == group
    before = coverage(stations, np.ones(len(taus)))
    if group == "quebec":
        # Au Québec entier, chaque station reçoit les facteurs de son territoire.
        mask = usable[evaluation]
        ratio_all, kept = [], []
        hits = {"cov50": 0, "cov90": 0, "above": 0, "below": 0, "n": 0}
        for t in sorted(set(territory)):
            r = coverage(territory == t, factors[t])
            for key in ("cov50", "cov90", "above", "below"):
                hits[key] += r[key] * r["n"]
            hits["n"] += r["n"]
        after = {key: (hits[key] / hits["n"] if key != "n" else hits["n"]) for key in hits}
    else:
        after = coverage(stations, factors[group])
    rows.append(dict(territoire=group, jours=before["n"], cov90_avant=before["cov90"], cov90_apres=after["cov90"], cov50_avant=before["cov50"], cov50_apres=after["cov50"], au_dessus_avant=before["above"], au_dessus_apres=after["above"], dessous_avant=before["below"], dessous_apres=after["below"]))
pd.set_option("display.width", 250)
print(f"\ncouverture sur l'évaluation {args.evaluation}, facteurs ajustés sur {args.calibration}")
print(pd.DataFrame(rows).round(1).to_string(index=False))
