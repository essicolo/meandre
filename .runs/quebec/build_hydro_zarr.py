"""Store zarr des hydrogrammes pour la carte feuillage, avec l'incertitude.

Contrat feuillage (README, section chroniques zarr) : un value_array indexe par une
feature_dim appariee a une propriete GeoJSON, un time_array, et des indexers pour les
dimensions supplementaires -- ici `percentile`. L'incertitude est donc une DIMENSION du
store : les taus reellement appris par la tete quantile, plus 50, la mediane etant le
debit physique simule. Aucune interpolation entre taus n'est fabriquee.

Correction d'Essi (2026-08-25) : « il n'y a pas de membres, seulement un modele
probabiliste ». La version precedente ajoutait des percentiles 0 et 100 pris comme
min/max de dix runs PyGMET -- c'etait traiter l'incertitude de forcage par REPETITION
EXTERNE, dix modeles deterministes recolles apres coup. Une enveloppe min/max de dix
tirages n'est d'ailleurs pas un quantile : sa largeur croit avec le nombre de membres.
L'incertitude de forcage doit entrer comme distribution d'ENTREE d'un modele unique,
et ressortir par la meme tete quantile, en une passe. Les membres PyGMET ne servent
plus qu'a calibrer et verifier cette entree, jamais a fabriquer la sortie livree.

Sortie : D:/meandre-data/quebec/carte/hydro.zarr, consolide, chunks par station pour
que chaque popup ne tire qu'un chunk (~100 ko).

    .venv/Scripts/python.exe .runs/quebec/build_hydro_zarr.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import zarr

from meandre.utils import paths as _paths

RESULTS = f"{_paths.DATA_ROOT}/quebec/results"
STORE = f"{_paths.DATA_ROOT}/quebec/carte/hydro.zarr"


def main():
    stations, dates_ref = [], None
    series = {}     # sid -> dict(obs, sim, quantiles?(T,K), taus?)
    for fq in sorted(glob.glob(f"{RESULTS}/nb-*-q.npz")):
        d = np.load(fq, allow_pickle=True)
        if dates_ref is None:
            dates_ref = d["dates"]
        if len(d["dates"]) != len(dates_ref):
            print(f"  {os.path.basename(fq)} ignore (axe temps different)")
            continue
        for j, sid in enumerate(d["station_ids"]):
            e = series.setdefault(str(sid), {})
            e["obs"] = d["q_obs"][:, j]
            e["sim"] = d["q_sim"][:, j]
            if "q_quantiles" in d.files:
                e["quantiles"] = d["q_quantiles"][:, j, :]
                e["taus"] = d["quantile_taus"]

    if not series:
        print("aucun cache nb-*-q.npz : rien a exporter")
        return
    sids = sorted(series)
    T = len(dates_ref)
    taus = None
    for e in series.values():
        if "taus" in e:
            taus = [float(x) for x in e["taus"]]
            break
    # percentiles du store : les taus appris par la tete, plus la mediane physique
    pcts = ([100.0 * t for t in taus] if taus else []) + [50.0]
    pcts = sorted(set(round(p, 1) for p in pcts))
    S, P = len(sids), len(pcts)
    disc = np.full((T, S, P), np.nan, dtype=np.float32)
    obs = np.full((T, S), np.nan, dtype=np.float32)
    i50 = pcts.index(50.0)
    for si, sid in enumerate(sids):
        e = series[sid]
        if "obs" in e:
            obs[:, si] = e["obs"]
        if "sim" in e:
            disc[:, si, i50] = e["sim"]
        if "quantiles" in e and taus:
            for k, t in enumerate(taus):
                disc[:, si, pcts.index(round(100.0 * t, 1))] = e["quantiles"][:, k]

    root = zarr.open_group(STORE, mode="w")
    root.create_array("discharge", data=disc, chunks=(T, 1, P))
    root.create_array("observed", data=obs, chunks=(T, 1))
    root.create_array("time", data=np.array([str(x) for x in dates_ref], dtype="U10"))
    root.create_array("station_id", data=np.array(sids, dtype=f"U{max(len(x) for x in sids)}"))
    root.create_array("percentile", data=np.array(pcts, dtype=np.float32))
    root.attrs["description"] = ("Hydrogrammes tenue de cote 2022-2024. percentile 50 = "
                                 "debit physique simule (mediane) ; autres percentiles = "
                                 "tete quantile du modele unique. Pas de membres : une "
                                 "seule passe, une seule distribution predictive.")
    zarr.consolidate_metadata(root.store)
    n_env = sum(1 for e in series.values() if "quantiles" in e)
    print(f"hydro.zarr : {S} stations x {T} jours x {P} percentiles "
          f"({n_env} stations avec enveloppe quantile)")
    print(f"  -> {STORE}")


if __name__ == "__main__":
    main()
