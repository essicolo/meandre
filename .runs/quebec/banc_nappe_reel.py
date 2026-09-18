"""La nappe libre confrontée aux puits mesurés, sous trois formes de recharge.

Le banc sur colonne fictive a montré qu'un réservoir sous recharge sinusoïdale ne peut pas
tenir ensemble le battement et la phase mesurés, et qu'une recharge en impulsion de fonte
lève le verrou. Reste à savoir ce que le module rend avec la recharge que la colonne produit
AUJOURD'HUI, ce qui sépare ce qui manque au module de ce qui manque à son forçage.

Trois forçages, même module, mêmes puits.

  telle_quelle  la recharge simulée aux nœuds portant un puits, sans retouche ;
  corrigee      la même, divisée par la part de journée que la boucle de sous-pas traite,
                approximation au premier ordre d'une colonne qui résout ses équations ;
  impulsion     une impulsion de fonte de même total annuel, centrée sur le 15 avril, qui
                représente ce qu'une colonne dotée d'une troisième couche qui respire
                livrerait.

Trois mesures, les mêmes que pour l'aquifère actuel : corrélation des anomalies mensuelles,
mois du niveau le plus haut, et porosité de drainage qu'il faudrait pour que le battement
simulé égale le battement mesuré. Cette dernière est le test de vraisemblance : elle doit
tomber entre 0,01 dans le roc fracturé et 0,30 dans les sables.

    .venv/Scripts/python.exe .runs/quebec/banc_nappe_reel.py outv slno gasp sagu
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths
from meandre.vertical.nappe import NappeLibre

torch.set_default_dtype(torch.float64)

DERIVES = f"{_paths.DERIVED_ROOT}/auxiliaires"
SY = 0.05
Z_RIV = 8.0
H_REF = 4.0
K_B = 2.0e-3                   # temps de réponse de 100 jours, dans la plage des récessions
E_MAX = 0.004                  # extraction maximale depuis la zone saturée (m/j)
Z_EXT = 9.0

# Part de journée TRAITÉE par la boucle de sous-pas, par mois et par territoire, mesurée le
# 2026-09-18. Sert à approximer une colonne convergée sans la resimuler.
TRAITEE = {
    "outv": [0.48, 0.53, 0.52, 0.38, 0.63, 0.83, 0.90, 0.92, 0.89, 0.79, 0.72, 0.54],
    "gasp": [0.55, 0.52, 0.49, 0.51, 0.70, 0.84, 0.89, 0.89, 0.86, 0.75, 0.63, 0.57],
    "slno": [0.52, 0.50, 0.48, 0.42, 0.62, 0.81, 0.87, 0.84, 0.83, 0.72, 0.62, 0.55],
    "sagu": [0.67, 0.66, 0.64, 0.66, 0.76, 0.83, 0.86, 0.85, 0.84, 0.77, 0.72, 0.68],
}


def forcages(recharge, mois, region):
    """Les trois séries de recharge, en m/j, de même total annuel."""
    r = recharge / 1000.0
    traitee = np.array(TRAITEE[region])[mois - 1][:, None]
    corrigee = r / np.clip(traitee, 0.05, 1.0)
    return {"telle_quelle": r, "corrigee": corrigee * r.mean() / corrigee.mean()}


def impulsion(mois, jour_annee, total_m_par_jour, n_puits):
    """Impulsion de fonte centrée sur le 15 avril, même moyenne que la recharge simulée."""
    pic = np.exp(-0.5 * ((np.minimum(np.abs(jour_annee - 105), 365 - np.abs(jour_annee - 105))) / 12.0) ** 2)
    forme = 0.3 + 0.7 * pic / pic.mean()
    s = forme[:, None] * np.ones((1, n_puits))
    return s * (total_m_par_jour / s.mean(axis=0, keepdims=True))


def simuler(recharge):
    """Profondeur de la nappe (m) pour une recharge (jours, puits) en m/j."""
    n = recharge.shape[1]
    col = lambda v: torch.full((n,), float(v))
    m = NappeLibre(n_substep=4, exposant=2.0)
    z = col(6.0)
    jour = np.arange(len(recharge)) % 365
    saison = np.clip(np.cos(2 * np.pi * (jour - 196) / 365.0), 0.0, None)
    zs = []
    for i in range(len(recharge)):
        z, _q, _e = m(z, torch.as_tensor(recharge[i]), col(SY), col(K_B), col(Z_RIV),
                      col(H_REF), col(E_MAX * saison[i]), col(Z_EXT))
        zs.append(z.numpy().copy())
    return np.array(zs)


def mensuel(dates, valeurs):
    d = pd.DataFrame({"date": pd.to_datetime(dates), "v": valeurs})
    return d.groupby(d.date.values.astype("datetime64[M]")).v.mean()


def juger(region, obs, nom, z, dates, puits):
    lignes = []
    for j, p in enumerate(puits):
        o = obs[obs.puits == str(p)]
        if len(o) < 365 * 3:
            continue
        no = mensuel(o.date.values, -o.niveau_m.values)
        ns = mensuel(dates, -z[:, j])
        c = pd.concat([no.rename("obs"), ns.rename("sim")], axis=1, sort=True).dropna()
        if len(c) < 60:
            continue
        ao = c.obs - c.obs.groupby(c.index.month).transform("mean")
        asim = c.sim - c.sim.groupby(c.index.month).transform("mean")
        clim_o = c.obs.groupby(c.index.month).mean()
        clim_s = c.sim.groupby(c.index.month).mean()
        amp_o = float(clim_o.max() - clim_o.min())
        amp_s = float(clim_s.max() - clim_s.min())
        lignes.append({"region": region, "forcage": nom, "puits": str(p),
                       "r_anomalies": float(np.corrcoef(ao, asim)[0, 1]),
                       "r_saison": float(np.corrcoef(clim_o, clim_s)[0, 1]),
                       "mois_max_obs": int(clim_o.idxmax()), "mois_max_sim": int(clim_s.idxmax()),
                       "sy_implique": SY * amp_s / amp_o if amp_o > 0 else np.nan})
    return lignes


def main(regions):
    obs = pd.read_parquet(f"{DERIVES}/rsesq-niveaux-journaliers.parquet")
    toutes = []
    for reg in regions:
        f = f"{DERIVES}/nappe-{reg}-tnt.npz"
        if not os.path.exists(f):
            print(f"{reg} : {f} absent")
            continue
        z0 = np.load(f, allow_pickle=True)
        dates = pd.to_datetime(z0["dates"])
        mois = dates.month.to_numpy()
        jour = dates.dayofyear.to_numpy()
        series = forcages(z0["recharge"], mois, reg)
        series["impulsion"] = impulsion(mois, jour, series["telle_quelle"].mean(axis=0), z0["recharge"].shape[1])
        for nom, r in series.items():
            toutes += juger(reg, obs, nom, simuler(r), z0["dates"], z0["puits"])
    t = pd.DataFrame(toutes)
    if t.empty:
        print("aucun puits comparable")
        return 1
    print(f"{t.puits.nunique()} puits, {len(regions)} territoires\n")
    print("forçage       | r anomalies | r saison | mois max simulé | porosité de drainage impliquée")
    for nom in ("telle_quelle", "corrigee", "impulsion"):
        s = t[t.forcage == nom]
        if s.empty:
            continue
        print(f"{nom:13s} | {s.r_anomalies.median():+.2f}       | {s.r_saison.median():+.2f}    |"
              f" {int(s.mois_max_sim.mode().iloc[0]):^15d} | {s.sy_implique.median():.3f}")
    s = t[t.forcage == "telle_quelle"]
    print(f"{'mesuré':13s} |             |          | {int(s.mois_max_obs.mode().iloc[0]):^15d} |")
    t.to_parquet(f"{DERIVES}/banc-nappe-reel.parquet", index=False)
    print(f"\n{DERIVES}/banc-nappe-reel.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main([a.lower() for a in sys.argv[1:]] or ["outv", "slno", "gasp", "sagu"]))
