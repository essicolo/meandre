"""Candidate 5.3, forme physique : l'exposant des aires saturées suffit-il à rendre la
forme de la réponse aux orages ?

L'épreuve 3.1 a montré que le coefficient d'écoulement observé croît avec l'humidité
antécédente (0,025 à 0,065) et avec la taille de la pluie (0,008 à 0,096), alors que le
simulé reste plat (0,035 à 0,041 et 0,022 à 0,058). La part ruisselée vaut
(theta1/thetas1)^b avec b = 8 uniforme. Plus b est grand, plus la part bascule d'un coup
quand la couche de surface approche la saturation ; plus b est petit, plus elle répond
progressivement. C'est exactement le levier de la forme, et c'est la première chose à
éprouver avant toute forme apprise.

MÉTHODE. Zéro époque, modèle provincial gelé, seule la clé `column.saturated_area_exponent`
varie (variable d'environnement MEANDRE_VSA_B, qui prime sur le fichier déclaré). Le pilote
écrit les séries de débit aux stations (ETL_DUMP_Q), une par territoire et par valeur de b.
La statistique est celle de `orages_ete.py`, recalculée ici pour que l'observé et le simulé
sortent du même chemin de code : orage d'été de 10 mm et plus après deux jours secs,
coefficient d'écoulement = hausse cumulée du débit spécifique sur cinq jours rapportée à la
pluie de ces cinq jours, humidité antécédente = indice de précipitation antérieure (0,9 par
jour) rapporté à sa médiane de la station.

    python balayage_vsa.py --dumps '/home/essi/planml-out/vsa-b{b}-{reg}.npz' --b 2 8 16

Les fichiers de forçage par bassin et la table des attributs viennent du dossier de
l'expérience du correcteur aval (`experiments/gru_residual`), qui porte les mêmes stations.
L'aire drainée est lue là où elle est juste : la table des attributs a porté un temps des
valeurs transformées pour le réseau (l'aire valait 1,3 pour 1 300 km 2).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p

REF = f"{_p.DATA_ROOT}/experiments/gru_residual"
CLASSES_API = [0.0, 0.5, 0.8, 1.2, 1.6, 2.5, 100.0]
ETIQUETTES_API = ["< 0,5", "0,5-0,8", "0,8-1,2", "1,2-1,6", "1,6-2,5", "> 2,5"]
BORNES_PLUIE = [0.0, 15.0, 25.0, 40.0, 1000.0]
ETIQUETTES_PLUIE = ["10-15", "15-25", "25-40", "> 40"]


def evenements(p, q_obs, q_sim):
    """Un enregistrement par orage d'été : pluie, humidité antécédente, volume relatifs."""
    api = np.zeros(len(p))
    pv = np.asarray(p, dtype="float64")
    for k in range(1, len(p)):
        api[k] = 0.9 * api[k - 1] + pv[k - 1]
    api = pd.Series(api, index=p.index)
    sec = (p.shift(1) < 2.0) & (p.shift(2) < 2.0)
    ev = (p >= 10.0) & sec & p.index.month.isin((6, 7, 8, 9))
    lignes = []
    for t in p.index[ev]:
        i = p.index.get_loc(t)
        if i + 5 >= len(p) or i < 1:
            continue
        fen = slice(i, i + 5)
        o, s = q_obs.iloc[fen], q_sim.iloc[fen]
        if o.isna().any() or s.isna().any() or not np.isfinite(q_obs.iloc[i - 1]):
            continue
        pl = float(p.iloc[fen].sum())
        if pl <= 0:
            continue
        lignes.append(dict(date=t, pluie=pl, api=float(api.iloc[i]),
                           vol_obs=float((o - q_obs.iloc[i - 1]).clip(lower=0).sum()) / pl,
                           vol_sim=float((s - q_sim.iloc[i - 1]).clip(lower=0).sum()) / pl))
    return pd.DataFrame(lignes)


def _nom_b(b):
    """Le nom de fichier porte l'exposant sans décimale inutile (2 et non 2.0)."""
    return f"{b:g}"


def charger(dumps, b, regions, fx, aire):
    """Tous les orages d'une valeur de b, territoires réunis."""
    blocs = []
    for reg in regions:
        f = dumps.format(b=_nom_b(b), reg=reg)
        if not os.path.exists(f):
            print(f"  (absent : {f})")
            continue
        z = np.load(f, allow_pickle=True)
        idx = pd.DatetimeIndex(z["dates"])
        for j, sid in enumerate(z["station_ids"]):
            st = f"{reg}:{sid}"
            if st not in fx.columns.get_level_values(0) or st not in aire.index:
                continue
            o = pd.Series(z["q_obs"][:, j], index=idx)
            s = pd.Series(z["q_sim"][:, j], index=idx)
            if o.notna().sum() < 365 or aire.loc[st] <= 0:
                continue
            conv = 86.4 / float(aire.loc[st])
            p = fx[st]["P"].reindex(idx)
            e = evenements(p, o * conv, s * conv)
            if len(e) < 5:
                continue
            e["station"] = st
            e["region"] = reg
            e["api_rel"] = e.api / e.api.median()
            # Volume de la période : dit si l'exposant change l'écoulement annuel, ce que le
            # critère d'arrêt de la candidate surveille.
            e["bilan_vol"] = float(s.sum() / o.sum()) if o.sum() > 0 else np.nan
            blocs.append(e)
    return pd.concat(blocs, ignore_index=True) if blocs else pd.DataFrame()


def rapport(E, b):
    print("")
    if not len(E):
        print(f"===== b = {b} : aucun orage =====")
        return
    print(f"===== b = {b} : {len(E)} orages, {E.station.nunique()} stations =====")
    E = E.copy()
    E["classe_api"] = pd.cut(E.api_rel, CLASSES_API, labels=ETIQUETTES_API)
    E["classe_pluie"] = pd.cut(E.pluie, BORNES_PLUIE, labels=ETIQUETTES_PLUIE)
    par_st = E.groupby("station").agg(obs=("vol_obs", "median"), sim=("vol_sim", "median"), vol=("bilan_vol", "median"))
    print(f"coefficient médian : observé {E.vol_obs.median():.3f}, simulé {E.vol_sim.median():.3f} ; "
          f"rapport médian par station sim/obs {(par_st.sim / par_st.obs.clip(lower=1e-4)).median():.2f} ; "
          f"volume simulé/observé médian {par_st.vol.median():.2f}")
    t = E.groupby("classe_api", observed=True).agg(n=("vol_obs", "size"), obs=("vol_obs", "median"), sim=("vol_sim", "median"))
    t["sim/obs"] = t.sim / t.obs
    print("par humidité antécédente :")
    print(t.round(3).to_string())
    t = E.groupby("classe_pluie", observed=True).agg(n=("vol_obs", "size"), obs=("vol_obs", "median"), sim=("vol_sim", "median"))
    t["sim/obs"] = t.sim / t.obs
    print("par pluie d'événement :")
    print(t.round(3).to_string())
    co = E[["api_rel", "vol_obs"]].corr(method="spearman").iloc[0, 1]
    cs = E[["api_rel", "vol_sim"]].corr(method="spearman").iloc[0, 1]
    print(f"corrélation de rang du coefficient avec l'indice de précipitation antérieure : observé {co:+.2f}, simulé {cs:+.2f}")
    for reg in sorted(E.region.unique()):
        g = E[E.region == reg]
        if len(g) < 30:
            continue
        co = g[["api_rel", "vol_obs"]].corr(method="spearman").iloc[0, 1]
        cs = g[["api_rel", "vol_sim"]].corr(method="spearman").iloc[0, 1]
        print(f"  {reg}: n {len(g)}, obs {g.vol_obs.median():.3f}, sim {g.vol_sim.median():.3f}, "
              f"rapport {g.vol_sim.median() / max(g.vol_obs.median(), 1e-4):.2f}, "
              f"rang-API obs {co:+.2f} sim {cs:+.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dumps", required=True, help="motif de chemin, avec {b} et {reg}")
    p.add_argument("--b", nargs="+", type=float, default=[8.0])
    p.add_argument("--regions", nargs="+", default=["slso", "slno", "mont", "outv", "sagu", "gasp"])
    p.add_argument("--sortie", default=None)
    a = p.parse_args()
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    at = pd.read_parquet(f"{REF}/attributs.parquet")
    aire = at["aire_km2"]
    tout = []
    for b in a.b:
        E = charger(a.dumps, b, a.regions, fx, aire)
        rapport(E, b)
        if len(E):
            E["b"] = b
            tout.append(E)
    if a.sortie and tout:
        pd.concat(tout, ignore_index=True).to_parquet(a.sortie)
        print(f"\necrit : {a.sortie}")


if __name__ == "__main__":
    main()
