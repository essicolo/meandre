"""Analyse d'une passe du banc fictif : réponse d'orage et étiage de chaque tronçon sous la météo fabriquée.

Lit le cache par tronçon (ETL_DUMP_REACH) et sa série journalière, et le fichier des orages de
make_forcing.py. Écoulement rapide : production de surface et hypodermique ; lent : production de
base et débit de la nappe. Toutes les productions en mm/j sur le tronçon.

Grandeurs par tronçon :
- coef_sec_<mm>, coef_humide_<mm> : écoulement rapide des 7 jours suivant l'orage, en sus de la
  passe témoin, rapporté à la pluie de l'orage ;
- pente_pluie : pente log-log du coefficient sur la pluie, orages après dix jours secs (observé
  sur les débits : 1,2 à 1,5) ;
- rapport_humide_sec_20 : coefficient après trois jours de pluie sur coefficient après dix jours
  secs, orage de 20 mm ;
- etiage_rapport : écoulement total minimal sur 7 jours de la sécheresse de 2001 (15 juillet au
  15 septembre) rapporté à sa moyenne de juin 2001 ;
- etr_2001_mm : évapotranspiration réelle de 2001.

- riviere_20j_<cas> : tout l'écoulement (rapide et lent) des 20 jours suivant l'orage, rapporté à
  la pluie de l'orage.

La réponse d'orage se mesure contre la passe `<étiquette>-temoin`, même météo sans les orages.
Usage : python analyse.py <dossier> <étiquette> [étiquette de référence]
Avec une référence, imprime la différence appariée tronçon par tronçon.
"""
import sys

import numpy as np
import pandas as pd

QUANTILES = [0.1, 0.5, 0.9]


def load(folder, tag):
    daily = np.load(f"{folder}/troncons-{tag}-journalier.npz")
    dates = pd.to_datetime(daily["dates"])
    rapid = daily["prod_surf"] + daily["prod_hypo"]
    slow = daily["prod_base"] + (daily["debit_base"] if "debit_base" in daily.files else 0.0)
    return dates, rapid, slow, daily["etr"]


def metrics(folder, tag):
    """Grandeurs par tronçon ; la réponse d'orage est la différence avec la passe témoin
    `<tag>-temoin`, identique sans les orages, ce qui retire la décrue des pluies précédentes."""
    dates, rapid, slow, etr = load(folder, tag)
    _, rapid0, slow0, _ = load(folder, f"{tag}-temoin")
    storms = pd.read_csv(f"{folder}/fictif-v1-orages.csv")
    out = {}
    for _, s in storms.iterrows():
        t = int(np.flatnonzero(dates == pd.Timestamp(s.date))[0])
        key = f"{'humide' if s.apres_pluie else 'sec'}_{int(s.pluie_mm)}"
        out[f"coef_{key}"] = (rapid[t:t + 7] - rapid0[t:t + 7]).sum(axis=0) / s.pluie_mm
        out[f"riviere_20j_{key}"] = ((rapid + slow)[t:t + 20] - (rapid0 + slow0)[t:t + 20]).sum(axis=0) / s.pluie_mm
    dry = storms[~storms.apres_pluie]
    x = np.log(dry.pluie_mm.to_numpy())
    y = np.log(np.clip(np.stack([out[f"coef_sec_{int(m)}"] for m in dry.pluie_mm]), 1e-6, None))
    out["pente_pluie"] = ((x - x.mean())[:, None] * (y - y.mean(axis=0))).sum(axis=0) / ((x - x.mean()) ** 2).sum()
    out["rapport_humide_sec_20"] = out["coef_humide_20"] / np.clip(out["coef_sec_20"], 1e-6, None)
    total = rapid + slow
    drought = (dates >= "2001-07-15") & (dates <= "2001-09-15")
    june = (dates >= "2001-06-01") & (dates <= "2001-06-30")
    q7 = np.lib.stride_tricks.sliding_window_view(total[drought], 7, axis=0).mean(axis=-1).min(axis=0)
    out["etiage_rapport"] = q7 / np.clip(total[june].mean(axis=0), 1e-6, None)
    out["etr_2001_mm"] = etr[dates.year == 2001].sum(axis=0)
    return pd.DataFrame(out)


def main():
    folder, tag = sys.argv[1], sys.argv[2]
    m = metrics(folder, tag)
    cols = ["coef_sec_10", "coef_sec_20", "coef_sec_60", "coef_humide_20", "riviere_20j_sec_20", "pente_pluie", "rapport_humide_sec_20", "etiage_rapport", "etr_2001_mm"]
    print(f"{tag} : {len(m)} tronçons, quantiles 10 %, 50 %, 90 %")
    print(m[cols].quantile(QUANTILES).round(3).T.to_string())
    m.to_parquet(f"{folder}/metriques-{tag}.parquet")
    if len(sys.argv) > 3:
        ref = metrics(folder, sys.argv[3])
        print(f"\ndifférence {tag} moins {sys.argv[3]}, appariée par tronçon")
        print((m[cols] - ref[cols]).quantile(QUANTILES).round(3).T.to_string())


if __name__ == "__main__":
    main()
