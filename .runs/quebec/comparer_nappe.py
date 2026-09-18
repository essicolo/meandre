"""Le stock souterrain simulé suit-il le niveau mesuré des nappes ? Comparaison de même nature.

Les deux grandeurs décrivent la même chose : l'eau stockée sous terre. Le modèle la donne en
millimètres d'eau, le réseau en mètres de profondeur sous le repère du tubage, signe opposé.
Elles se comparent donc en dynamique, et leur rapport définit la porosité de drainage, dont la
valeur plausible se situe entre 0,01 dans le roc fracturé et 0,30 dans les sables.

Quatre mesures par puits, toutes sans hypothèse d'échelle :
- corrélation des anomalies mensuelles, chaque série centrée par mois calendaire ;
- corrélation du cycle saisonnier moyen, douze valeurs ;
- mois du niveau le plus haut, simulé contre mesuré ;
- porosité de drainage impliquée par la pente de régression, à comparer aux valeurs connues.

    .venv/Scripts/python.exe .runs/quebec/comparer_nappe.py slso mont [--suffixe=-variante]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

DERIVES = f"{_paths.DERIVED_ROOT}/auxiliaires"


def mensuel(dates, valeurs):
    d = pd.DataFrame({"date": pd.to_datetime(dates), "v": valeurs})
    return d.groupby(d.date.values.astype("datetime64[M]")).v.mean()


def anomalies(s):
    return s - s.groupby(s.index.month).transform("mean")


def main(regions, suffixe=""):
    obs = pd.read_parquet(f"{DERIVES}/rsesq-niveaux-journaliers.parquet")
    lignes = []
    for reg in regions:
        f = f"{DERIVES}/nappe-{reg}{suffixe}.npz"
        if not os.path.exists(f):
            print(f"{reg} : {f} absent")
            continue
        z = np.load(f, allow_pickle=True)
        dates = z["dates"]
        for j, puits in enumerate(z["puits"]):
            o = obs[obs.puits == str(puits)]
            if len(o) < 365 * 3:
                continue
            # Le niveau est une PROFONDEUR : on l'inverse pour que les deux montent ensemble.
            no = mensuel(o.date.values, -o.niveau_m.values)
            ns = mensuel(dates, z["s_gw"][:, j])
            c = pd.concat([no.rename("obs"), ns.rename("sim")], axis=1).dropna()
            if len(c) < 60:
                continue
            ao, asim = anomalies(c.obs), anomalies(c.sim)
            clim_o = c.obs.groupby(c.index.month).mean()
            clim_s = c.sim.groupby(c.index.month).mean()
            pente = np.polyfit(c.obs.values, c.sim.values, 1)[0] / 1000.0
            lignes.append({"region": reg, "puits": str(puits), "mois": len(c),
                           "r_anomalies": float(np.corrcoef(ao, asim)[0, 1]),
                           "r_saison": float(np.corrcoef(clim_o, clim_s)[0, 1]),
                           "mois_max_obs": int(clim_o.idxmax()), "mois_max_sim": int(clim_s.idxmax()),
                           "porosite_drainage": float(pente),
                           "amplitude_obs_m": float(clim_o.max() - clim_o.min()),
                           "amplitude_sim_mm": float(clim_s.max() - clim_s.min())})
    t = pd.DataFrame(lignes)
    if t.empty:
        print("aucun puits comparable")
        return 1
    t.to_parquet(f"{DERIVES}/comparaison-nappe{suffixe}.parquet", index=False)
    dec = ((t.mois_max_sim - t.mois_max_obs + 6) % 12) - 6
    print(f"{len(t)} puits comparés, {t.mois.median():.0f} mois en médiane\n")
    print(f"anomalies mensuelles : r médian {t.r_anomalies.median():+.2f}, quartiles {t.r_anomalies.quantile(.25):+.2f} à {t.r_anomalies.quantile(.75):+.2f}, part au-dessus de 0,5 : {(t.r_anomalies > 0.5).mean():.2f}")
    print(f"cycle saisonnier      : r médian {t.r_saison.median():+.2f}, quartiles {t.r_saison.quantile(.25):+.2f} à {t.r_saison.quantile(.75):+.2f}")
    print(f"mois le plus haut     : décalage médian {int(dec.median())} mois, écart absolu médian {int(dec.abs().median())}")
    print(f"   observé : {t.mois_max_obs.value_counts().head(3).to_dict()} | simulé : {t.mois_max_sim.value_counts().head(3).to_dict()}")
    print(f"porosité de drainage impliquée : médiane {t.porosite_drainage.median():.3f}, quartiles {t.porosite_drainage.quantile(.25):.3f} à {t.porosite_drainage.quantile(.75):.3f}")
    print(f"   part hors de la plage plausible 0,01 à 0,30 : {((t.porosite_drainage < 0.01) | (t.porosite_drainage > 0.30)).mean():.2f}")
    print(f"amplitude saisonnière : observée {t.amplitude_obs_m.median():.2f} m, simulée {t.amplitude_sim_mm.median():.0f} mm")
    print(f"\n{DERIVES}/comparaison-nappe{suffixe}.parquet")
    return 0


if __name__ == "__main__":
    _args = [a for a in sys.argv[1:] if not a.startswith("--suffixe=")]
    _suf = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--suffixe=")), "")
    sys.exit(main([a.lower() for a in _args] or ["slso", "mont"], _suf))
