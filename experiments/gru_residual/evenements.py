"""Étape 2, suite : l'écart est-il un décalage dans le temps, une amplitude de crue, ou un
étiage ? Sur les sorties du modèle provincial gelé, 2019-2024, jours mesurés. Une minute.

    .venv/Scripts/python.exe experiments/gru_residual/evenements.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p
from plafond import kge

DOSSIER = f"{_p.DATA_ROOT}/experiments/gru_residual"


def main():
    q = pd.read_parquet(f"{DOSSIER}/debits.parquet")
    lignes = []
    for st in q["q_obs"].columns:
        o = q["q_obs"][st]
        s = q["q_sim"][st]
        if o.notna().sum() < 365 or o.mean() <= 0:
            continue
        # Décalage : k > 0 si le simulé est en retard sur l'observé.
        corr = {k: o.corr(s.shift(-k)) for k in range(-5, 6)}
        k = max(corr, key=corr.get)
        m = o.notna() & s.notna()
        oo, ss = o[m], s[m]
        err = (ss - oo) ** 2
        monte = (oo.diff() > 0).reindex(oo.index).fillna(False)
        haut = oo >= oo.quantile(0.9)
        bas = oo <= oo.quantile(0.3)
        amax = pd.DataFrame({"o": oo, "s": ss}).groupby(oo.index.year).max()
        lignes.append(dict(
            station=st, region=st.split(":")[0], decalage=k,
            gain_r_decale=corr[k] - corr[0],
            kge=kge(s.to_numpy(), o.to_numpy()),
            kge_decale=kge(s.shift(-k).to_numpy(), o.to_numpy()),
            part_err_montee=err[monte].sum() / err.sum(), part_jours_montee=monte.mean(),
            part_err_hauts=err[haut].sum() / err.sum(),
            part_err_bas=err[bas].sum() / err.sum(),
            pointe_annuelle=np.median(amax.s / amax.o),
            r_jour=corr[0],
            r_7j=oo.rolling(7).mean().corr(ss.rolling(7).mean()),
            r_30j=oo.rolling(30).mean().corr(ss.rolling(30).mean()),
            ecart_type=ss.std() / oo.std(),
            cv_jour_obs=oo.diff().abs().mean() / oo.mean(),
            cv_jour_sim=ss.diff().abs().mean() / ss.mean()))
    L = pd.DataFrame(lignes)
    L.to_csv(f"{DOSSIER}/evenements.csv", index=False)
    print(f"{len(L)} stations, 2019-2024, jours mesurés")
    print(f"décalage du simulé (jours, > 0 = en retard) : {L.decalage.value_counts().sort_index().to_dict()}")
    print(f"KGE médian {L.kge.median():.3f} ; avec le simulé recalé du meilleur décalage {L.kge_decale.median():.3f}")
    print(f"corrélation médiane du débit journalier {L.r_jour.median():.3f}, des moyennes sur 7 jours {L.r_7j.median():.3f}, sur 30 jours {L.r_30j.median():.3f}")
    print(f"rapport médian des pointes annuelles simulé/observé {L.pointe_annuelle.median():.2f} (quartiles {L.pointe_annuelle.quantile(0.25):.2f}-{L.pointe_annuelle.quantile(0.75):.2f})")
    print(f"rapport médian des écarts-types simulé/observé {L.ecart_type.median():.2f}")
    print(f"variation journalière moyenne relative : observé {L.cv_jour_obs.median():.3f}, simulé {L.cv_jour_sim.median():.3f}")
    print(f"part de l'erreur quadratique sur les jours de montée {L.part_err_montee.median():.2f} pour {L.part_jours_montee.median():.2f} des jours")
    print(f"part de l'erreur quadratique sur les 10 % de débits les plus forts {L.part_err_hauts.median():.2f}, sur les 30 % les plus faibles {L.part_err_bas.median():.3f}")
    print("\npar territoire (médianes) :")
    print(L.groupby("region")[["kge", "decalage", "pointe_annuelle", "ecart_type", "r_jour", "r_30j"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
