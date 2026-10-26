"""Phase 0.1 : operator contre operator-lagged, à poids gelés, même modèle provincial.

Compare les deux passes ETL_DUMP_Q d'un territoire : KGE par station sur 2019-2024, retard du
simulé par corrélation croisée, délai de réponse à la pluie simulé moins observé, le tout
croisé avec le nombre de lacs en série vers la station (decalage.csv de experiments/gru_residual,
qui porte aussi le délai observé).

    .venv/Scripts/python.exe experiments/plan_ml/comparer_routage.py outv
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gru_residual"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from meandre.utils import paths as _p
from plafond import kge, etiage
from decalage import delai

DOSSIER = f"{_p.DATA_ROOT}/experiments/plan_ml"
REF = f"{_p.DATA_ROOT}/experiments/gru_residual"


def charger(reg, mode):
    z = np.load(f"{DOSSIER}/q-{reg}-{mode}.npz", allow_pickle=True)
    t = pd.DatetimeIndex(z["dates"])
    ids = [f"{reg}:{s}" for s in z["station_ids"]]
    return pd.DataFrame(z["q_sim"], index=t, columns=ids), pd.DataFrame(z["q_obs"], index=t, columns=ids)


def main():
    reg = sys.argv[1] if len(sys.argv) > 1 else "outv"
    fx = pd.read_parquet(f"{REF}/forcage_bassin.parquet")
    lacs = pd.read_csv(f"{REF}/decalage.csv").set_index("station")
    sims = {m: charger(reg, m) for m in ("lagged", "operator")}
    obs = sims["lagged"][1]
    lignes = []
    for st in obs.columns:
        o = obs[st]
        if o.notna().sum() < 365 or o.mean() <= 0:
            continue
        p = fx[st]["P"].reindex(o.index) if st in fx.columns.get_level_values(0) else None
        row = dict(station=st, lacs_serie=lacs.loc[st, "lacs_serie"] if st in lacs.index else np.nan, delai_obs=delai(p, o) if p is not None else np.nan)
        for m, (s, _) in sims.items():
            q = s[st]
            corr = {k: o.corr(q.shift(-k)) for k in range(-3, 6)}
            row[f"kge_{m}"] = kge(q.to_numpy(), o.to_numpy())
            row[f"decalage_{m}"] = max(corr, key=corr.get)
            row[f"delai_{m}"] = delai(p, q) if p is not None else np.nan
            row[f"etiage_{m}"] = etiage(q.to_numpy(), o.to_numpy(), o.index)
            m_ok = o.notna()
            row[f"vol_{m}"] = q[m_ok].sum() / o[m_ok].sum()
            row[f"var_jour_{m}"] = q.diff().abs().mean() / q.mean()
        row["var_jour_obs"] = o.diff().abs().mean() / o.mean()
        lignes.append(row)
    L = pd.DataFrame(lignes)
    L.to_csv(f"{DOSSIER}/comparaison-routage-{reg}.csv", index=False)
    print(f"{reg}, {len(L)} stations, 2019-2024, poids gelés du modèle provincial")
    for m in ("lagged", "operator"):
        print(f"  {m:9s} KGE médian {L[f'kge_{m}'].median():.3f}, moyen {L[f'kge_{m}'].mean():.3f} ; retard médian {L[f'decalage_{m}'].median():.1f} j, stations en retard d'au moins 1 j {(L[f'decalage_{m}'] >= 1).sum()} ; délai simulé moins observé moyen {(L[f'delai_{m}'] - L.delai_obs).mean():+.2f} j ; |volume-1| médian {(L[f'vol_{m}'] - 1).abs().median():.3f} ; |log étiage| médian {np.abs(np.log(L[f'etiage_{m}'])).median():.3f} ; variation journalière {L[f'var_jour_{m}'].median():.3f} (observée {L.var_jour_obs.median():.3f})")
    d = L.kge_operator - L.kge_lagged
    print(f"  écart de KGE operator moins lagged : médian {d.median():+.3f}, stations améliorées {(d > 0).sum()} sur {len(L)}")
    L["classe_lacs"] = pd.cut(L.lacs_serie, [-1, 0, 2, 4, 1000], labels=["0", "1-2", "3-4", "5+"])
    g = L.groupby("classe_lacs", observed=True).agg(n=("station", "size"), kge_lagged=("kge_lagged", "median"), kge_operator=("kge_operator", "median"), retard_lagged=("decalage_lagged", "mean"), retard_operator=("decalage_operator", "mean"))
    g["delai_lagged"] = L.groupby("classe_lacs", observed=True).apply(lambda x: (x.delai_lagged - x.delai_obs).mean())
    g["delai_operator"] = L.groupby("classe_lacs", observed=True).apply(lambda x: (x.delai_operator - x.delai_obs).mean())
    print("\n  par nombre de lacs en série (retard : corrélation croisée ; délai : réponse à la pluie, simulé moins observé, jours) :")
    print(g.round(2).to_string())


if __name__ == "__main__":
    main()
