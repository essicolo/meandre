"""Indicateurs d'étiage et KGE par station, à partir des séries aux stations exportées par le
pilote (ETL_DUMP_Q) : minimum annuel du débit moyen sur 7 jours, jours sous le Q90 observé,
volume d'août-septembre, KGE. Médianes sur les stations, un fichier par ligne.

    python .runs/quebec/indicateurs_stations.py <q-<reg>-<tag>.npz> [...]
"""
import sys
import numpy as np
import pandas as pd


def kge(s, o):
    m = np.isfinite(s) & np.isfinite(o)
    s, o = s[m], o[m]
    if len(s) < 60:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2 + ((s.std() / s.mean()) / (o.std() / o.mean()) - 1) ** 2)


res = {}
for f in sys.argv[1:]:
    z = np.load(f, allow_pickle=True)
    t = pd.to_datetime(z["dates"])
    qs, qo = z["q_sim"], z["q_obs"]
    lignes = []
    for j in range(qo.shape[1]):
        o = pd.Series(qo[:, j], index=t)
        s = pd.Series(qs[:, j], index=t)
        ok = o.notna() & s.notna()
        if ok.sum() < 300:
            continue
        q7 = []
        for an in sorted(set(t.year)):
            oa, sa = o[ok & (t.year == an)], s[ok & (t.year == an)]
            if len(oa) > 300:
                q7.append(sa.rolling(7).mean().min() / oa.rolling(7).mean().min())
        q90 = o[ok].quantile(0.10)
        ete = ok & t.month.isin([8, 9])
        lignes.append(dict(q7=np.median(q7) if q7 else np.nan, jours_sim=int((s[ok] <= q90).sum()), jours_obs=int((o[ok] <= q90).sum()),
                           aout_sept=s[ete].sum() / o[ete].sum(), kge=kge(s[ok].to_numpy(), o[ok].to_numpy())))
    d = pd.DataFrame(lignes)
    res[f] = d
    print(f"{f.split('/')[-1]} : {len(d)} stations | KGE median {d.kge.median():.3f} | Q7min median {d.q7.median():.2f} (q25 {d.q7.quantile(0.25):.2f}, q75 {d.q7.quantile(0.75):.2f}) | jours sous Q90 sim/obs median {(d.jours_sim / d.jours_obs).median():.2f} | aout-sept median {d.aout_sept.median():.2f}")
