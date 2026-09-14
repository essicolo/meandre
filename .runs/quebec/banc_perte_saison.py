"""Y a-t-il une saison ou un objectif ponctuel est sur ? Idee d'Essi, 2026-09-14.

Le banc de perte a etabli qu'au-dela d'environ 40 pour cent d'erreur d'amplitude par
evenement, tout terme ponctuel prefere une serie retrecie, et qu'en deca aucun ne la
prefere. Or cette erreur n'est pas uniforme dans l'annee : mesuree sur la periode
d'evaluation, elle vaut 0,38 pendant la crue d'avril-mai et 0,61 le reste de l'annee.
La crue serait donc du bon cote du seuil et le reste de l'annee du mauvais.

Si cela se verifie, une ponderation saisonniere cesse d'etre un reglage arbitraire : elle
devient la frontiere entre le domaine ou un objectif ponctuel travaille correctement et
celui ou il faut un terme de distribution. Ce script le tranche en utilisant l'erreur
REELLE de chaque station et de chaque saison, lue dans le cache de `erreur_amplitude.py`,
plutot qu'une valeur choisie.

    .venv/Scripts/python.exe .runs/quebec/banc_perte_saison.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".runs/quebec")
from banc_perte import termes, REGS, DOS, POIDS

CACHE = ".reports/quebec/caches/erreur-amplitude.csv"
CRUE = (4, 5)
RETRAIT = 0.25


def main():
    err = pd.read_csv(CACHE)
    rng = np.random.default_rng(0)
    lig = []
    for reg in REGS:
        z = np.load(f"{DOS}/q-{reg}-A-v4.npz", allow_pickle=True)
        dts = pd.to_datetime([str(v)[:10] for v in z["dates"]])
        mois = dts.month.to_numpy()
        for j in range(z["q_obs"].shape[1]):
            e = err[(err.region == reg) & (err.station == j)]
            if e.empty:
                continue
            q = z["q_obs"][:, j].astype(float)
            if not np.isfinite(q).all() or (q <= 0).any():
                continue
            var, q75 = float(q.var()), float(np.quantile(q, 0.75))
            for nom, sais, col in (("crue d'avril-mai", CRUE, "sigma_crue"),
                                   ("reste de l'année", None, "sigma_ete")):
                sig = float(e[col].iloc[0])
                if not np.isfinite(sig):
                    continue
                m = np.isin(mois, sais) if sais else ~np.isin(mois, CRUE)
                if m.sum() < 120:
                    continue
                # Bruit d'amplitude correle sur cinq jours, d'ecart-type MESURE.
                b = rng.standard_normal(len(q) // 5 + 1)
                bruit = np.exp(sig * np.repeat(b, 5)[:len(q)] - sig ** 2 / 2)
                A = q * bruit
                B = (1 - RETRAIT) * A + RETRAIT * A[m].mean()
                # La perte est evaluee sur la SAISON seulement : c'est ce que ferait une
                # ponderation saisonniere poussee a son extreme.
                for cand, serie in (("bonne variance", A), ("rétréci", B)):
                    v = termes(q[m], serie[m], var, q75)
                    v["total"] = sum(POIDS[k] * v[k] for k in POIDS)
                    v.update(region=reg, station=j, saison=nom, cand=cand, sigma=sig)
                    lig.append(v)
    d = pd.DataFrame(lig)
    if d.empty:
        print("aucune station appariee au cache d'erreur d'amplitude")
        return
    d.to_csv(".reports/quebec/caches/banc-perte-saison.csv", index=False)
    cols = list(POIDS) + ["total"]
    print(f"{d.station.nunique()} stations. Bruit d'amplitude d'ecart-type MESURE par station")
    print(f"et par saison, puis retrecissement de {100 * RETRAIT:.0f} % vers la moyenne saisonniere.")
    print("Part des stations ou la perte PREFERE la serie retrecie : au-dessus de 50 %, un")
    print("objectif ponctuel rabote les pointes de cette saison.\n")
    for saison, g in d.groupby("saison", sort=False):
        a = g[g.cand == "bonne variance"].set_index(["region", "station"])
        b = g[g.cand == "rétréci"].set_index(["region", "station"])
        print(f"  {saison} : {len(a)} stations, erreur d'amplitude médiane {a.sigma.median():.2f}")
        for c in cols:
            pref = 100 * float((b[c] < a[c]).mean())
            print(f"    {c:24s} prefere le rétréci à {pref:5.0f} % des stations")
        print()


    # Quelles combinaisons de termes NE rabotent PAS, saison par saison ? La recette
    # actuelle est le premier jeu ; les suivants retirent les termes que le banc designe.
    JEUX = {
        "recette actuelle": POIDS,
        "sans le terme de pics": {k: v for k, v in POIDS.items() if k != "pics"},
        "sans pics ni écart quadratique": {k: v for k, v in POIDS.items()
                                           if k not in ("pics", "écart quadratique")},
        "KGE et écart quadratique log": {"KGE": 1.0, "écart quadratique log": 0.3},
        "KGE seul": {"KGE": 1.0},
    }
    print("  part des stations où chaque JEU DE TERMES préfère la série rétrécie")
    print(f"    {'jeu':34s} " + " ".join(f"{n[:18]:>20s}" for n in d.saison.unique()))
    for nom, jeu in JEUX.items():
        cases = []
        for saison in d.saison.unique():
            g = d[d.saison == saison]
            a_ = g[g.cand == "bonne variance"].set_index(["region", "station"])
            b_ = g[g.cand == "rétréci"].set_index(["region", "station"])
            ta = sum(w * a_[k] for k, w in jeu.items())
            tb = sum(w * b_[k] for k, w in jeu.items())
            cases.append(f"{100 * float((tb < ta).mean()):19.0f} %")
        print(f"    {nom:34s} " + " ".join(cases))


if __name__ == "__main__":
    main()
