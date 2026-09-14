"""Erreur d'amplitude par evenement : la mesure qui decide entre deux chantiers.

Le banc de perte a montre que tout terme ponctuel prefere une serie rabotee des que
l'amplitude des evenements se trompe d'environ 40 % averse par averse, et qu'aucun ne la
prefere a 20 %. La dispersion reelle de cette erreur, sur les modeles entraines, dit donc
si le rabotage des pointes est un plafond de forcage ou un defaut d'objectif.

Un evenement est un maximum local du debit observe au-dessus de son quantile 0,90,
separe du precedent d'au moins sept jours. Le pic simule est cherche dans une fenetre de
plus ou moins deux jours, ce qui neutralise un decalage de calendrier. Le rapport des deux
pics, en logarithme, a pour moyenne le retrecissement et pour ecart-type l'erreur
d'amplitude par evenement. L'entrainement deplace la moyenne ; il touche peu la
dispersion, qui est celle du forcage et de la physique.

    .venv/Scripts/python.exe .runs/quebec/erreur_amplitude.py
"""
import numpy as np
import pandas as pd

REGS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cndb", "cndc"]
DOS = "D:/meandre-data/quebec/flotte"
SORTIE = ".reports/quebec/caches/erreur-amplitude.csv"


def evenements(o, seuil, ecart=7):
    idx = [i for i in range(1, len(o) - 1)
           if o[i] > seuil and o[i] >= o[i - 1] and o[i] >= o[i + 1]]
    garde = []
    for i in idx:
        if not garde or i - garde[-1] >= ecart:
            garde.append(i)
        elif o[i] > o[garde[-1]]:
            garde[-1] = i
    return garde


def main():
    lig = []
    for reg in REGS:
        z = np.load(f"{DOS}/q-{reg}-A-v4.npz", allow_pickle=True)
        dts = pd.to_datetime([str(v)[:10] for v in z["dates"]])
        mois = dts.month.to_numpy()
        for j in range(z["q_sim"].shape[1]):
            o, s = z["q_obs"][:, j].astype(float), z["q_sim"][:, j].astype(float)
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() < 700:
                continue
            oo = np.where(m, o, np.nan)
            ev = evenements(np.nan_to_num(oo, nan=-1.0), np.nanquantile(oo, 0.90))
            r = []
            for i in ev:
                lo, hi = max(0, i - 2), min(len(s), i + 3)
                if np.isfinite(s[lo:hi]).any() and o[i] > 0:
                    r.append((np.log(np.nanmax(s[lo:hi]) / o[i]), mois[i]))
            if len(r) < 8:
                continue
            lr = np.array([x[0] for x in r])
            mo = np.array([x[1] for x in r])
            pr = np.isin(mo, (4, 5))
            lig.append(dict(region=reg, station=j, n_ev=len(lr),
                            retrecissement=float(np.exp(lr.mean())),
                            sigma=float(lr.std()),
                            sigma_crue=float(lr[pr].std()) if pr.sum() >= 4 else np.nan,
                            sigma_ete=float(lr[~pr].std()) if (~pr).sum() >= 4 else np.nan))
    d = pd.DataFrame(lig)
    d.to_csv(SORTIE, index=False)
    print(f"{len(d)} stations, {int(d.n_ev.sum())} evenements au-dessus du quantile 0,90, periode d'evaluation")
    print(f"  retrecissement median (pic simule / pic observe, moyenne geometrique) : {d.retrecissement.median():.2f}")
    print(f"  erreur d'amplitude par evenement, ecart-type du log-rapport :")
    print(f"    toutes saisons  mediane {d.sigma.median():.2f} | quartiles {d.sigma.quantile(.25):.2f} a {d.sigma.quantile(.75):.2f}")
    print(f"    crue d'avril-mai mediane {d.sigma_crue.median():.2f}")
    print(f"    autres saisons  mediane {d.sigma_ete.median():.2f}")
    print(f"  stations dont l'erreur depasse 0,40 : {100 * float((d.sigma > 0.40).mean()):.0f} % | sous 0,20 : {100 * float((d.sigma < 0.20).mean()):.0f} %")
    print("\n  par region : retrecissement | erreur par evenement")
    for reg, g in d.groupby("region", sort=False):
        print(f"    {reg.upper():5s} {g.retrecissement.median():5.2f} | {g.sigma.median():5.2f} | n={len(g)}")
    print(f"\n{SORTIE} ecrit")


if __name__ == "__main__":
    main()
