"""A L'ENVERS : quel terme, a quel poids, forcerait le modele a coller aux etiages ?

Le banc de perte mesure chaque terme sur des hydrogrammes deformes. Ce script s'en sert
dans l'autre sens. La question n'est plus « la perte recompense-t-elle le defaut » mais
« quel terme ajouter, et a quel poids, pour qu'elle cesse de le faire ».

Aucun calcul nouveau : tout se lit dans le cache du banc. Ajouter un terme de poids w
deplace le total de w fois la valeur de ce terme, donc la comparaison entre deux series
se renverse des que w depasse un seuil qui se calcule directement.

Le defaut a corriger est celui que le banc a mesure : partant d'un modele en retard d'une
journee, lisser sur sept jours AMELIORE la perte de huit pour cent, sur 64 % des stations.
On cherche donc le poids qui rend la serie nette gagnante partout, sans casser les autres
comparaisons.

    .venv/Scripts/python.exe .runs/quebec/banc_perte_dosage.py
"""
import numpy as np
import pandas as pd

CACHE = ".reports/quebec/caches/banc-perte.csv"
POIDS = {"KGE": 1.0, "biais de volume": 0.5, "écart quadratique": 0.1,
         "écart quadratique log": 0.3, "pics": 0.5}
CANDIDATS = ["variations d'un jour", "variations log", "soutien d'étiage",
             "pics", "écart quadratique log", "KGE"]
REF = "net, en retard d'un jour"
DEFAUTS = {"retard puis lissage 7 j": "lissage 7 j",
           "retard puis pics rabotés": "pics rabotés",
           "retard puis lissage 15 j": "lissage 15 j",
           "retard puis hiver figé, volume conservé": "hiver figé"}


def main():
    d = pd.read_csv(CACHE)
    cle = ["region", "station"]
    ref = d[d.candidat == REF].set_index(cle)
    tot_ref = sum(POIDS[k] * ref[k] for k in POIDS)

    print("Etat de depart : part des stations ou la perte de la recette prefere le defaut")
    for nom in DEFAUTS:
        g = d[d.candidat == nom].set_index(cle)
        tot = sum(POIDS[k] * g[k] for k in POIDS)
        print(f"  {DEFAUTS[nom]:16s} {100 * float((tot < tot_ref).mean()):5.0f} %")

    print()
    print("Poids supplementaire qu'il faudrait donner a chaque terme pour que la serie")
    print("NETTE l'emporte sur le defaut a 90 % des stations. Un tiret veut dire que le")
    print("terme ne discrimine pas : aucun poids ne renverse la preference.")
    print()
    print(f"{'terme ajouté':24s} " + " ".join(f"{DEFAUTS[n]:>14s}" for n in DEFAUTS))
    lignes = {}
    for terme in CANDIDATS:
        cases = []
        for nom in DEFAUTS:
            g = d[d.candidat == nom].set_index(cle)
            tot = sum(POIDS[k] * g[k] for k in POIDS)
            manque = tot_ref - tot            # positif = le defaut est prefere
            gain = g[terme] - ref[terme]      # positif = le terme punit le defaut
            besoin = np.where(manque > 0, manque / np.maximum(gain, 1e-12), 0.0)
            besoin = np.where(gain <= 0, np.inf, besoin)
            besoin = np.where(manque <= 0, 0.0, besoin)
            w90 = float(np.quantile(besoin, 0.90))
            cases.append(f"{'aucun':>14s}" if not np.isfinite(w90) else f"{w90:14.3f}")
        lignes[terme] = cases
        print(f"{terme:24s} " + " ".join(cases))

    print()
    print("Effet de bord : au poids trouve pour le lissage 7 j, le terme punit-il encore")
    print("les autres deformations, ou en rend-il une preferable ?")
    for terme in CANDIDATS:
        g7 = d[d.candidat == "retard puis lissage 7 j"].set_index(cle)
        tot7 = sum(POIDS[k] * g7[k] for k in POIDS)
        manque = tot_ref - tot7
        gain = g7[terme] - ref[terme]
        besoin = np.where((manque > 0) & (gain > 0), manque / np.maximum(gain, 1e-12), 0.0)
        besoin = np.where((manque > 0) & (gain <= 0), np.inf, besoin)
        w = float(np.quantile(besoin, 0.90))
        if not np.isfinite(w):
            continue
        pires = []
        for nom in DEFAUTS:
            g = d[d.candidat == nom].set_index(cle)
            tot = sum(POIDS[k] * g[k] for k in POIDS) + w * (g[terme] - ref[terme])
            pires.append(f"{DEFAUTS[nom]} {100 * float((tot < tot_ref).mean()):.0f} %")
        print(f"  {terme:24s} poids {w:7.3f} -> " + " | ".join(pires))


if __name__ == "__main__":
    main()
