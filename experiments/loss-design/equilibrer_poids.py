"""Poser les poids de la perte par ÉQUILIBRAGE mesuré, au lieu de les régler au jugé.

Les poids de la recette en vigueur ont été choisis un par un, au fil des hypothèses, et jamais
comparés entre eux. Rien ne dit qu'un défaut de volume et un défaut d'étiage de sévérité
comparable produisent une variation de perte comparable, et la mesure du 2026-09-20 dit même le
contraire : le terme de pics pèse un demi-point et ne voit rien des basses eaux, tandis que le
seul terme qui voit un prélèvement est à poids nul.

Le critère proposé ici est l'ÉQUILIBRE et non la maximisation. On se donne une liste de défauts
de référence, chacun d'une sévérité qu'un hydrologue juge comparable, et on cherche les poids
tels que chaque défaut produise à peu près la même variation de perte. Maximiser la détection
d'un défaut serait dégénéré : il suffirait d'annuler les autres termes.

La méthode ne suppose rien de la forme des termes. On mesure la matrice de sensibilité, une
ligne par défaut et une colonne par terme, puis on cherche les poids positifs qui rendent le
vecteur des détections le plus uniforme possible, en écart-type du logarithme.

    .venv/Scripts/python.exe .runs/quebec/equilibrer_poids.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from importlib.machinery import SourceFileLoader

_bp = SourceFileLoader("bp", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "banc_perte.py")).load_module()
_be = SourceFileLoader("be", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "banc_etiage.py")).load_module()

# Termes candidats de la recette décomposée, avec le nom de leur poids.
CANDIDATS = [("calendrier", "w_r"), ("volume", "w_beta"), ("amplitude", "w_gamma"),
             ("pics par rapport", "w_peak_ratio"), ("soutien d'étiage", "w_fdc_bas"),
             ("variations", "w_dq")]

# Défauts de référence. La sévérité de chacun est choisie pour qu'un hydrologue les juge
# comparables : ce jugement est le SEUL ingrédient subjectif de la méthode, et il est ici
# explicite au lieu d'être caché dans des poids.
def defauts(q, mois):
    ete = np.isin(mois, (7, 8, 9))
    d = {}
    s = q.copy()
    s[ete] = np.maximum(s[ete] - 0.05 * q.mean(), 0.01 * q.mean())
    d["prélèvement estival de 5 %"] = s
    d["volume trop fort de 10 %"] = q * 1.10
    d["amplitude amputée de 20 %"] = q.mean() + (q - q.mean()) * 0.80
    haut = q >= np.quantile(q, 0.75)
    s = q.copy()
    s[haut] = s[haut] * 0.85
    d["pointes rabotées de 15 %"] = s
    bas = q <= np.quantile(q, 0.25)
    s = q.copy()
    s[bas] = s[bas] * 0.80
    d["étiage creusé de 20 %"] = s
    tard = np.concatenate([[q[0]], q[:-1]])
    k = np.ones(7) / 7
    d["hydrogramme figé, lissage 7 j"] = np.convolve(np.pad(tard, (7, 7), mode="edge"),
                                                     k, mode="same")[7:-7]
    return d


def sensibilites():
    """Matrice défaut par terme, variation médiane de la note sur les stations."""
    lignes = []
    for reg in _bp.REGS:
        f = f"{_bp.DOS}/q-{reg}-A-v4.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        mois = pd.to_datetime([str(v)[:10] for v in z["dates"]]).month.to_numpy()
        for j in range(z["q_obs"].shape[1]):
            q = z["q_obs"][:, j].astype(float)
            if not np.isfinite(q).all() or (q <= 0).any():
                continue
            var, q75 = float(q.var()), float(np.quantile(q, 0.75))
            jeu = defauts(q, mois)
            for nom_t, cle in CANDIDATS:
                n0 = _be.note(cle, q, q, var, q75)
                for nom_d, s in jeu.items():
                    lignes.append({"defaut": nom_d, "terme": nom_t,
                                   "sensibilite": _be.note(cle, q, s, var, q75) - n0})
    t = pd.DataFrame(lignes)
    return t.pivot_table(index="defaut", columns="terme", values="sensibilite", aggfunc="median")


def equilibrer(S, plancher=0.15, plafond=1.0):
    """Poids positifs rendant le vecteur des détections le plus uniforme, en log.

    Un PLANCHER est indispensable et la première version l'avait oublié. Sans lui, la
    minimisation trouve une solution de COIN : elle annule les termes dont le défaut est déjà
    couvert par un autre, et rendait ainsi un poids nul à l'amplitude, aux pics et aux
    variations. Or l'uniformité de la détection n'est pas le seul objectif. Un terme sert
    aussi de CONTRAINTE, il interdit une façon de tricher, et le terme d'amplitude est
    précisément celui qui empêche le modèle d'aplatir. Le plancher encode cela.
    """
    from scipy.optimize import minimize

    A = np.clip(S.to_numpy(), 0.0, None)

    def cout(u):
        w = plancher + (plafond - plancher) / (1.0 + np.exp(-u))
        d = np.clip(A @ w, 1e-12, None)
        return float(np.std(np.log(d)))

    meilleur, val = None, np.inf
    for graine in range(8):
        rng = np.random.default_rng(graine)
        r = minimize(cout, rng.normal(size=A.shape[1]), method="Nelder-Mead",
                     options={"maxiter": 4000, "fatol": 1e-10, "xatol": 1e-8})
        if r.fun < val:
            meilleur, val = r.x, r.fun
    w = plancher + (plafond - plancher) / (1.0 + np.exp(-meilleur))
    return w / w.max()


def main():
    S = sensibilites()
    ordre = [n for n, _ in CANDIDATS if n in S.columns]
    S = S[ordre]
    pd.set_option("display.width", 220)
    print("Sensibilite de chaque terme a chaque defaut, mediane sur les stations.\n")
    print(S.round(4).to_string())

    w = equilibrer(S)
    poids = pd.Series(w, index=S.columns)
    print("\nPoids equilibres, normalises au plus grand :")
    for nom, _cle in CANDIDATS:
        if nom in poids.index:
            print(f"  {nom:20s} {poids[nom]:.2f}")

    for nom_r, vecteur in (("en vigueur, transposee", None), ("equilibree", poids)):
        if vecteur is None:
            continue
        d = S.to_numpy() @ vecteur.to_numpy()
        ecart = float(np.std(np.log(np.clip(d, 1e-12, None))))
        print(f"\nDetection par defaut, recette {nom_r} "
              f"(dispersion en log : {ecart:.2f}, plus petit est mieux) :")
        for nom_d, v in zip(S.index, d):
            print(f"  {nom_d:32s} {v:.4f}")
    print("\nUn ecart-type en log proche de zero signifie que la perte reagit de la meme")
    print("facon a des defauts que l'hydrologue juge de severite comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
