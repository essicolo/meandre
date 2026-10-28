"""Banc de FONCTION DE PERTE : quel terme voit quel defaut, sans entrainer.

POURQUOI (question d'Essi, 2026-09-13 : pourquoi faut-il des heures pour regler une perte
qui nivelle les etiages ?). Calibrer une perte est une question sur la perte, pas sur le
modele. On prend un hydrogramme OBSERVE, on le deforme exactement comme le modele se
trompe, et on demande a chaque terme s'il prefere l'original. Un terme qui note mieux une
serie lissee que la verite ne pourra jamais empecher le lissage, quel que soit son poids
et quelle que soit la duree de l'entrainement. La reponse coute une seconde par terme.

Les deformations sont celles que le diagnostic a mesurees sur les sorties reelles :
  lissage      moyenne mobile de 7, 30 et 90 jours, le nivellement general ;
  gel d'hiver  les mois de decembre a mars remplaces par une recession exponentielle
               perdant 17 % sur la periode, ce qui est exactement la forme des 395 plages
               figees relevees sur les neuf regions ;
  rabotage     les debits au-dessus du troisieme quartile ramenes vers ce quartile,
               le defaut mesure a 0,77 apres vingt epoques ;
  retard       la serie decalee d'un et de trois jours, l'erreur de calendrier ordinaire ;
  volume       la serie multipliee par 1,1, pour situer les autres ecarts.

Un terme est DECLARE AVEUGLE a une deformation quand il la note mieux que la verite ou
qu'il l'en separe de moins de un pour cent.

    .venv/Scripts/python.exe .runs/quebec/banc_perte.py
"""
import numpy as np
import pandas as pd
import torch

from meandre.training.loss import HydroLoss

REGS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cndb", "cndc"]
DOS = "D:/meandre-data/quebec/flotte"
SORTIE = ".reports/quebec/caches/banc-perte.csv"
HIVER = (12, 1, 2, 3)


def deformations(q, mois):
    """Les series a comparer a la verite. q est l'observation, sans lacune."""
    d = {"vérité": q.copy()}
    for f in (7, 30, 90):
        k = np.ones(f) / f
        s = np.convolve(np.pad(q, (f, f), mode="edge"), k, mode="same")[f:-f]
        d[f"lissage {f} j"] = s
    # Gel d'hiver : chaque bloc hivernal devient une recession exponentielle perdant 17 %.
    g = q.copy()
    hiv = np.isin(mois, HIVER)
    i = 0
    while i < len(g):
        if not hiv[i]:
            i += 1
            continue
        j = i
        while j < len(g) and hiv[j]:
            j += 1
        n = j - i
        if n > 20:
            g[i:j] = g[i] * np.exp(np.log(0.83) * np.arange(n) / n)
        i = j
    d["gel d'hiver"] = g
    # Rabotage : ce qui depasse le troisieme quartile est ramene de 30 % vers lui.
    s = q.copy()
    q75 = np.quantile(q, 0.75)
    haut = s > q75
    s[haut] = q75 + 0.7 * (s[haut] - q75)
    d["rabotage des pics"] = s
    for r in (1, 3):
        d[f"retard {r} j"] = np.concatenate([np.full(r, q[0]), q[:-r]])
    d["volume +10 %"] = q * 1.1
    return d


def termes(q_obs, q_sim, var, q75):
    """Chaque terme de la perte, isole, sur une seule station."""
    o = torch.tensor(q_obs, dtype=torch.float32).unsqueeze(1)
    s = torch.tensor(q_sim, dtype=torch.float32).unsqueeze(1)
    mk = torch.ones(1, dtype=torch.bool)
    sv = torch.tensor([var], dtype=torch.float32)
    pt = torch.tensor([q75], dtype=torch.float32)
    base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0,
                w_log_nse=0.0, w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0, w_dq_log=0.0,
                w_peak=0.0, per_station=True, station_var=sv)
    out = {}
    for nom, cle, extra in [("KGE", "w_kge", {}), ("biais de volume", "w_pbias", {}),
                            ("écart quadratique", "w_mse", {}),
                            ("écart quadratique log", "w_log_mse", {}),
                            ("pics", "w_peak", {"peak_threshold": pt}),
                            ("variations d'un jour", "w_dq", {}),
                            ("variations log", "w_dq_log", {}),
                            ("soutien d'étiage", "w_fdc_bas", {})]:
        f = HydroLoss(**{**base, cle: 1.0}, **extra)
        with torch.no_grad():
            r = f(q_obs=o, q_sim=s, station_mask=mk)
        out[nom] = float(r[0] if isinstance(r, tuple) else r)
    return out


# Poids de la recette du socle. GRACE est absent du banc, comme du sous-bassin.
POIDS = {"KGE": 1.0, "biais de volume": 0.5, "écart quadratique": 0.1,
         "écart quadratique log": 0.3, "pics": 0.5}


def main():
    """LA QUESTION QUI DECIDE. Le modele ne choisit pas entre la verite et une deformation,
    il choisit entre deux erreurs. Il est en retard d'une journee, ce qui est la regle des
    qu'une averse est mal localisee. Deux issues s'offrent alors : rester net et en retard,
    ou lisser pour reduire l'ecart quotidien. Si la perte note MIEUX la version lissee que
    la version nette, elle recompense le nivellement, et aucun entrainement n'y changera
    rien. Le meme raisonnement vaut pour l'hiver : reste-t-il moins cher de figer les
    etiages que de les suivre avec un jour de retard ?
    """
    lignes = []
    for reg in REGS:
        z = np.load(f"{DOS}/q-{reg}-A-v4.npz", allow_pickle=True)
        dts = pd.to_datetime([str(v)[:10] for v in z["dates"]])
        mois = dts.month.to_numpy()
        for j in range(z["q_obs"].shape[1]):
            q = z["q_obs"][:, j].astype(float)
            if not np.isfinite(q).all() or (q <= 0).any():
                continue
            var, q75 = float(q.var()), float(np.quantile(q, 0.75))
            tard = np.concatenate([np.full(1, q[0]), q[:-1]])
            cand = {"net, en retard d'un jour": tard}
            for f in (7, 15, 30):
                k = np.ones(f) / f
                cand[f"retard puis lissage {f} j"] = np.convolve(
                    np.pad(tard, (f, f), mode="edge"), k, mode="same")[f:-f]
            g = tard.copy()
            hiv = np.isin(mois, HIVER)
            i = 0
            while i < len(g):
                if not hiv[i]:
                    i += 1
                    continue
                e = i
                while e < len(g) and hiv[e]:
                    e += 1
                if e - i > 20:
                    g[i:e] = g[i] * np.exp(np.log(0.83) * np.arange(e - i) / (e - i))
                i = e
            cand["retard puis hiver figé"] = g
            # A VOLUME CONSERVE. La version ci-dessus change le volume hivernal, que le
            # terme de biais punit a plus de cinq mille pour cent, alors que les plateaux
            # reels du modele ont presque le bon volume : l'hiver simule pese 13,3 % du
            # total contre 11,5 % observe. Comparer une deformation qui deplace l'eau a
            # une qui ne fait que la lisser melange deux defauts. Chaque bloc hivernal est
            # donc remis a sa moyenne d'origine.
            g2 = tard.copy()
            i = 0
            while i < len(g2):
                if not hiv[i]:
                    i += 1
                    continue
                e = i
                while e < len(g2) and hiv[e]:
                    e += 1
                if e - i > 20:
                    n_ = e - i
                    forme = np.exp(np.log(0.83) * np.arange(n_) / n_)
                    g2[i:e] = forme * (tard[i:e].mean() / forme.mean())
                i = e
            cand["retard puis hiver figé, volume conservé"] = g2
            r = tard.copy()
            haut = r > q75
            r[haut] = q75 + 0.7 * (r[haut] - q75)
            cand["retard puis pics rabotés"] = r
            for nom, s_ in cand.items():
                v = termes(q, s_, var, q75)
                v["total recette"] = sum(POIDS[k] * v[k] for k in POIDS)
                v.update(region=reg, station=j, candidat=nom)
                lignes.append(v)
    d = pd.DataFrame(lignes)
    d.to_csv(SORTIE, index=False)
    cols = list(POIDS) + ["total recette"]
    ref = d[d.candidat == "net, en retard d'un jour"].set_index(["region", "station"])[cols]
    n = len(ref)
    print(f"{n} stations completes. Reference : la serie observee decalee d'un jour, nette.")
    print("Ecart de chaque candidat a cette reference, en pour cent du terme.")
    print("NEGATIF = la perte PREFERE le candidat a la serie nette, donc elle recompense le defaut.")
    print()
    print(f"{'candidat':28s} " + " ".join(f"{c[:13]:>14s}" for c in cols))
    for nom, g in d[d.candidat != "net, en retard d'un jour"].groupby("candidat", sort=False):
        g2 = g.set_index(["region", "station"])[cols]
        rel = 100 * (g2 - ref) / (ref.abs() + 1e-12)
        print(f"{nom:28s} " + " ".join(f"{np.median(rel[c]):+13.1f}%" for c in cols))
    print()
    print("part des stations ou la perte de la recette PREFERE le defaut a la serie nette")
    for nom, g in d[d.candidat != "net, en retard d'un jour"].groupby("candidat", sort=False):
        g2 = g.set_index(["region", "station"])["total recette"]
        print(f"  {nom:28s} {100 * float((g2 < ref['total recette']).mean()):5.0f} %")
    print()
    print(f"{SORTIE} ecrit")


if __name__ == "__main__":
    main()
