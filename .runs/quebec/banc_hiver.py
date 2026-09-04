"""L'hiver est-il MODELISE, ou ecrase ? Comparaison appariee de deux jeux de sorties.

Le KGE median ne dit pas si l'hiver existe. Un modele qui descend une recession
exponentielle de decembre a avril, sans redoux ni pluie sur neige, peut afficher un bon
KGE annuel parce que la crue de printemps et l'ete portent la variance. Le defaut se voit
ailleurs : la simulation reste PLATE, elle varie de moins de un pour cent par jour,
pendant des semaines d'affilee.

Mesure du 2026-09-03 sur les sorties du rapport : le simule est plat 6.5 a 54.7 % du
temps selon la region contre 2.5 a 17.7 % pour l'observe, avec des suites de 16 a 123
jours commencant en hiver, et le classement des regions par platitude reproduit celui de
leur echec. C'est le defaut R65.

Ce banc compare deux jeux de dumps ETL_DUMP_Q sur les memes stations et les memes jours,
et rend, pour l'hiver comme pour l'annee : la part de jours plats, la plus longue suite
plate, le nombre d'evenements simules contre observes, et le rapport de variabilite. Un
hiver rendu vivant doit faire monter les evenements et la variabilite, et baisser la
platitude -- que le KGE bouge ou non.

  python .runs/quebec/banc_hiver.py <dossier> [--a ancien --b corrige]
"""
import argparse
import glob
import os
import sys

import numpy as np

HIVER = (12, 1, 2, 3)


def _charge(chemin):
    d = np.load(chemin, allow_pickle=True)
    S, O = d["q_sim"].astype(float), d["q_obs"].astype(float)
    mois = np.array([int(str(x)[5:7]) for x in d["dates"]])
    return S, O, mois


def _plat(x, seuil=0.01):
    return np.abs(np.diff(x)) / np.maximum(x[:-1], 1e-9) < seuil


def _suite_max(masque):
    n = mx = 0
    for c in masque:
        n = n + 1 if c else 0
        mx = max(mx, n)
    return mx


def _evenements(x, facteur=1.25):
    """Montees de plus de 25 % en un jour : un evenement hydrologique visible."""
    r = np.diff(x) / np.maximum(x[:-1], 1e-9)
    return int((r > (facteur - 1.0)).sum())


def mesures(chemin):
    S, O, mois = _charge(chemin)
    h = np.isin(mois[:-1], HIVER)
    out = {k: [] for k in ("plat", "plat_h", "suite", "ev_sim_h", "ev_obs_h",
                           "cv_sim_h", "cv_obs_h")}
    for j in range(S.shape[1]):
        s, o = S[:, j], O[:, j]
        m = np.isfinite(s) & np.isfinite(o)
        if m.sum() < 200:
            continue
        cs = _plat(s)
        v = np.isfinite(s[:-1]) & np.isfinite(o[:-1])
        out["plat"].append(cs[v].mean())
        if (v & h).any():
            out["plat_h"].append(cs[v & h].mean())
        out["suite"].append(_suite_max(cs))
        mh = np.isin(mois, HIVER) & np.isfinite(s) & np.isfinite(o)
        if mh.sum() > 100:
            sh, oh = s[mh], o[mh]
            out["ev_sim_h"].append(_evenements(sh))
            out["ev_obs_h"].append(_evenements(oh))
            out["cv_sim_h"].append(sh.std() / max(sh.mean(), 1e-9))
            out["cv_obs_h"].append(oh.std() / max(oh.mean(), 1e-9))
    if not out["plat"]:
        return None
    return {k: float(np.nanmedian(v)) if v else float("nan") for k, v in out.items()} | \
           {"n": len(out["plat"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dossier")
    ap.add_argument("--a", default="ancien", help="etiquette du bras de reference")
    ap.add_argument("--b", default="corrige", help="etiquette du bras compare")
    x = ap.parse_args()

    fichiers = sorted(glob.glob(os.path.join(x.dossier, "q-*.npz")))
    regions = sorted({os.path.basename(f).split("-")[1] for f in fichiers})
    if not regions:
        raise SystemExit(f"aucun fichier q-*.npz dans {x.dossier}")

    print(f"{'region':7s} {'bras':9s} {'plat':>6s} {'plat hiver':>11s} {'suite max':>10s} "
          f"{'evts hiver sim/obs':>19s} {'variabilite hiver sim/obs':>26s}")
    ecarts = []
    for reg in regions:
        ligne = {}
        for bras in (x.a, x.b):
            f = os.path.join(x.dossier, f"q-{reg}-{bras}.npz")
            if not os.path.exists(f):
                continue
            m = mesures(f)
            if m is None:
                continue
            ligne[bras] = m
            print(f"{reg:7s} {bras:9s} {100*m['plat']:5.1f}% {100*m['plat_h']:10.1f}% "
                  f"{m['suite']:9.0f}j {m['ev_sim_h']:9.0f} /{m['ev_obs_h']:8.0f} "
                  f"{m['cv_sim_h']:14.2f} /{m['cv_obs_h']:10.2f}")
        if x.a in ligne and x.b in ligne:
            ecarts.append((reg,
                           100 * (ligne[x.b]["plat_h"] - ligne[x.a]["plat_h"]),
                           ligne[x.b]["ev_sim_h"] - ligne[x.a]["ev_sim_h"]))
        print()
    if ecarts:
        print(f"{'region':7s} {'platitude hiver':>17s} {'evenements hiver':>18s}")
        for reg, dp, de in ecarts:
            print(f"{reg:7s} {dp:+16.1f}pt {de:+17.0f}")
        dp = np.median([e[1] for e in ecarts]); de = np.median([e[2] for e in ecarts])
        print(f"\nmediane : platitude hivernale {dp:+.1f} point(s), "
              f"evenements hivernaux {de:+.0f}")
        if dp < -2:
            print("L'hiver est PLUS VIVANT qu'avant : la platitude recule.")
        elif dp > 2:
            print("L'hiver est PLUS ECRASE qu'avant : la platitude progresse.")
        else:
            print("L'hiver n'a pas change de nature : la platitude est stable.")


if __name__ == "__main__":
    main()
