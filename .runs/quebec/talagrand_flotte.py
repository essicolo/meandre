"""Diagrammes de Talagrand de la flotte provinciale, a partir des sorties quantiles.

POURQUOI (Essi, 2026-09-08, pour la presentation de la semaine prochaine). Le rapport doit
porter les diagrammes de rang de la branche probabiliste. La tete quantile ne produit pas
un ensemble de membres mais K quantiles ; l'objet standard est alors l'histogramme de la
transformee integrale de probabilite (PIT), qui est la generalisation continue du
diagramme de Talagrand et qui se lit de la meme facon : plat si l'enveloppe est calibree,
en U si elle est trop etroite, en cloche si elle est trop large, en pente si elle est
biaisee.

Entree : les depots de la flotte q-<region>-<bras>.npz portant `q_quantiles` (T, S, K) et
`quantile_taus`, ecrits par le pilote quand la tete quantile est chargee.

    .venv/Scripts/python.exe .runs/quebec/talagrand_flotte.py --bras A-v3q
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np

from meandre.diagnostics.talagrand import flatness_metrics, pit_histogram


def pit_depuis_quantiles(q_obs, q_quant, taus):
    """PIT de chaque observation dans l'enveloppe quantile, par interpolation lineaire.

    Sous les K quantiles, la valeur observee se situe entre deux niveaux tau consecutifs :
    on interpole. Sous le plus bas des quantiles la PIT vaut 0, au-dessus du plus haut
    elle vaut 1 ; ces deux cas sont les queues que le diagramme doit montrer.
    """
    o = np.asarray(q_obs, float)
    q = np.sort(np.asarray(q_quant, float), axis=-1)
    tt = np.asarray(taus, float)
    pit = np.full(o.shape, np.nan)
    fini = np.isfinite(o) & np.isfinite(q).all(axis=-1)
    if not fini.any():
        return pit
    oo, qq = o[fini], q[fini]
    sous = (qq < oo[:, None]).sum(axis=1)          # nombre de quantiles sous l'observation
    p = np.where(sous == 0, 0.0, np.where(sous == len(tt), 1.0, np.nan))
    mil = np.isnan(p)
    if mil.any():
        i = sous[mil] - 1
        q1 = qq[mil, i]
        q2 = qq[mil, i + 1]
        t1, t2 = tt[i], tt[i + 1]
        frac = np.clip((oo[mil] - q1) / np.maximum(q2 - q1, 1e-12), 0.0, 1.0)
        p[mil] = t1 + frac * (t2 - t1)
    pit[fini] = p
    return pit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default="D:/meandre-data/quebec/flotte")
    ap.add_argument("--bras", default="A-v3q")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--sortie", default="D:/meandre-data/quebec/carte/talagrand.npz")
    a = ap.parse_args()

    fichiers = sorted(glob.glob(f"{a.dossier}/q-*-{a.bras}.npz"))
    if not fichiers:
        raise SystemExit(f"aucun depot q-*-{a.bras}.npz dans {a.dossier}")
    print(f"{'region':7s} {'stations':>8s} {'valeurs':>10s} {'delta':>7s} {'chi2':>9s} "
          f"{'sous':>6s} {'sur':>6s}  lecture")
    tout, par_region = [], {}
    for f in fichiers:
        reg = os.path.basename(f)[2:-(len(a.bras) + 5)]
        d = np.load(f, allow_pickle=True)
        if "q_quantiles" not in d.files:
            print(f"{reg:7s} {'':>8s} {'':>10s} pas d'enveloppe quantile dans ce depot")
            continue
        pit = pit_depuis_quantiles(d["q_obs"], d["q_quantiles"], d["quantile_taus"])
        v = pit[np.isfinite(pit)]
        if v.size < 500:
            continue
        cnt, _ = pit_histogram(v, n_bins=a.bins)
        m = flatness_metrics(cnt)
        sous = 100.0 * float((v <= 1e-9).mean())
        sur = 100.0 * float((v >= 1 - 1e-9).mean())
        # Lecture : U = enveloppe trop etroite, cloche = trop large, pente = biaisee.
        moitie = float((v < 0.5).mean())
        creux = cnt[a.bins // 4:3 * a.bins // 4].mean() / max(cnt.mean(), 1e-9)
        lect = ("trop etroite" if creux < 0.85 else
                "trop large" if creux > 1.15 else
                "biaisee vers le haut" if moitie > 0.58 else
                "biaisee vers le bas" if moitie < 0.42 else "calibree")
        print(f"{reg:7s} {d['q_obs'].shape[1]:8d} {v.size:10,} {m['delta']:7.3f} "
              f"{m['chi2']:9.1f} {sous:5.1f}% {sur:5.1f}%  {lect}")
        par_region[reg] = cnt
        tout.append(v)
    if tout:
        v = np.concatenate(tout)
        cnt, bords = pit_histogram(v, n_bins=a.bins)
        m = flatness_metrics(cnt)
        print(f"\nPROVINCE : {v.size:,} couples station-jour | delta {m['delta']:.3f} | "
              f"chi2 {m['chi2']:.1f} | sous l'enveloppe {100 * (v <= 1e-9).mean():.1f} % | "
              f"au-dessus {100 * (v >= 1 - 1e-9).mean():.1f} %")
        os.makedirs(os.path.dirname(a.sortie), exist_ok=True)
        np.savez_compressed(a.sortie, bords=bords, province=cnt,
                            **{f"r_{k}": v_ for k, v_ in par_region.items()})
        print(f"histogrammes sauves dans {a.sortie}")


if __name__ == "__main__":
    main()
