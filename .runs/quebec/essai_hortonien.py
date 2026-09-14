"""L'exces d'infiltration sous-journalier rend-il les pointes ? Passe avant seule.

Le pic simule vaut 0,59 du pic observe alors que la pluie recue n'a pas de biais, et la
colonne isolee ne produit que 0,4 % d'ecoulement rapide sous une averse de 40 mm contre
5 a 30 % attendus. Le clone porte deja le mecanisme manquant, l'exces d'infiltration
sous-journalier, inerte faute d'un canal de duree d'orage dans le forcage quebecois.

Ce script compare deux simulations du meme sous-bassin, sans aucun entrainement : le
forcage a six canaux contre le forcage a sept, mecanisme actif. Meme graine, memes
ancrages, meme sol impose. Le juge est le rapport des pointes annuelles et l'erreur
d'amplitude par evenement, non le KGE, qu'un champ non entraine ne renseigne pas.

    .venv/Scripts/python.exe .runs/quebec/essai_hortonien.py gasp 011003
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def erreur_amplitude(o, s, seuil=None):
    """Biais et dispersion du logarithme du rapport des pics, evenement par evenement."""
    m = np.isfinite(o) & np.isfinite(s) & (o > 0)
    if m.sum() < 300:
        return np.nan, np.nan, 0
    oo = np.where(m, o, np.nan)
    seuil = np.nanquantile(oo, 0.90) if seuil is None else seuil
    idx = [i for i in range(1, len(o) - 1)
           if m[i] and o[i] > seuil and o[i] >= o[i - 1] and o[i] >= o[i + 1]]
    garde = []
    for i in idx:
        if not garde or i - garde[-1] >= 7:
            garde.append(i)
        elif o[i] > o[garde[-1]]:
            garde[-1] = i
    r = []
    for i in garde:
        lo, hi = max(0, i - 2), min(len(s), i + 3)
        if np.isfinite(s[lo:hi]).any():
            r.append(np.log(np.nanmax(s[lo:hi]) / o[i]))
    if len(r) < 8:
        return np.nan, np.nan, len(r)
    r = np.array(r)
    return float(np.exp(r.mean())), float(r.std()), len(r)


def main(reg, station):
    import banc_sousbassin as banc
    from meandre.model import HydroModel

    res = {}
    for nom, sfx, horton in (("six canaux, mécanisme inerte", "-budyko", False),
                             ("sept canaux, mécanisme actif", "-budyko-dt", True)):
        os.environ["JOINT_FX_SUFFIX"] = sfx
        os.environ["ETL_SEED"] = "1234"
        _orig = HydroModel.__init__

        def _init(self, *a, **kw):
            kw["use_hortonian"] = horton
            _orig(self, *a, **kw)

        HydroModel.__init__ = _init
        try:
            t, q, o = banc.simuler(reg, station, melt_saison=0.5, sol="sauf_ks", annees=6)
        finally:
            HydroModel.__init__ = _orig
        g = t.year > t.year.min()
        k, r, b, gm = banc._kge(q[g], o[g])
        f = banc.forme(t, q, o, g)
        ete = np.isin(t.month, (6, 7, 8, 9)) & g
        bi, di, n = erreur_amplitude(o[g], q[g])
        bie, die, ne = erreur_amplitude(np.where(ete, o, np.nan), np.where(ete, q, np.nan))
        res[nom] = dict(kge=k, r=r, beta=b, gamma=gm, pic=f["pic"], q99=f["q99"],
                        plat_ete=f["plat_ete"], biais=bi, disp=di, n=n,
                        biais_ete=bie, disp_ete=die, n_ete=ne)
        print(f"  {nom}")
        print(f"    KGE {k:.3f} | r {r:.3f} | beta {b:.3f} | gamma {gm:.3f}")
        print(f"    pointes annuelles {f['pic']:.2f} | q99 {f['q99']:.2f} | platitude d'été {f['plat_ete']:.1f} %")
        print(f"    amplitude des événements : rapport {bi:.2f}, dispersion {di:.2f} sur {n} pics")
        print(f"    en été seulement        : rapport {bie:.2f}, dispersion {die:.2f} sur {ne} pics", flush=True)

    a, b_ = list(res.values())
    print("\n  écart, mécanisme actif moins inerte")
    for c, nom in (("pic", "pointes annuelles"), ("biais", "rapport d'amplitude"),
                   ("biais_ete", "rapport en été"), ("disp", "dispersion"),
                   ("gamma", "variabilité"), ("plat_ete", "platitude d'été")):
        if np.isfinite(a[c]) and np.isfinite(b_[c]):
            print(f"    {nom:22s} {a[c]:7.2f} -> {b_[c]:7.2f}  ({b_[c] - a[c]:+.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gasp",
         sys.argv[2] if len(sys.argv) > 2 else "011003")
