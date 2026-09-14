"""Etalement au versant plus canal advectif : rend-il sa nervosite a l'hydrogramme ?

Mesure du 2026-09-14 : la colonne produit une eau trois fois plus nerveuse que la riviere
observee, le reseau en retire 87 pour cent, et il reste 40 pour cent de l'observe. Le
sur-lissage nait donc entierement entre la production et l'exutoire.

Le depot porte depuis juin le mecanisme prevu pour cela, et il est debranche. L'hydrogramme
unitaire de versant est une cascade de deux reservoirs de Nash qui etale les composantes
rapides AVANT le canal, avec deux echelles apprenables, 0,3 jour pour le ruissellement de
surface et 2,5 jours pour l'hypodermique, le debit de base passant directement. Il est
concu pour aller avec un canal purement advectif, sans attenuation. Ni l'un ni l'autre
n'apparait dans la recette du socle.

Quatre configurations, en passe avant seule, meme graine, meme sol impose :
  la recette actuelle, ni versant ni advection ;
  le canal advectif seul ;
  le versant seul ;
  le couple, qui est la configuration prevue.

Le juge est la variabilite des variations d'un jour a l'autre rapportee a l'observee, et
le rapport des pointes annuelles, qu'un canal sans attenuation pourrait exagerer.

    .venv/Scripts/python.exe .runs/quebec/essai_versant.py sagu 062803
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def variabilite(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 100 or x.mean() <= 0:
        return np.nan
    return float(np.std(np.diff(x)) / x.mean())


def main(reg, station):
    import banc_sousbassin as banc
    from meandre.model import HydroModel

    os.environ.setdefault("JOINT_FX_SUFFIX", "-budyko")
    os.environ.setdefault("ETL_SEED", "1234")
    _init = HydroModel.__init__
    res = {}
    for nom, uh, adv in (("recette actuelle", False, False),
                         ("canal advectif seul", False, True),
                         ("versant seul", True, False),
                         ("versant et advection", True, True)):
        def _i(self, *a, _i=_init, _uh=uh, _adv=adv, **kw):
            kw["use_hillslope_uh"] = _uh
            kw["pure_advection"] = _adv
            _i(self, *a, **kw)
        HydroModel.__init__ = _i
        try:
            t, q, o = banc.simuler(reg, station, melt_saison=0.5, sol="sauf_ks",
                                   annees=6, verbeux=False)
        finally:
            HydroModel.__init__ = _init
        an = np.asarray(t.year)
        g = (an > an.min()) & np.isfinite(o) & np.isfinite(q)
        prod = sum(banc._DERNIERE_PARTITION.values())
        k, r, b, gm = banc._kge(q[g], o[g])
        f = banc.forme(t, q, o, g)
        vq, vo, vp = variabilite(q[g]), variabilite(o[g]), variabilite(prod[g])
        res[nom] = dict(kge=k, r=r, beta=b, gamma=gm, pic=f["pic"], q99=f["q99"],
                        nerf=vq / vo, retire=vq / vp)
        print(f"  {nom:22s} nervosité {vq / vo:.2f} | réseau garde {vq / vp:.2f} | "
              f"KGE {k:.3f} | r {r:.3f} | gamma {gm:.3f} | pointes {f['pic']:.2f} | q99 {f['q99']:.2f}",
              flush=True)
    print("\n  nervosité = variabilité des variations d'un jour, simulée sur observée ; cible 1,00")
    base = res["recette actuelle"]
    print(f"  écart à la recette actuelle :")
    for nom, v in res.items():
        if nom == "recette actuelle":
            continue
        print(f"    {nom:22s} nervosité {v['nerf'] - base['nerf']:+.2f} | "
              f"KGE {v['kge'] - base['kge']:+.3f} | r {v['r'] - base['r']:+.3f} | "
              f"pointes {v['pic'] - base['pic']:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "sagu",
                  sys.argv[2] if len(sys.argv) > 2 else "062803"))
