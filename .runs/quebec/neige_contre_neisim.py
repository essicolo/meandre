"""La neige simulée, comparée à NEISIM sur TOUS les nœuds d'un territoire.

Le réseau au sol compte 76 sites en Outaouais et aucun au Saint-Laurent sud-ouest : la
comparaison de la masse du manteau n'existait que par endroits, et pas du tout là où le
réseau se tait. NEISIM couvre toute la grille et s'accorde aux relevés au sol à 0,98 de
rapport médian et 0,87 de corrélation par site (`neisim_contre_canswe.py`). Il permet donc
de dire, nœud par nœud, où le manteau simulé s'écarte, et de combien.

La comparaison porte sur trois grandeurs que la sortie par tronçon porte déjà, sans qu'il
faille resimuler : le maximum annuel moyen, la climatologie mensuelle, et la série mensuelle
qui porte l'interannuel.

    .venv/bin/python .runs/quebec/neige_contre_neisim.py outv --variantes essai3-vigueur
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"


def _stats_neisim(reg, node_idx_attendus=None):
    """Les trois mêmes grandeurs, calculées sur NEISIM."""
    f = f"{DERIVES}/neisim-{reg}.npz"
    if not os.path.exists(f):
        raise SystemExit(f"cible NEISIM absente : {f}")
    z = np.load(f)
    v, idx = z["valeurs"], z["node_idx"]
    times = pd.DatetimeIndex(z["times"])
    ans, mois = times.year.to_numpy(), times.month.to_numpy()
    max_annuel = np.stack([np.nanmax(v[ans == a], axis=0) for a in np.unique(ans)]).mean(axis=0)
    mensuel = np.stack([np.nanmean(v[mois == m], axis=0) for m in range(1, 13)])
    cle = times.year.to_numpy() * 100 + mois
    mois_u = np.unique(cle)
    serie = np.stack([np.nanmean(v[cle == c], axis=0) for c in mois_u])
    return max_annuel, mensuel, serie, idx, mois_u


def _correlation_par_noeud(a, b):
    """Corrélation colonne par colonne, NaN là où une des deux séries est plate."""
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    ok = (na > 1e-9) & (nb > 1e-9)
    out = np.full(a.shape[1], np.nan)
    out[ok] = (a[:, ok] * b[:, ok]).sum(axis=0) / (na[ok] * nb[ok])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    p.add_argument("--variantes", nargs="+", required=True)
    a = p.parse_args()
    reg = a.region.lower()

    n_max, n_mens, n_serie, idx, mois_u = _stats_neisim(reg)
    print(f"{reg} : NEISIM sur {len(idx)} nœuds, {len(mois_u)} mois\n")
    print(f"maximum annuel moyen de NEISIM : mediane par nœud {np.nanmedian(n_max):.0f} mm, "
          f"quartiles {np.nanpercentile(n_max, 25):.0f} a {np.nanpercentile(n_max, 75):.0f} mm")
    print("")
    print(f"{'variante':<28s} {'max simule':>11s} {'rapport':>8s} {'mois du max':>12s} "
          f"{'r saisonnier':>13s} {'r interannuel':>14s}")

    for var in a.variantes:
        f = f"{DERIVES}/reach-{reg}-{var}.npz"
        if not os.path.exists(f):
            print(f"{var:<28s} {'sortie absente':>12s}")
            continue
        z = np.load(f)
        if "swe_max_annuel" not in z:
            print(f"{var:<28s} {'pas de neige dans la sortie':>12s}")
            continue
        s_max = z["swe_max_annuel"][idx]
        s_mens = z["swe_mensuel"][:, idx]
        rapport = np.nanmedian(s_max / np.maximum(n_max, 1e-6))
        mois_sim = int(np.bincount(np.argmax(s_mens, axis=0)).argmax()) + 1
        mois_nei = int(np.bincount(np.argmax(n_mens, axis=0)).argmax()) + 1
        r_sais = np.nanmedian(_correlation_par_noeud(s_mens, n_mens))
        if "swe_mois_serie" in z and z["swe_mois_serie"].shape[0] == len(mois_u):
            r_inter = np.nanmedian(_correlation_par_noeud(z["swe_mois_serie"][:, idx], n_serie))
            inter = f"{r_inter:>14.2f}"
        else:
            inter = f"{'—':>14s}"
        print(f"{var:<28s} {np.nanmedian(s_max):>8.0f} mm {rapport:>8.2f} "
              f"{mois_sim:>7d} / {mois_nei:<2d} {r_sais:>13.2f} {inter}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
