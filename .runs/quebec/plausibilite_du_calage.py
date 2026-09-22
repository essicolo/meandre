"""Chaque pièce du calage imposé correspond-elle à un sol réel.

La loi des ancrages impose par nœud tout le calage du sol d'Hydrotel sauf les conductivités et
les porosités. Sa révision d'août l'a mesurée sur le KGE ; personne n'a vérifié que ce qu'elle
impose décrive un sol qui existe. Le 2026-09-22, déduire la capacité au champ de la courbe
imposée a coûté 0,394 de KGE en Gaspésie, et la cause était que l'exposant de Campbell y vaut
1,99 quand aucune texture réelle ne descend sous 4.

Ce banc étend le contrôle à toutes les pièces, et il ne se contente pas de vérifier chaque
paramètre séparément : il cherche, pour chaque nœud, la texture de Clapp et Hornberger la plus
proche et mesure la distance au triplet (b, psi_s, porosité). Un paramètre peut être dans sa
plage et le TRIPLET ne correspondre à aucun sol, ces trois grandeurs covariant avec la texture.

    .venv/bin/python .runs/quebec/plausibilite_du_calage.py gasp outv slno sagu abit cndc
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.hydrotel_calib import load_calibrated_soil
from meandre.utils import paths as _paths

PLATEFORMES = os.environ.get("MEANDRE_PLATEFORMES", f"{_paths.DATA_ROOT}/plateformes")

# Clapp et Hornberger 1978, tableau 2 : porosite, succion d'entree en metres, exposant b.
# Les onze textures couvrent tout le triangle textural.
TEXTURES = {
    "sable": (0.395, 0.121, 4.05), "sable loameux": (0.410, 0.090, 4.38),
    "loam sableux": (0.435, 0.218, 4.90), "loam limoneux": (0.485, 0.786, 5.30),
    "loam": (0.451, 0.478, 5.39), "loam argilo-sableux": (0.420, 0.299, 7.12),
    "loam argilo-limoneux": (0.477, 0.356, 7.75), "loam argileux": (0.476, 0.630, 8.52),
    "argile sableuse": (0.426, 0.153, 10.4), "argile limoneuse": (0.492, 0.490, 10.4),
    "argile": (0.482, 0.405, 11.4)}


def _texture_la_plus_proche(thetas, psis, b):
    """Texture la plus proche au sens d'une distance relative sur les trois grandeurs."""
    noms = list(TEXTURES)
    ref = np.array([TEXTURES[n] for n in noms])
    x = np.stack([thetas, psis, b], axis=1)
    d = np.abs(x[:, None, :] - ref[None, :, :]) / ref[None, :, :]
    d = d.mean(axis=2)
    j = d.argmin(axis=1)
    return [noms[k] for k in j], d[np.arange(len(j)), j]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    a = p.parse_args()
    import duckdb

    print(f"{'territoire':<12s} {'grandeur':<14s} {'mediane':>9s} {'min':>8s} {'max':>8s} "
          f"{'plage litterature':>20s} {'hors plage':>11s}")
    plages = {"b": (4.05, 11.4), "psis": (0.090, 0.786), "z2": (0.05, 3.0), "z3": (0.1, 6.0),
              "slope": (0.0, 1.0)}
    distances = {}
    for reg in [x.lower() for x in a.regions]:
        maj = reg.upper()
        try:
            c = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
            ids = [r[0] for r in c.execute("SELECT node_id FROM nodes ORDER BY node_idx").fetchall()]
            c.close()
            cs = load_calibrated_soil(f"{PLATEFORMES}/LN24HA/{maj}_LN24HA_2020", ids, 0.15,
                                      device="cpu")
        except Exception as exc:
            print(f"{reg:<12s} erreur : {type(exc).__name__} {exc}")
            continue
        for cle, nom in (("b1", "b"), ("psis1", "psis"), ("z2", "z2"), ("z3", "z3"),
                         ("slope", "slope")):
            if cle not in cs:
                continue
            v = cs[cle].float().abs().numpy()
            lo, hi = plages[nom]
            hors = float(((v < lo) | (v > hi)).mean())
            print(f"{reg:<12s} {nom:<14s} {np.median(v):>9.3f} {v.min():>8.3f} {v.max():>8.3f} "
                  f"{f'{lo:g} a {hi:g}':>20s} {100 * hors:>10.0f} %")
        if all(k in cs for k in ("thetas1", "psis1", "b1")):
            noms, d = _texture_la_plus_proche(cs["thetas1"].float().numpy(),
                                              cs["psis1"].float().abs().numpy(),
                                              cs["b1"].float().numpy())
            distances[reg] = (noms, d)
        print("")

    if distances:
        print(f"{'territoire':<12s} {'texture la plus proche':<24s} {'ecart relatif median':>21s} "
              f"{'part au-dela de 20 %':>21s}")
        for reg, (noms, d) in distances.items():
            u, n = np.unique(noms, return_counts=True)
            print(f"{reg:<12s} {u[n.argmax()]:<24s} {np.median(d):>21.2f} "
                  f"{100 * float((d > 0.20).mean()):>20.0f} %")
        print("")
        print("L'ecart relatif est la moyenne des ecarts relatifs sur la porosite, la succion")
        print("d'entree et l'exposant. Au-dela de 20 %, le triplet ne decrit aucune texture.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
