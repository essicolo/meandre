"""L'exposant de Campbell du calage d'Hydrotel est-il une propriété de texture.

La capacité au champ se déduit de la courbe de rétention imposée, theta = thetas
(psi_s/psi)^(1/b). La déduction ne vaut donc que si la courbe elle-même est une propriété du
sol. Or l'exposant b du calage est aussi un levier d'ajustement, et rien ne garantit qu'il
soit resté dans la plage des textures réelles : chez Clapp et Hornberger, b vaut 4,05 pour un
sable, 4,90 pour un loam sableux, 5,39 pour un loam et jusqu'à 11,4 pour une argile. Aucune
texture ne descend sous 4.

Mesuré le 2026-09-22 : déduire la capacité au champ coûte 0,394 de KGE en Gaspésie contre
0,041 en Outaouais, sans aucune erreur d'unité. Ce banc vérifie si l'exposant en est la cause.

    .venv/bin/python .runs/quebec/courbe_de_retention.py gasp outv slno sagu
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.data.hydrotel_calib import load_calibrated_soil
from meandre.utils import paths as _paths

PSI_FC = 3.37
PLATEFORMES = os.environ.get("MEANDRE_PLATEFORMES", f"{_paths.DATA_ROOT}/plateformes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    a = p.parse_args()
    import duckdb

    print(f"{'territoire':<12s} {'couche':>7s} {'b median':>9s} {'part b<4':>9s} "
          f"{'psi_s med':>10s} {'capacite deduite':>17s}")
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
        for k in (1, 2, 3):
            b = cs[f"b{k}"].float()
            ps = cs[f"psis{k}"].float().abs()
            th = cs[f"thetas{k}"].float()
            om = ((ps.clamp(min=1e-6) / PSI_FC) ** (1.0 / b.clamp(min=0.1))).clamp(0.05, 0.95)
            tfc = th * om
            print(f"{reg:<12s} {k:>7d} {float(b.median()):>9.2f} "
                  f"{float((b < 4).float().mean()):>9.2f} {float(ps.median()):>10.3f} "
                  f"{float(tfc.median()):>17.3f}")
    print("")
    print("Clapp et Hornberger : b = 4,05 pour un sable, 5,39 pour un loam, 11,4 pour une")
    print("argile. Aucune texture reelle ne descend sous 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
