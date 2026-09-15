"""Quel poids donner au terme des variations d'un jour ?

Le banc de perte du 2026-09-15 etablit que la recette du socle prefere une serie lissee sur
sept jours a la serie nette decalee d'un jour, sur 64 pour cent des stations, et que le
terme `w_dq` est le seul qui ne s'y laisse jamais prendre. Reste a fixer son poids, que le
banc ne peut pas donner puisqu'il deforme l'observation elle-meme et que le terme y vaut
presque zero par construction.

Deux facons de le fixer, et on les calcule toutes les deux.

1. PAR LA PART DU TOTAL. Sur des SIMULATIONS REELLES, on cherche le poids pour lequel le
   terme pese une part voulue du total de la perte. C'est un reglage d'echelle, sans
   garantie d'effet.

2. PAR LE RENVERSEMENT DU VERDICT. On cherche le poids minimal pour lequel la perte cesse
   de preferer la serie lissee a la serie nette, sur la mediane des stations puis sur une
   part voulue d'entre elles. C'est le reglage qui repond a la question posee.

    .venv/Scripts/python.exe .runs/quebec/poids_variations.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.training.loss import HydroLoss
from meandre.utils import paths as _paths

REGS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cndb", "cndc"]
POIDS = {"KGE": 1.0, "biais de volume": 0.5, "écart quadratique": 0.1,
         "écart quadratique log": 0.3, "pics": 0.5}
CLE = {"KGE": "w_kge", "biais de volume": "w_pbias", "écart quadratique": "w_mse",
       "écart quadratique log": "w_log_mse", "pics": "w_peak"}


def termes_reels():
    """Valeur de chaque terme sur les SIMULATIONS reelles, station par station."""
    dos = f"{_paths.DATA_ROOT}/quebec/flotte"
    lignes = []
    for reg in REGS:
        f = f"{dos}/q-{reg}-A-v4.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        qo, qs = z["q_obs"], z["q_sim"]
        for j in range(qs.shape[1]):
            o, s = qo[:, j].astype(float), qs[:, j].astype(float)
            m = np.isfinite(o) & np.isfinite(s)
            if m.sum() < 700 or o[m].mean() <= 0:
                continue
            o, s = o[m], s[m]
            ot = torch.tensor(o, dtype=torch.float32).unsqueeze(1)
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(1)
            mk = torch.ones(1, dtype=torch.bool)
            sv = torch.tensor([float(np.var(o))], dtype=torch.float32)
            pt = torch.tensor([float(np.quantile(o, 0.75))], dtype=torch.float32)
            base = dict(w_kge=0.0, w_pbias=0.0, w_mse=0.0, w_nse=0.0, w_nrmse=0.0,
                        w_log_nse=0.0, w_log_mse=0.0, w_dq=0.0, w_fdc_bas=0.0,
                        w_dq_log=0.0, w_peak=0.0, per_station=True, station_var=sv)
            d = {"region": reg, "station": j}
            for nom, cle in list(CLE.items()) + [("variations d'un jour", "w_dq")]:
                extra = {"peak_threshold": pt} if cle == "w_peak" else {}
                fl = HydroLoss(**{**base, cle: 1.0}, **extra)
                with torch.no_grad():
                    r = fl(q_obs=ot, q_sim=st, station_mask=mk)
                d[nom] = float(r[0] if isinstance(r, tuple) else r)
            lignes.append(d)
    return pd.DataFrame(lignes)


def main():
    d = termes_reels()
    if d.empty:
        print("aucune station")
        return 1
    d["total"] = sum(POIDS[k] * d[k] for k in POIDS)
    dq = d["variations d'un jour"]
    print(f"{len(d)} stations, simulations réelles\n")
    print("valeur médiane de chaque terme, sur une simulation réelle")
    for k in list(POIDS) + ["variations d'un jour"]:
        print(f"  {k:<24} {d[k].median():.5f}" + (f"  (poids {POIDS[k]})" if k in POIDS else "  (absent)"))
    print(f"  {'total pondéré':<24} {d.total.median():.5f}")

    for part in (0.05, 0.10, 0.20):
        w = part / (1 - part) * (d.total / np.maximum(dq, 1e-12))
        print(f"\npoids pour que le terme pèse {100 * part:.0f} pour cent du total : "
              f"médiane {w.median():.3f}, q25-q75 {w.quantile(.25):.3f}-{w.quantile(.75):.3f}")

    # 2. Le poids qui renverse le verdict du banc de deformation.
    f = ".reports/quebec/caches/banc-perte.csv"
    if not os.path.exists(f):
        print("\nbanc de déformation absent : second réglage non calculé")
        return 0
    b = pd.read_csv(f)
    ref = b[b.candidat == "net, en retard d'un jour"].set_index(["region", "station"])
    print("\npoids minimal pour que la perte cesse de préférer le défaut à la série nette")
    for cand in ("retard puis lissage 7 j", "retard puis pics rabotés"):
        g = b[b.candidat == cand].set_index(["region", "station"])
        if g.empty:
            continue
        idx = g.index.intersection(ref.index)
        g, r = g.loc[idx], ref.loc[idx]
        tot_g = sum(POIDS[k] * g[k] for k in POIDS)
        tot_r = sum(POIDS[k] * r[k] for k in POIDS)
        ddq = g["variations d'un jour"] - r["variations d'un jour"]
        besoin = (tot_r - tot_g) / np.maximum(ddq, 1e-18)
        pire = besoin[tot_g < tot_r]          # stations ou le defaut est prefere
        if pire.empty:
            print(f"  {cand:<26} aucune station ne préfère le défaut")
            continue
        print(f"  {cand:<26} {len(pire)} stations sur {len(g)} préfèrent le défaut ; "
              f"poids requis médian {pire.median():.2e}, "
              f"pour en corriger 90 pour cent {pire.quantile(.9):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
