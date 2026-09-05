"""Banc de trois minutes : quels parametres appris fabriquent un plateau d'ete ?

POURQUOI (audit du 2026-09-05, R83). La flotte du 4 septembre produit des plateaux d'ete de
30 a 143 jours pendant lesquels le debit simule ne repond a aucune pluie. La relecture de la
colonne designe deux mecanismes : (A) la troisieme couche seule porte le debit, drainage
lineaire de constante 1/krec, hypodermique eteint par K_sat_2 bas ou couche seche ;
(B) l'evapotranspiration consomme la pluie d'ete avant la riviere (K_c haut, seuil de
stress). Ce banc les departage sur la colonne isolee ANCREE (Linacre, fonte, seuil et
occupation de la plateforme), forcage reel, huit noeuds gaspesiens, un ete reel.

  python .runs/quebec/banc_plateaux_colonne.py            # tests A et B
  python .runs/quebec/banc_plateaux_colonne.py --test A   # un seul

Mesures sur juin a septembre de l'annee cible, production totale = surface + hypodermique
+ nappe (reservoir lineaire k_gw = 0.08 /j, comme le champ provincial) :
  part hypo, part nappe ; jours plats (variation relative < 1 %) ; plus longue suite ;
  reponse a la plus grosse pluie : surplus de production sur 5 jours rapporte a la pluie ;
  ETR/ETP ; coefficient d'ecoulement de l'ete.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from banc_synthetique import colonne_ancree  # noqa: E402
from meandre.vertical.aquifer import AquiferModule  # noqa: E402
from meandre.utils import paths as _p  # noqa: E402


def forcage_noeuds(reg, idx, debut, fin):
    import pandas as pd
    import xarray as xr
    d = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-hyb.nc")
    t = pd.DatetimeIndex(d["time"].values)
    sel = (t >= debut) & (t <= fin)
    f = d["forcing"].isel(node=idx).values[sel]
    d.close()
    return f, t[sel]


def simuler(col, f, temps, k_gw=0.08):
    n = f.shape[1]
    st = col.init_state(n, theta_init=(0.30, 0.30, 0.30))
    doy = temps.dayofyear.to_numpy()
    aq = AquiferModule()
    S = torch.zeros(n, dtype=torch.float64)
    kg = torch.full((n,), float(k_gw), dtype=torch.float64)
    tt = lambda a: torch.tensor(a, dtype=torch.float64)
    out = {k: [] for k in ("surf", "hypo", "base", "nappe", "etr", "etp", "P")}
    with torch.no_grad():
        for i in range(f.shape[0]):
            _, st, dg = col(tt(f[i, :, 0]), tt(f[i, :, 1]), tt(f[i, :, 2]),
                            tt(f[i, :, 3]), tt(f[i, :, 4]), tt(f[i, :, 5]), float(doy[i]), st)
            qbf, S = aq(dg["prod_base"].double().clamp(min=0), S, kg)
            out["surf"].append(float(dg["prod_surf"].mean()))
            out["hypo"].append(float(dg["prod_hypo"].mean()))
            out["base"].append(float(dg["prod_base"].mean()))
            out["nappe"].append(float(qbf.mean()))
            out["etr"].append(float(dg["etr"].mean()))
            out["etp"].append(float(dg["etp"].mean()))
            out["P"].append(float(f[i, :, 0].mean()))
    return {k: np.array(v) for k, v in out.items()}


def mesures(o, temps, annee):
    m = (temps.year == annee) & (temps.month >= 6) & (temps.month <= 9)
    q = o["surf"][m] + o["hypo"][m] + o["nappe"][m]
    tot = max(q.sum(), 1e-9)
    plat = np.abs(np.diff(q)) / np.maximum(q[:-1], 1e-9) < 0.01
    n_ = mx = 0
    for c in plat:
        n_ = n_ + 1 if c else 0
        mx = max(mx, n_)
    P = o["P"][m]
    j = int(np.argmax(P))
    base = q[max(j - 3, 0):j].mean() if j > 0 else q[j]
    surplus = (q[j:j + 5] - base).sum()
    return dict(hypo=o["hypo"][m].sum() / tot, nappe=o["nappe"][m].sum() / tot,
                plat=plat.mean(), suite=mx, pluie=P[j], reponse=surplus / max(P[j], 1e-9),
                etr_etp=o["etr"][m].sum() / max(o["etp"][m].sum(), 1e-9),
                coef=q.sum() / max(P.sum(), 1e-9), q_moy=q.mean())


def ligne(nom, r):
    print(f"  {nom:34s} hypo {100*r['hypo']:4.0f} % | nappe {100*r['nappe']:4.0f} % | plat {100*r['plat']:4.0f} % "
          f"| suite {r['suite']:3d} j | pluie {r['pluie']:4.0f} mm -> reponse {100*r['reponse']:5.1f} % "
          f"| ETR/ETP {r['etr_etp']:4.2f} | coef {r['coef']:4.2f} | Q {r['q_moy']:4.2f} mm/j", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--region", default="gasp")
    ap.add_argument("--annee", type=int, default=2013)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    col, idx = colonne_ancree(a.region, n_noeuds=8, graine=0)
    f, temps = forcage_noeuds(a.region, idx, f"{a.annee - 1}-01-01", f"{a.annee}-09-30")
    sol, etr = col._static["soil"], col._static["etr"]
    sol["cin"] = torch.tensor(0.03)   # valeur imposee par la recette, pas le defaut du banc
    ks_ref = float(sol["ks2"].mean()) if torch.is_tensor(sol["ks2"]) else float(sol["ks2"])
    print(f"{a.region.upper()} : 8 noeuds ancres | ete {a.annee} | ks2 de reference {ks_ref*24:.3f} m/j "
          f"| krec de reference {float(sol['krec']):.1e} m/h", flush=True)

    if a.test in ("A", "AB"):
        print("Test A : krec x K_sat_2 (K_c = 1, seuils de stress de la texture)", flush=True)
        for krec in (1e-7, 2e-5, 1e-4):
            for div in (1, 30, 1000):
                sol["krec"] = torch.tensor(krec)
                sol["ks2"] = torch.tensor(ks_ref / div)
                r = mesures(simuler(col, f, temps), temps, a.annee)
                ligne(f"krec {krec:.0e} | ks2 / {div:<4d}", r)
        sol["ks2"] = torch.tensor(ks_ref)

    if a.test in ("B", "AB"):
        print("Test B : K_c x seuil de stress (krec = 2e-5, ks2 de reference)", flush=True)
        sol["krec"] = torch.tensor(2e-5)
        cc0 = etr["thetacc"].clone() if torch.is_tensor(etr["thetacc"]) else float(etr["thetacc"])
        for kc in (0.3, 1.0, 1.5):
            for fac in (1.0, 1.4):
                etr["K_c"] = torch.tensor(kc)
                etr["thetacc"] = cc0 * fac
                r = mesures(simuler(col, f, temps), temps, a.annee)
                ligne(f"K_c {kc:.1f} | thetacc x {fac:.1f}", r)


if __name__ == "__main__":
    main()
