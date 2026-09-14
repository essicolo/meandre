"""Controle du canal de duree d'averse avant de s'en servir.

Le canal DT_eff pilote l'exces d'infiltration sous-journalier, et la colonne y est tres
sensible : deux heures d'averse font passer l'ecoulement rapide de 0,4 a 66 pour cent sur
40 mm. Un canal faux produirait donc des pointes fausses sans que rien ne le signale. Ce
controle verifie les proprietes que la grandeur doit avoir par construction et par
physique, avant toute simulation.

    .venv/Scripts/python.exe .runs/quebec/controle_dt_eff.py gasp
"""
import sys

import numpy as np
import pandas as pd
import xarray as xr

from meandre.utils import paths as _p

SEUILS = dict(mediane_min=1.5, mediane_max=12.0, part_24h_max=0.15, ete_sous_hiver=True)


def main(reg, sfx="-budyko-dt"):
    f = f"{_p.DATA_ROOT}/quebec/forcing-{reg}-{sfx.lstrip('-')}.nc"
    ds = xr.open_dataset(f)
    noms = [str(v) for v in ds["var"].values]
    assert noms[-1] == "DT_eff", f"dernier canal {noms[-1]}, attendu DT_eff"
    t = pd.DatetimeIndex(ds["time"].values)
    P = ds["forcing"].values[..., 0]
    D = ds["forcing"].values[..., noms.index("DT_eff")]
    ds.close()
    print(f"{reg.upper()} : {f.split('/')[-1]}")
    print(f"  canaux {noms}")
    ok = True

    fini = np.isfinite(D)
    print(f"  valeurs finies {100 * fini.mean():.2f} % | bornes [{np.nanmin(D):.2f}, {np.nanmax(D):.2f}] h")
    if not fini.all() or np.nanmin(D) < 1.0 - 1e-6 or np.nanmax(D) > 24.0 + 1e-6:
        print("  ECHEC : valeurs non finies ou hors des bornes [1, 24]")
        ok = False

    pluie = P > 1.0
    d = D[pluie]
    med = float(np.median(d))
    part24 = float(np.mean(d >= 23.9))
    print(f"  jours de plus de 1 mm : {100 * pluie.mean():.0f} % des couples jour-tronçon")
    print(f"    durée médiane {med:.1f} h | part sous 3 h {100 * np.mean(d < 3.0):.0f} % | "
          f"part à 24 h {100 * part24:.0f} %")
    if not (SEUILS["mediane_min"] <= med <= SEUILS["mediane_max"]):
        print(f"  ECHEC : médiane hors de [{SEUILS['mediane_min']}, {SEUILS['mediane_max']}] h")
        ok = False
    if part24 > SEUILS["part_24h_max"]:
        print("  ECHEC : trop de jours à 24 h, le maximum horaire est probablement nul ou absent")
        ok = False

    # Les jours SANS pluie doivent porter la valeur de repli, 24 h, et non une duree courte
    # qui ferait ruisseler une pluie inexistante.
    sec = P <= 0.01
    if sec.any():
        print(f"    jours secs : durée médiane {np.median(D[sec]):.1f} h")
        if np.median(D[sec]) < 23.9:
            print("  ECHEC : les jours secs doivent porter 24 h")
            ok = False

    mois = t.month.values
    ete = np.isin(mois, (6, 7, 8))
    hiver = np.isin(mois, (11, 12, 1, 2))
    de = D[ete][:, None].ravel() if D.ndim == 1 else D[ete][P[ete] > 1.0]
    dh = D[hiver][:, None].ravel() if D.ndim == 1 else D[hiver][P[hiver] > 1.0]
    if len(de) and len(dh):
        print(f"    été {np.median(de):.1f} h | hiver {np.median(dh):.1f} h")
        if SEUILS["ete_sous_hiver"] and np.median(de) >= np.median(dh):
            print("  ECHEC : l'été doit concentrer ses averses plus que l'hiver, "
                  "les précipitations convectives étant estivales")
            ok = False

    # Une averse plus grosse doit etre plus concentree, non l'inverse.
    fort = P > 20.0
    if fort.sum() > 100:
        print(f"    pluies de plus de 20 mm : durée médiane {np.median(D[fort]):.1f} h "
              f"({int(fort.sum())} couples)")

    print(f"\n  {'CONTROLE REUSSI' if ok else 'CONTROLE ECHOUE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "gasp"))
