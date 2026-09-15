"""Compte provincial de l'effet des prelevements et rejets sur le debit annuel.

Refait le chiffre qui porte la conclusion de la partie prelevements de la presentation :
combien de troncons voient leur debit annuel deplace de plus d'un pour cent, dans un sens
et dans l'autre. Le 2026-09-15, avant correction du double comptage des rejets, ce compte
valait 328 troncons augmentes contre 34 diminues, d'ou la phrase disant que le signal
anthropique est d'abord un ajout d'eau.

    .venv/Scripts/python.exe .runs/quebec/bilan_naturalisation.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.utils import paths as _paths

REGIONS = ["outv", "gasp", "sagu", "mont", "slno", "slso", "abit", "cnda", "cndb",
           "cndc", "cndd", "cnde", "labi", "outm", "vaud"]
# Vaudreuil est ecarte : la region ne porte AUCUNE station et AUCUNE observation, donc elle
# n'a pas de cible pour le recalage du volume et son forcage -budyko ne peut pas exister.
# Le recalage tente le 2026-09-15 a produit une precipitation entierement non definie, sur
# 1 835 532 valeurs. Ses 52 troncons a plus d'un metre cube par seconde ne pesent rien dans
# le compte provincial.
REGIONS = [r for r in REGIONS if r != "vaud"]
SEUIL_Q = 1.0
SEUIL_PCT = 1.0


def main():
    rep = os.environ.get("MEANDRE_RAPPORT", f"{_paths.DATA_ROOT}/quebec/rapport")
    lignes, tous = [], []
    manque = []
    for reg in REGIONS:
        fa, fs = f"{rep}/rap-{reg}-avec.npz", f"{rep}/rap-{reg}-sans.npz"
        if not (os.path.exists(fa) and os.path.exists(fs)):
            manque.append(reg)
            continue
        a, s = np.load(fa, allow_pickle=True), np.load(fs, allow_pickle=True)
        qn = np.clip(s["q_annuel"], 1e-6, None)
        imp = 100.0 * (a["q_annuel"] - s["q_annuel"]) / qn
        m = (qn > SEUIL_Q) & np.isfinite(imp)
        import datetime as _dt
        _age = _dt.datetime.fromtimestamp(min(os.path.getmtime(fa), os.path.getmtime(fs)))
        lignes.append({"region": reg, "écrit": _age.strftime("%m-%d %H:%M"),
                       "troncons": int(m.sum()),
                       "hausse": int((imp[m] > SEUIL_PCT).sum()),
                       "baisse": int((imp[m] < -SEUIL_PCT).sum()),
                       "min_pct": float(imp[m].min()) if m.any() else np.nan,
                       "max_pct": float(imp[m].max()) if m.any() else np.nan})
        tous.append(imp[m])
    if manque:
        print(f"régions absentes : {', '.join(manque)}")
    if not lignes:
        return 1
    d = pd.DataFrame(lignes)
    print(d.to_string(index=False))
    # GARDE-FOU. Un compte provincial melant des caches d'ages differents n'a pas de sens,
    # et c'est le genre de chiffre qui circule ensuite sans son avertissement. Le
    # 2026-09-15, l'Outaouais aval refait le matin cotoyait quatorze regions du 1er
    # septembre, et la somme sortait un rapport de 6,86 qui ne decrivait aucun etat.
    _j = sorted({x[:5] for x in d["écrit"]})
    if len(_j) > 1:
        print(f"ARRET : les caches ne sont pas du même jour ({', '.join(_j)}). "
              f"Un compte provincial mêlant deux états ne décrit aucun des deux.")
        return 2
    t = np.concatenate(tous)
    nh, nb = int((t > SEUIL_PCT).sum()), int((t < -SEUIL_PCT).sum())
    print(f"\nPROVINCE : {len(t)} tronçons à débit naturalisé supérieur à {SEUIL_Q:.0f} m³/s")
    print(f"  débit annuel augmenté de plus de {SEUIL_PCT:.0f} pour cent : {nh}")
    print(f"  débit annuel diminué  de plus de {SEUIL_PCT:.0f} pour cent : {nb}")
    print(f"  effet de {t.min():+.1f} à {t.max():+.1f} pour cent")
    if nb:
        print(f"  rapport hausses sur baisses : {nh / nb:.2f}")
    sortie = os.environ.get("MEANDRE_CACHES", ".reports/quebec/caches")
    if os.path.isdir(sortie):
        d.to_csv(f"{sortie}/bilan-naturalisation.csv", index=False)
        print(f"  cache : {sortie}/bilan-naturalisation.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
