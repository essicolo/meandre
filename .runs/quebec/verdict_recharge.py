"""Tableau de verdict des variantes de drainage profond, une ligne par variante.

Chaque variante modifie le taux de percolation au bas de la colonne, le temps de séjour de
l'aquifère, ou la courbure de la loi de drainage. Le critère du chantier demande quatre choses
en même temps, et une variante qui n'en tient que trois est rejetée :

- recharge annuelle entre 100 et 250 mm, l'ordre de grandeur mesuré au Québec méridional ;
- couche 3 désaturée, faute de quoi aucune loi dépendant de sa teneur en eau n'agit ;
- débit conservé, jugé sur l'écart-type rapporté à celui du témoin et sur le KGE ;
- nappe en phase, maximum simulé en avril ou mai et corrélation des anomalies positive.

    .venv/bin/python .runs/quebec/verdict_recharge.py outv slno --variantes krec2-sejour100 krec3-sejour100
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_racine = os.environ.get("MEANDRE_DERIVES")
if not _racine:
    from meandre.utils import paths as _paths

    _racine = _paths.DERIVED_ROOT
DERIVES = f"{_racine}/auxiliaires"


def kge_du_log(region, variante):
    """KGE médian tenu de côté, lu dans le journal de la passe."""
    f = f"{DERIVES}/eval-{region}-{variante}.log"
    if not os.path.exists(f):
        return np.nan
    m = re.findall(r"HELD-OUT.*médian\s+([-0-9.]+)", open(f, encoding="utf-8", errors="replace").read())
    return float(m[-1]) if m else np.nan


def mesures_region(region, variante, temoin):
    """Recharge, saturation de la couche 3 et variabilité du débit, rapportées au témoin."""
    f = f"{DERIVES}/reach-{region}-{variante}-journalier.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    r = np.load(f"{DERIVES}/reach-{region}-{variante}.npz", allow_pickle=True)
    ths = r["param_porosity_3"]
    mois = np.array([int(d[5:7]) for d in z["dates"]])
    idx = np.argsort(z["q"].mean(0))[-40:]
    rech = z["recharge"].mean() * 365.25
    om = np.median(z["theta3"] / ths[None, :])
    cyc = np.array([z["recharge"][mois == m].mean() for m in range(1, 13)])
    tnt = z["temps_non_traite"] if "temps_non_traite" in z.files else None
    sortie = {}
    # Eau gravitaire de la couche 3 : la lame retenue au-dessus de la capacite au champ,
    # celle que la gravite draine par definition. 289 a 465 mm au temoin selon le territoire.
    if "param_theta_fc_3" in r.files and "param_Z3" in r.files:
        sortie["eau_gravitaire_mm"] = float(
            np.mean((z["theta3"].mean(axis=0) - r["param_theta_fc_3"]) * r["param_Z3"]) * 1000.0)
    if "profondeur_nappe" in z.files:
        pn = z["profondeur_nappe"]
        cyc_n = np.array([pn[mois == m].mean() for m in range(1, 13)])
        sortie["nappe_m"] = float(pn.mean())
        sortie["battement_m"] = float(cyc_n.max() - cyc_n.min())
        sortie["mois_nappe_haute"] = int(cyc_n.argmin() + 1)
    if tnt is not None:
        cyc_t = np.array([tnt[mois == m].mean() for m in range(1, 13)])
        sortie["part_jour_non_traite"] = float(tnt.mean())
        sortie["part_non_traite_avril"] = float(cyc_t[3])
    return {**sortie,
            "recharge_mm_an": rech, "omega_couche3": om, "mois_max_recharge": int(cyc.argmax() + 1),
            "ecart_type_debit": z["q"][:, idx].std(0).mean() / temoin["ecart_type_debit_brut"],
            "debit_moyen": z["q"][:, idx].mean() / temoin["debit_moyen_brut"],
            "etr_mm_an": z["etr"].mean() * 365.25}


def temoin_region(region):
    z = np.load(f"{DERIVES}/reach-{region}-nappe-journalier.npz", allow_pickle=True)
    idx = np.argsort(z["q"].mean(0))[-40:]
    return {"ecart_type_debit_brut": z["q"][:, idx].std(0).mean(), "debit_moyen_brut": z["q"][:, idx].mean()}


def nappe_variante(regions, variante):
    """Reprend les métriques de puits déjà calculées par comparer_nappe."""
    f = f"{DERIVES}/comparaison-nappe-{variante}.parquet"
    if not os.path.exists(f):
        return {}
    t = pd.read_parquet(f)
    t = t[t.region.isin(regions)]
    if t.empty:
        return {}
    return {"r_anomalies": t.r_anomalies.median(), "mois_max_nappe": int(t.mois_max_sim.mode().iloc[0]),
            "amplitude_sim_mm": t.amplitude_sim_mm.median(), "n_puits": len(t)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("regions", nargs="+")
    p.add_argument("--variantes", nargs="+", required=True)
    a = p.parse_args()
    regions = [r.lower() for r in a.regions]
    temoins = {r: temoin_region(r) for r in regions}
    lignes = []
    for v in ["nappe"] + a.variantes:
        par_region = [m for r in regions if (m := mesures_region(r, v, temoins[r])) is not None]
        if not par_region:
            print(f"{v} : aucune sortie")
            continue
        ligne = {"variante": v, "n_regions": len(par_region)}
        for k in par_region[0]:
            ligne[k] = float(np.median([m[k] for m in par_region]))
        for _k in ("mois_max_recharge", "mois_nappe_haute"):
            if _k in ligne:
                ligne[_k] = int(ligne[_k])
        ligne["kge"] = float(np.median([kge_du_log(r, v) for r in regions]))
        ligne.update(nappe_variante(regions, "" if v == "nappe" else v))
        lignes.append(ligne)
    t = pd.DataFrame(lignes)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    t.to_parquet(f"{DERIVES}/verdict-recharge.parquet", index=False)
    print(f"\n{DERIVES}/verdict-recharge.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
