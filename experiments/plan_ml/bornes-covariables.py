"""Les indices d'épaisseur des dépôts prédisent-ils l'épaisseur que le champ choisit une fois les
bornes relâchées ? Paramètres par tronçon des affinages (`reach-<reg>-ft-<bras>.npz`, ETL_DUMP_REACH)
croisés avec SIGEOM (épaisseur estimée, parts mince et épaisse), les puits du SIH (profondeur du
roc), les Pédo-paysages (restriction racinaire, nappe), l'IRDA (contact lithique mince) et les
familles de perméabilité SIGEOM.

Deux lectures : la médiane par territoire (ce qu'une borne commune ne peut pas rendre), et une
prédiction de chaque territoire par les autres (régression linéaire sur les covariables
disponibles partout dans les territoires comparés, erreur sur le logarithme de l'épaisseur
contre le témoin qui prédit la médiane des autres territoires).

    .venv/Scripts/python.exe experiments/plan_ml/bornes-covariables.py mont slso outv
"""
import sys

import duckdb
import numpy as np
import pandas as pd

D = "D:/meandre-data/experiments/plan_ml"
A = "D:/meandre-data/derives/auxiliaires"
K = ["region", "troncon"]


def parametres(reg, bras):
    r = np.load(f"{D}/reach-{reg}-ft-{bras}.npz", allow_pickle=True)
    con = duckdb.connect(f"D:/meandre-data/quebec/{reg}.duckdb", read_only=True)
    ids = con.sql("select node_id from nodes order by node_idx").df().node_id.to_numpy()
    con.close()
    return pd.DataFrame({"region": reg, "troncon": ids, f"Z2_{bras}": r["param_Z2"], f"Z3_{bras}": r["param_Z3"], f"por1_{bras}": r["param_porosity_1"]})


def covariables():
    ep = pd.read_parquet(f"{D}/epaisseur-depots-troncons.parquet")
    sih = pd.read_parquet(f"{A}/sih-troncons.parquet")[K + ["log_roc_m", "couv_log_roc_m", "part_non_atteint"]]
    slc = pd.read_parquet(f"{A}/slc-troncons.parquet")[K + ["restriction_cm", "nappe_cm", "couv_restriction_cm"]]
    irda = pd.read_parquet(f"{A}/irda-troncons.parquet")[K + ["lithique", "couv_lithique"]]
    dep = pd.read_parquet(f"{A}/depots-sigeom-troncons.parquet")[K + ["depot_tres_permeable", "depot_permeable", "depot_moyennement_permeable", "depot_peu_permeable", "depot_tres_peu_permeable", "depot_roc", "depot_non_carto"]]
    d = ep.merge(sih, on=K, how="left").merge(slc, on=K, how="left").merge(irda, on=K, how="left").merge(dep, on=K, how="left")
    d["log_epaisseur"] = np.log(d.epaisseur_m)
    d["permeable"] = d.depot_tres_permeable + d.depot_permeable
    d["peu_permeable"] = d.depot_peu_permeable + d.depot_tres_peu_permeable
    return d


def main(regions):
    p = pd.concat([parametres(r, "bornes").merge(parametres(r, "casr"), on=K) for r in regions], ignore_index=True)
    d = p.merge(covariables(), on=K, how="left")
    d["log_Z3"] = np.log(d.Z3_bornes)
    d["log_Z23"] = np.log(d.Z2_bornes + d.Z3_bornes)
    print("Médianes par territoire, bornes relâchées (actuelles entre parenthèses) :")
    for r, g in d.groupby("region"):
        print(f"  {r}: Z2 {g.Z2_bornes.median():.2f} ({g.Z2_casr.median():.2f}) m, Z3 {g.Z3_bornes.median():.2f} ({g.Z3_casr.median():.2f}) m, "
              f"porosité 1 {g.por1_bornes.median():.2f} ({g.por1_casr.median():.2f}) | épaisseur SIGEOM {g.epaisseur_m.median():.2f} m, "
              f"roc SIH {np.exp(g.log_roc_m.median()):.1f} m, mince {g.part_mince.mean():.2f}, perméables {g.permeable.mean():.2f}, peu perméables {g.peu_permeable.mean():.2f}")
    covs = ["log_epaisseur", "part_mince", "part_epais", "log_roc_m", "part_non_atteint", "permeable", "peu_permeable", "depot_roc", "lithique"]
    print("\nCorrélation de rang avec le logarithme de Z3 relâché, par territoire (n tronçons avec la covariable) :")
    rows = []
    for c in covs:
        row = {"covariable": c}
        for r, g in d.groupby("region"):
            ok = g[c].notna()
            row[r] = f"{g.loc[ok, 'log_Z3'].corr(g.loc[ok, c], method='spearman'):+.2f} ({ok.sum()})" if ok.sum() > 30 else "-"
        rows.append(row)
    print(pd.DataFrame(rows).set_index("covariable").to_string())
    # Prédiction de chaque territoire par les autres, covariables SIGEOM disponibles partout.
    X = ["log_epaisseur", "part_mince", "permeable", "peu_permeable", "depot_roc"]
    print("\nPrédiction hors territoire du logarithme de Z3 (erreur absolue médiane ; témoin = médiane des autres) :")
    for r in regions:
        tr = d[(d.region != r)].dropna(subset=X + ["log_Z3"])
        te = d[(d.region == r)].dropna(subset=X + ["log_Z3"])
        if len(tr) < 50 or len(te) < 30:
            continue
        M = np.column_stack([np.ones(len(tr))] + [tr[x] for x in X])
        b = np.linalg.lstsq(M, tr.log_Z3, rcond=None)[0]
        pred = np.column_stack([np.ones(len(te))] + [te[x] for x in X]) @ b
        e = np.median(np.abs(pred - te.log_Z3))
        e0 = np.median(np.abs(tr.log_Z3.median() - te.log_Z3))
        print(f"  {r}: {len(te)} tronçons, erreur {e:.3f} contre témoin {e0:.3f} ({100 * (e0 - e) / e0:+.0f} %), Z3 prédit médian {np.exp(np.median(pred)):.2f} m, appris {te.Z3_bornes.median():.2f} m")


if __name__ == "__main__":
    main(sys.argv[1:])
