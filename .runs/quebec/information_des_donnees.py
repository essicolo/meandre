"""Combien de directions indépendantes TOUTES les observations portent-elles ensemble.

Le champ spatial produit 43 paramètres par tronçon, soit plusieurs milliers d'inconnues par
territoire. La question n'est pas combien d'observations on possède, mais combien de
DIRECTIONS INDÉPENDANTES elles contraignent, et si ces directions sont distinctes les unes
des autres. Une contrainte qui duplique le débit ne lève aucune équifinalité du débit.

Même mesure que pour la neige : moyennes mensuelles, corrélation sur les mois communs à
chaque paire, composantes à 95 % de la variance, rapport de participation, et recouvrement
des quatre premières directions entre les deux moitiés de la période, qui sépare une
direction indépendante d'un bruit indépendant.

Le débit entre en LOGARITHME : c'est l'échelle sur laquelle ses variations sont comparables
d'une station à l'autre, un bassin de mille kilomètres carrés et un de dix ne se comparant
pas en mètres cubes par seconde.

    .venv/bin/python .runs/quebec/information_des_donnees.py outv
"""
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_ici = os.path.dirname(os.path.abspath(__file__))
_inf = SourceFileLoader("inf", os.path.join(_ici, "information_de_la_neige.py")).load_module()

from meandre.utils import paths as _paths

DERIVES = f"{os.environ.get('MEANDRE_DERIVES', _paths.DERIVED_ROOT)}/auxiliaires"
DATE_START, DATE_END = "2000-01-01", "2024-12-31"


def _serie_large(df, col_id, col_date, col_val, axe):
    """Passe une table longue en matrice jours x entites, alignée sur l'axe demandé."""
    df = df.dropna(subset=[col_val])
    if df.empty:
        return None
    t = pd.to_datetime(df[col_date]).dt.normalize()
    large = pd.DataFrame({"t": t, "id": df[col_id].astype(str), "v": df[col_val].astype(float)})
    large = large.groupby(["t", "id"], as_index=False)["v"].mean()
    m = large.pivot(index="t", columns="id", values="v").reindex(axe)
    return m.to_numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region")
    a = p.parse_args()
    reg = a.region.lower()
    import duckdb

    axe = pd.date_range(DATE_START, DATE_END, freq="D")
    tout = pd.Series(True, index=axe).to_numpy()
    hiver = np.isin(axe.month, (11, 12, 1, 2, 3, 4, 5))
    con = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)

    sources = []
    q = con.execute("SELECT station_id, date, discharge FROM observations").df()
    m = _serie_large(q, "station_id", "date", "discharge", axe)
    if m is not None:
        sources.append(("debit (log)", np.log(np.clip(m, 1e-3, None)), tout))

    g = con.execute("SELECT date, tws_mm FROM grace_tws WHERE quality_ok").df()
    mg = _serie_large(g.assign(id="bassin"), "id", "date", "tws_mm", axe)
    if mg is not None:
        sources.append(("GRACE, moyenne du bassin", mg, tout))

    try:
        e = con.execute("SELECT node_idx, date, etr_mm_day FROM modis_et WHERE quality_ok").df()
        me = _serie_large(e, "node_idx", "date", "etr_mm_day", axe)
        if me is not None:
            sources.append(("evapotranspiration MODIS", me, tout))
    except Exception as exc:
        print(f"[modis_et] non lu : {exc}")
    con.close()

    f = f"{DERIVES}/neisim-{reg}.npz"
    if os.path.exists(f):
        sources.append(("neige NEISIM", np.load(f)["valeurs"], hiver))
    pu = f"{DERIVES}/rsesq-niveaux-journaliers.parquet"
    if os.path.exists(pu):
        d = pd.read_parquet(pu)
        d = d[d["region"].astype(str).str.lower() == reg]
        # Memes ecarts que le chargeur de la perte : un puits captif mesure une charge et
        # non un stock, un puits sous pompage mesure l'exploitation.
        d = d[(d["confinement"].astype(str) != "Captive") & (d["influence"].astype(str) == "Non")]
        mn = _serie_large(d, "puits", "date", "niveau_m", axe)
        if mn is not None:
            sources.append(("nappes RSESQ", mn, tout))

    print(f"{reg} : directions independantes par source\n")
    print(f"{'source':<26s} {'series':>8s} {'95 %':>7s} {'participation':>14s} "
          f"{'reproductible':>14s} {'hasard':>8s}")
    for nom, mat, masque in sources:
        mm = _inf._mensualise(mat, axe, masque)
        n95, part, cols = _inf._rang_effectif(mm)
        rep, pp = _inf._reproductibilite(mm)
        n95_t = f"{n95:>7d}" if np.isfinite(n95) else f"{'—':>7s}"
        part_t = f"{part:>14.1f}" if np.isfinite(part) else f"{'—':>14s}"
        rep_t = f"{rep:>14.2f}" if np.isfinite(rep) else f"{'—':>14s}"
        haz = f"{4.0 / pp:>8.2f}" if pp and np.isfinite(rep) else f"{'—':>8s}"
        print(f"{nom:<26s} {cols:>8d} {n95_t} {part_t} {rep_t} {haz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
