"""Plafond de percolation préféré par station, et ce qui le prédit.

Le plafond de percolation de la couche 3 est le levier le plus fort de l'étiage en passe avant,
mais sa valeur préférée change d'un territoire à l'autre et la texture ne la prédit pas. Une
valeur par territoire serait un calage territorial ; la question est de savoir si une propriété
mesurée du terrain (épaisseur des dépôts, perméabilité des dépôts, texture) la prédit station
par station, auquel cas le plafond peut devenir une règle physique valable partout.

Entrée : les débits simulés des passes avant à plafond uniforme, `q-plafXX-REG.npz`, sur
2019-2024. Par station, la perte de débit (mêmes termes et poids que la ligne PERTE du pilote)
est calculée pour chaque plafond ; le plafond préféré est le minimum d'une parabole en
logarithme du plafond sur les trois points autour du meilleur. Puis chaque territoire est
prédit par un modèle ajusté sur les autres, contre le témoin qui prédit la moyenne des autres.

    .venv/Scripts/python.exe .runs/quebec/plafond_prefere.py <dossier des npz> mont slso slno outv gasp sagu
"""
import os
import re
import sys

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from meandre.training import loss as L
from meandre.utils import paths as _paths

_ici = os.path.dirname(os.path.abspath(__file__))
_ii = SourceFileLoader("ii", os.path.join(_ici, "identifiabilite_indice_base.py")).load_module()
POIDS = {"kge": (1.0, L.differentiable_kge_loss), "pbias": (0.5, L.differentiable_pbias_loss), "log_mse": (0.3, L.differentiable_log_mse_loss), "etiage": (1.0, L.differentiable_etiage_loss), "fdc_bas": (1.0, L.differentiable_fdc_bas_loss)}
SOURCES = [["physitel-texture"], ["depots-sigeom"], ["depots-sigeom-carto"], ["drainage-fusion-sigeom2"], ["drainage-hierarchique"], ["slc-drainage"], ["slc-hydro", "slc-drainage"]]


def perte_station(qo, qs):
    v = np.isfinite(qo) & np.isfinite(qs)
    if v.sum() < 60:
        return np.nan
    o, s = torch.tensor(qo[v], dtype=torch.float32), torch.tensor(qs[v], dtype=torch.float32)
    return sum(w * float(f(o, s)) for w, f in POIDS.values())


def plafond(nom: str) -> float:
    c = nom.replace("plaf", "")
    return float(c[0] + "." + c[1:]) if c.startswith("0") else (float(c) / 10 if len(c) == 2 and c.endswith("0") else float(c))


def pertes_region(dossier: Path, reg: str) -> pd.DataFrame:
    lignes = []
    for f in sorted(dossier.glob(f"q-plaf*-{reg}.npz")):
        nom = f.name[2:-len(f"-{reg}.npz")]
        if not re.fullmatch(r"plaf\d+", nom):
            continue
        z = np.load(f, allow_pickle=True)
        for j, sid in enumerate(z["station_ids"]):
            lignes.append({"region": reg, "station": str(sid), "plafond": plafond(nom), "perte": perte_station(z["q_obs"][:, j], z["q_sim"][:, j])})
    return pd.DataFrame(lignes)


def prefere(g: pd.DataFrame) -> float:
    g = g.dropna().sort_values("plafond")
    if len(g) < 3:
        return np.nan
    x, y = np.log10(g.plafond.to_numpy()), g.perte.to_numpy()
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    a, b, _ = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(np.clip(-b / (2 * a), x[i - 1], x[i + 1])) if a > 0 else float(x[i])


def perte_predite(p, t, cols, blocs):
    """Perte moyenne si chaque station recevait le plafond prédit hors territoire, interpolée
    linéairement en logarithme du plafond entre les points du balayage, bornée à ses extrémités."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, y = t[cols].to_numpy(), t.log_plafond.to_numpy()
    pred = np.full(len(t), np.nan)
    for b in np.unique(blocs):
        m = HistGradientBoostingRegressor(max_depth=2, max_iter=120, random_state=0, min_samples_leaf=3, l2_regularization=1.0)
        m.fit(X[blocs != b], y[blocs != b])
        pred[blocs == b] = m.predict(X[blocs == b])
    glob = p.groupby("plafond").perte.mean().idxmin()
    a, g = [], []
    for (reg, sid), lp in zip(t[["region", "station"]].itertuples(index=False), pred):
        c = p[(p.region == reg) & (p.station == sid)].dropna().sort_values("plafond")
        x = np.log10(c.plafond.to_numpy())
        a.append(float(np.interp(lp, x, c.perte.to_numpy())))
        g.append(float(c.perte[c.plafond == glob].iloc[0]))
    return float(np.mean(a)), float(np.mean(g))


def main():
    dossier = Path(sys.argv[1])
    regions = [r.lower() for r in sys.argv[2:]]
    p = pd.concat([pertes_region(dossier, r) for r in regions], ignore_index=True)
    tot = p.groupby(["region", "plafond"]).perte.mean().unstack()
    print("perte moyenne des stations par plafond uniforme (mm/j), 2019-2024 :")
    print(tot.round(3).to_string())
    pref = p.groupby(["region", "station"]).apply(prefere, include_groups=False).rename("log_plafond").reset_index().dropna()
    meilleure = p.groupby(["region", "station"]).perte.min().rename("perte_min")
    unif = p.merge(tot.idxmin(axis=1).rename("p_terr"), left_on="region", right_index=True)
    unif = unif[unif.plafond == unif.p_terr].set_index(["region", "station"]).perte.rename("perte_terr")
    glob = p.groupby("plafond").perte.mean().idxmin()
    g1 = p[p.plafond == glob].set_index(["region", "station"]).perte.rename("perte_glob")
    pref = pref.join(meilleure, on=["region", "station"]).join(unif, on=["region", "station"]).join(g1, on=["region", "station"])
    print(f"\nplafond uniforme le meilleur pour l'ensemble : {glob} mm/j")
    print("plafond préféré par station (mm/j), médiane et quartiles, et perte moyenne : plafond global, meilleur par territoire, meilleur par station")
    for reg, g in pref.groupby("region"):
        q = 10 ** g.log_plafond.quantile([0.25, 0.5, 0.75])
        print(f"  {reg:5s} {len(g):3d} st. | {q.iloc[0]:.2f} {q.iloc[1]:.2f} {q.iloc[2]:.2f} | perte {g.perte_glob.mean():.3f} {g.perte_terr.mean():.3f} {g.perte_min.mean():.3f}")
    print(f"  tout  {len(pref):3d} st. | perte {pref.perte_glob.mean():.3f} {pref.perte_terr.mean():.3f} {pref.perte_min.mean():.3f}")
    # Attributs moyennés sur le bassin amont de chaque station.
    import duckdb

    print("\nprédiction du logarithme du plafond préféré, chaque territoire prédit par les autres :")
    for src in SOURCES:
        morceaux = []
        for reg in regions:
            con = duckdb.connect(f"{_paths.DATA_ROOT}/quebec/{reg}.duckdb", read_only=True)
            st = con.execute("SELECT station_id, node_idx FROM stations").df()
            con.close()
            g = pref[pref.region == reg].merge(st.assign(station=st.station_id.astype(str)), on="station")
            if g.empty:
                continue
            att = _ii.attributs_de_station(reg, "finale-n2", g.node_idx.to_numpy(), sources=tuple(src))
            if att is None:
                continue
            morceaux.append(pd.concat([g.reset_index(drop=True), att], axis=1))
        t = pd.concat(morceaux, ignore_index=True)
        cols = [c for c in t.columns if "__" in c and "__couv_" not in c]
        t = t.dropna(subset=cols + ["log_plafond"])
        gros = t.region.value_counts()
        t = t[t.region.isin(gros[gros >= 10].index)]
        blocs = t.region.astype("category").cat.codes.to_numpy()
        e, w = _ii.erreur_hors_bloc(t[cols].to_numpy(), t.log_plafond.to_numpy(), blocs)
        gain = perte_predite(p, t, cols, blocs)
        print(f"  {'+'.join(src):38s} {len(t):3d} st., {t.region.nunique()} terr. | erreur {e:.3f} contre témoin {w:.3f} (log10 mm/j), soit {100 * (1 - e / w):+.0f} % | perte au plafond prédit {gain[0]:.3f} contre {gain[1]:.3f} au plafond global, mêmes stations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
