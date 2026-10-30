# Pentes du coefficient de ruissellement des episodes sur les series simulees des passes INF_PDM
# (2026-10-30, suite de R447). Meme protocole que meandre-analyse/analyses/event-runoff.py : le catalogue
# d'episodes observes (pluie CaSR, veille) est fixe ; seul le volume de crue vient de la passe simulee.
# Les pentes log-log sous effets fixes de station sont invariantes a un facteur d'echelle du debit simule :
# la lecture compare les variantes a la cible (1,2 sur la pluie, 0,6 sur la veille) et au temoin gele.
#
# Usage : python pentes-pdm.py [dossier_des_npz] [territoire] [essai_1 essai_2 ...]
#   defaut : D:/meandre-data/analysis/pdm-2026-10-30, mont, essais pdm pdm-b2 pdm-f3 pdm-b2f3 gele
import sys
from pathlib import Path

import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
CACHE = Path("C:/Users/parse01/documents-locaux/GitHub/meandre-analyse/results/_cache/event-runoff-events.parquet")
DATA_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "D:/meandre-data/analysis/pdm-2026-10-30")
TERRITORY = sys.argv[2] if len(sys.argv) > 2 else "mont"
TRIALS = sys.argv[3] if len(sys.argv) > 3 else None
if TRIALS:
    TRIALS = TRIALS.split()
else:
    TRIALS = ["pdm", "pdm-b2", "pdm-f3", "pdm-b2f3", "gele"]
CLASSES = ((10, 20), (20, 35), (35, 55), (55, 1e9))

events = pl.read_parquet(CACHE).filter(pl.col("territory") == TERRITORY)
first = None


def episodes_of(trial):
    """Catalogue des episodes du passage, volume de crue par la passe simulee (coef_sim) et observe (coef_obs)."""
    z = np.load(DATA_DIR / f"q-{trial}-{TERRITORY}.npz", allow_pickle=True)
    ids = [str(s) for s in z["station_ids"]]
    q = z["q_sim"]
    base = z["dates"].astype(str)
    rows = []
    for (st,), g in events.group_by("station", maintain_order=True):
        code = st.split(":")[-1]
        if code not in ids:
            continue
        j = ids.index(code)
        indices = np.searchsorted(base, g["date"].to_numpy().astype(str))
        keep, vs, scales = [], [], []
        for k, i in enumerate(indices):
            i = int(i)
            if i < 1 or i + 7 > len(base) or base[i] != g["date"][k]:
                continue
            if not np.isfinite(q[i - 1:i + 7, j]).all() or q[i - 1, j] <= 0:
                continue
            # echelle : le npz porte le debit sur l'aire du troncon du pilote, le catalogue sur
            # l'aire hydrologique de la station ; le rapport median des deux sur les episodes corrige
            scales.append(q[i - 1, j] / g["q_prev"][k])
            vs.append(np.clip(q[i:i + 7, j] - q[i - 1, j], 0, None).sum())
            keep.append(k)
        g = g[keep]
        if g.height == 0:
            continue
        rows.append(g.with_columns(volume_sim=pl.Series(vs) / float(np.median(scales))))
    return pl.concat(rows).filter(pl.col("volume_sim") > 0).with_columns(
        coef_sim=pl.col("volume_sim") / pl.col("p_week"), coef_obs=pl.col("volume") / pl.col("p_week"))


def slopes(frame, coef_column):
    """Pentes de log(coefficient) sur log(pluie de la semaine) et log(débit de la veille),
    chaque variable prise en écart à sa moyenne de station, moindres carrés."""
    f = frame.with_columns(lc=pl.col(coef_column).log(), lp=pl.col("p_week").log(), lq=pl.col("q_prev").log())
    f = f.with_columns([(pl.col(c) - pl.col(c).mean().over("station")).alias(c) for c in ("lc", "lp", "lq")])
    x = np.column_stack([f["lp"].to_numpy(), f["lq"].to_numpy()])
    y = f["lc"].to_numpy()
    beta, res, _, _ = np.linalg.lstsq(x, y, rcond=None)
    n = len(y)
    s2 = float(res[0]) / max(n - 2, 1) if len(res) else np.nan
    se = np.sqrt(np.diag(s2 * np.linalg.inv(x.T @ x)))
    return beta[0], se[0], beta[1], se[1], n, f["station"].n_unique()


rows = []
classes_of = None
for trial in TRIALS:
    try:
        frame = episodes_of(trial)
    except FileNotFoundError:
        print(f"{trial}: npz absent, passe probablement echouee")
        continue
    if first is None:
        first = frame
        sr, ser, sw, sew, n_obs, n_st = slopes(frame, "coef_obs")
        rows.append({"essai": "obs", "pente_pluie": round(sr, 2), "se_pluie": round(ser, 2),
                     "pente_veille": round(sw, 2), "se_veille": round(sew, 2),
                     "episodes": n_obs, "stations": n_st})
    sr, ser, sw, sew, n_events, n_st = slopes(frame, "coef_sim")
    classes = []
    for lo, hi in CLASSES:
        sel = frame.filter((pl.col("p_week") >= lo) & (pl.col("p_week") < hi))
        if sel.height == 0:
            continue
        ratio = float(sel.with_columns(rat=pl.col("coef_sim") / pl.col("coef_obs"))["rat"].median())
        c = float(sel["coef_sim"].median())
        classes.append(f"{lo}-{int(hi) if hi < 1e9 else '+'}: sim {c:.3f} ratio {ratio:.2f}")
    rows.append({"essai": trial, "pente_pluie": round(sr, 2), "se_pluie": round(ser, 2),
                 "pente_veille": round(sw, 2), "se_veille": round(sew, 2),
                 "episodes": n_events, "stations": n_st, "classes": " ; ".join(classes)})

table = pl.DataFrame(rows)
cols = [c for c in ("essai", "pente_pluie", "se_pluie", "pente_veille", "se_veille", "episodes", "stations") if c in table.columns]
print(table.select(cols))
for row in rows:
    if row.get("classes"):
        print(f"  {row['essai']:8s} {row['classes']}")
out = DATA_DIR / f"pentes-{TERRITORY}.csv"
table.write_csv(out)
print("ecrit", out)