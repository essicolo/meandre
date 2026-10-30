"""Comparaison PHYSITEL / WorldCover par UHRH et table du bâti par tronçon, tous les territoires.

Sorties : <dossier>/<territoire>-uhrh-compare.parquet, et la table
{DERIVES}/auxiliaires/impermeable-worldcover-troncons.parquet (territory, node_id, impervious)
lue par le réglage `land_cover.impervious_table` (R455).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from station_errors import reach_cover

PLATFORMS = Path("C:/Users/parse01/documents-locaux/GitHub/plateformes-hydrotel/LN24HA")
WC = Path("D:/meandre-data/externes/worldcover")
TABLE = Path("D:/meandre-data/derives/auxiliaires/impermeable-worldcover-troncons.parquet")


def compare(project, territory):
    lines = [l.split() for l in (project / "physitel" / "occupation_sol.cla").read_text(encoding="latin-1").splitlines() if l.strip()]
    names = [x.strip('"') for x in lines[2][1:]]
    A = np.array([[float(x) for x in r[1:]] for r in lines[3:]])
    ph = pd.DataFrame(A / np.clip(A.sum(1, keepdims=True), 1e-9, None), columns=names)
    ph["uhrh"] = [int(r[0]) for r in lines[3:]]
    ph["pix"] = A.sum(1)
    m = ph.merge(pd.read_parquet(WC / f"{territory}-uhrh.parquet"), on="uhrh")
    m.to_parquet(WC / f"{territory}-uhrh-compare.parquet")
    return m


blocks, summary = [], []
for project in sorted(PLATFORMS.glob("*_LN24HA_2020")):
    t = project.name.split("_")[0].lower()
    m = compare(project, t)
    c = reach_cover(project, WC / f"{t}-uhrh-compare.parquet")
    blocks.append(pd.DataFrame({"territory": t, "node_id": c.node_id.astype(int), "impervious": c.wc_built}))
    w = m["wc_pixels"]
    summary.append({"territory": t, "physitel_impermeable": np.average(m["impermeable"], weights=w), "worldcover_built": np.average(m["wc_built"], weights=w),
                    "physitel_water": np.average(m["eau"], weights=w), "worldcover_water": np.average(m["wc_water"], weights=w),
                    "physitel_wetland": np.average(m["tourbiere"] + m["milieu_humide"], weights=w), "worldcover_wetland": np.average(m["wc_wetland"], weights=w)})
table = pd.concat(blocks)
table.to_parquet(TABLE)
print(pd.DataFrame(summary).round(4).to_string(index=False))
print(len(table), "tronçons")
