"""Série de l'indice foliaire MODIS (MOD15A2H, 500 m, 8 jours) sur l'emprise d'un sous-bassin.

Juge indépendant de la phénologie du modèle : l'indice foliaire observé montre la levée,
le plateau et la chute d'automne (sénescence des feuillus, récolte des cultures), année par
année, sans hypothèse sur le pilote. Lecture fenêtrée à distance sur le Planetary Computer,
aucune tuile entière n'est téléchargée. Rend, par date de composite, la médiane et les
quartiles de l'indice foliaire des pixels valides de l'emprise.

    python .runs/quebec/serie_lai_modis.py mont 030905 2011 2013
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from meandre.data.open_data import _windowed_read
from meandre.utils import paths as _p


def emprise(reg, station):
    from banc_sousbassin import extraire
    s = extraire(reg, station)
    c = s["node_coords"].cpu().numpy()
    return (float(c[:, 0].min()) - 0.05, float(c[:, 1].min()) - 0.05, float(c[:, 0].max()) + 0.05, float(c[:, 1].max()) + 0.05)


def main(reg, station, a0, a1):
    import planetary_computer
    from pystac_client import Client
    bbox = emprise(reg, station)
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["modis-15A2H-061"], bbox=bbox, datetime=f"{a0}-01-01/{a1}-12-31").items())
    print(f"{len(items)} composites sur l'emprise {tuple(round(x, 2) for x in bbox)}", flush=True)
    lignes, cartes = [], {}
    _date = lambda x: x.datetime or pd.Timestamp(x.properties.get("start_datetime"))
    for it in sorted(items, key=lambda x: pd.Timestamp(_date(x))):
        if "Lai_500m" not in it.assets:
            continue
        r = _windowed_read(it.assets["Lai_500m"].href, bbox)
        if r is None:
            continue
        a = r[0].astype(np.float32)
        a[a > 100] = np.nan
        a *= 0.1
        d = pd.Timestamp(_date(it)).tz_convert(None).normalize() if pd.Timestamp(_date(it)).tzinfo else pd.Timestamp(_date(it)).normalize()
        cartes.setdefault(d, []).append(a)
    for d, liste in sorted(cartes.items()):
        # Deux tuiles peuvent couvrir l'emprise : on concatène leurs pixels valides.
        v = np.concatenate([x[np.isfinite(x)].ravel() for x in liste])
        if v.size == 0:
            continue
        lignes.append({"date": d, "lai_median": float(np.median(v)), "lai_p25": float(np.percentile(v, 25)), "lai_p75": float(np.percentile(v, 75)), "n": int(v.size)})
    t = pd.DataFrame(lignes)
    f = f"{_p.DATA_ROOT}/derives/auxiliaires/lai-modis-{reg}-{station}-{a0}-{a1}.csv"
    t.to_csv(f, index=False)
    t["mois"] = t.date.dt.month
    print(t.groupby("mois")[["lai_median", "lai_p25", "lai_p75"]].mean().round(2).to_string())
    print(f"ecrit : {f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
