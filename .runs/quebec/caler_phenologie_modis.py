"""Cale la phénologie du modèle sur l'indice foliaire MODIS, sans simulation hydrologique.

MODIS ne peut pas piloter le modèle en prédiction : seule la météo sera disponible. Il sert
en revanche à FIXER les paramètres d'une phénologie pilotée par la météo. On ajuste la
forme du modulateur (`meandre/temporal/phenology_modulator.py`) sur l'indice foliaire
observé, rapporté à son maximum, avec deux pilotes :

    degrés-jours : montée et chute pilotées par le cumul de degrés-jours au-dessus de 10 °C
    photopériode : montée aux degrés-jours, chute par la longueur du jour après le solstice

Température moyenne du bassin tirée du forçage CaSR, longueur du jour de la latitude. On
compare les deux pilotes par l'erreur quadratique sur toutes les années, puis en validation
croisée par année : chaque année prédite avec les paramètres ajustés sur les autres.

    python .runs/quebec/caler_phenologie_modis.py mont 030905 2011 2013
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import least_squares

from meandre.utils import paths as _p


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def duree_du_jour(lat, doy):
    decl = math.radians(23.44) * np.sin(2.0 * math.pi * (284 + doy) / 365.0)
    x = np.clip(-math.tan(math.radians(lat)) * np.tan(decl), -1.0, 1.0)
    return 24.0 / math.pi * np.arccos(x)


def forme(params, pilote, gdd, doy, lat):
    if pilote == "degres-jours":
        emerg, mid, s1, s2 = params
        return sig((gdd - emerg) / s1) * sig(-(gdd - mid) / s2)
    emerg, crit, s1, s2 = params
    sen = np.where(doy > 172, sig((duree_du_jour(lat, doy) - crit) / s2), 1.0)
    return sig((gdd - emerg) / s1) * sen


def main(reg, station, a0, a1):
    from banc_sousbassin import extraire
    s = extraire(reg, station)
    ids = np.asarray(s["node_ids"]) if "node_ids" in s else None
    lat = float(s["node_coords"][:, 1].mean())
    ds = xr.open_dataset(f"{_p.DATA_ROOT}/quebec/forcing-{reg}-budyko.nc")
    idx = np.asarray(s["idx"])
    f7 = ds["forcing"].isel(node=idx)
    tm = (0.5 * (f7.sel(var="Tmin") + f7.sel(var="Tmax"))).mean("node").to_series()
    tm = tm[(tm.index.year >= a0) & (tm.index.year <= a1)]
    gdd = tm.clip(lower=10.0).sub(10.0).groupby(tm.index.year).cumsum()
    lai = pd.read_csv(f"{_p.DATA_ROOT}/derives/auxiliaires/lai-modis-{reg}-{station}-{a0}-{a1}.csv", parse_dates=["date"])
    # Un composite couvre 8 jours : on le compare au milieu de sa fenêtre.
    lai["jour"] = lai.date + pd.Timedelta(days=4)
    lai = lai[lai.jour.isin(gdd.index)]
    y = (lai.lai_median / lai.lai_median.max()).to_numpy()
    g = gdd.loc[lai.jour].to_numpy()
    d = lai.jour.dt.dayofyear.to_numpy()
    an = lai.jour.dt.year.to_numpy()
    init = {"degres-jours": [80.0, 1200.0, 50.0, 100.0], "photopériode": [80.0, 12.0, 50.0, 0.5]}
    bornes = {"degres-jours": ([0, 300, 5, 10], [600, 2500, 300, 500]), "photopériode": ([0, 8, 5, 0.05], [600, 16, 300, 3])}
    print(f"sous-bassin {station}, latitude moyenne {lat:.2f}, {len(y)} composites, {a0}-{a1}")
    for pilote in ("degres-jours", "photopériode"):
        f = lambda p, m: forme(p, pilote, g[m], d[m], lat) - y[m]
        tout = np.ones(len(y), dtype=bool)
        r = least_squares(f, init[pilote], bounds=bornes[pilote], args=(tout,))
        rmse = float(np.sqrt(np.mean(r.fun ** 2)))
        cv = []
        for a in sorted(set(an)):
            ap, te = an != a, an == a
            rr = least_squares(f, init[pilote], bounds=bornes[pilote], args=(ap,))
            cv.append(float(np.sqrt(np.mean(f(rr.x, te) ** 2))))
        # Écart de forme en automne, septembre et octobre, sur l'ajustement global.
        aut = (d >= 244) & (d <= 304)
        ec_aut = float(np.mean((forme(r.x, pilote, g, d, lat) - y)[aut]))
        print(f"  {pilote:13s} parametres {np.round(r.x, 2).tolist()} | erreur quadratique {rmse:.3f} | par annee tenue de cote {np.round(cv, 3).tolist()} | biais septembre-octobre {ec_aut:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
