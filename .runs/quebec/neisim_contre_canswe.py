"""NEISIM est-il assez proche des relevés au sol pour servir de cible.

NEISIM est un produit du gouvernement du Québec : équivalent en eau de la neige et apport
vertical au sol, sur grille, au pas de trois heures ou journalier, de 1980 à 2025. Sa
couverture est complète là où le réseau CanSWE est très inégal, de 76 sites en Outaouais à
aucun au Saint-Laurent sud-ouest. Il pourrait donc étendre à tout le domaine une contrainte
qui n'existe aujourd'hui que par endroits.

Mais NEISIM est un MODÈLE, pas une mesure. L'employer comme cible imposerait ses biais au
nôtre. La question préalable est donc simple : sur les sites où les deux existent, NEISIM
s'accorde-t-il aux relevés au sol ? S'il s'en écarte autant que notre propre modèle, il
n'apporte rien qu'une autre opinion.

La comparaison se fait au point de grille le plus proche de chaque site, sur les jours où un
relevé existe, en gardant la saison de neige seulement.

    .venv/Scripts/python.exe .runs/quebec/neisim_contre_canswe.py
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NEISIM = os.environ.get("MEANDRE_NEISIM", "D:/meandre-data/neisim")
CANSWE = os.environ.get("MEANDRE_CANSWE",
                        "D:/meandre-data/canswe/CanSWE-CanEEN_1928-2025_v8.nc")
# Domaine du Quebec meridional couvert par les modeles regionaux.
BBOX = (-80.0, 45.0, -64.0, 51.0)
MOIS_NEIGE = (11, 12, 1, 2, 3, 4, 5)


def sites_canswe():
    """Sites du réseau au sol dans le domaine, avec leurs relevés d'équivalent en eau."""
    c = xr.open_dataset(CANSWE, decode_timedelta=False)
    lat, lon = c.lat.values, c.lon.values
    dedans = ((lon >= BBOX[0]) & (lon <= BBOX[2]) & (lat >= BBOX[1]) & (lat <= BBOX[3]))
    idx = np.flatnonzero(dedans)
    snw = c.snw.isel(station_id=idx).values
    n_obs = np.isfinite(snw).sum(axis=1)
    garde = n_obs >= 50
    return (lon[idx][garde], lat[idx][garde], snw[garde],
            pd.to_datetime(c.time.values), int(garde.sum()))


def main():
    # Fichier JOURNALIER, et lecture par TRANCHES DE TEMPS. Le decoupage interne est une
    # carte complete par pas de temps : lire la serie d'un seul point force a decompresser
    # tout le fichier. L'acces efficace est donc par tranche, pas par point.
    f = f"{NEISIM}/NEISIM_QCMERI_EENEIG_GOENS_QCHRES_24H_MEDIAN.nc"
    if not os.path.exists(f):
        print(f"fichier absent : {f}")
        return 1
    lon_s, lat_s, snw, temps_obs, n = sites_canswe()
    print(f"{n} sites du reseau au sol dans le domaine, avec au moins 50 releves\n")

    d = xr.open_dataset(f, decode_timedelta=False)
    # La grille de NEISIM porte ses coordonnees dans les variables x et y, non dans les
    # dimensions : on les remet en index pour pouvoir selectionner par position.
    d = d.assign_coords(lon=d.x.values, lat=d.y.values)
    t_n = pd.to_datetime(d.time.values)

    # Points de grille les plus proches, reperes une fois.
    ilon = np.abs(d.lon.values[None, :] - lon_s[:, None]).argmin(axis=1)
    ilat = np.abs(d.lat.values[None, :] - lat_s[:, None]).argmin(axis=1)
    uniques, inverse = np.unique(np.stack([ilon, ilat], axis=1), axis=0, return_inverse=True)
    print(f"{len(uniques)} points de grille distincts pour {n} sites", flush=True)

    import netCDF4 as nc

    racine = nc.Dataset(f)
    var = racine.variables["een"]
    n_t = var.shape[2]
    bloc = 512
    sorties = np.empty((len(uniques), n_t), dtype="float32")
    for a0 in range(0, n_t, bloc):
        a1 = min(a0 + bloc, n_t)
        tranche = var[:, :, a0:a1]
        sorties[:, a0:a1] = np.asarray(tranche)[uniques[:, 0], uniques[:, 1], :]
        if a0 % (bloc * 8) == 0:
            print(f"  {100 * a1 / n_t:3.0f} %", flush=True)
    racine.close()
    # NEISIM horodate ses journees a 05 h UTC et le reseau au sol a minuit : sans
    # normalisation a la DATE, l'intersection des deux index est vide.
    jour = pd.DataFrame(sorties.T, index=t_n.normalize())
    print("lecture faite, comparaison", flush=True)

    lignes = []
    for k in range(n):
        s = jour.iloc[:, int(inverse[k])]
        o = pd.Series(snw[k], index=temps_obs.normalize()).dropna()
        o = o[o.index.month.isin(MOIS_NEIGE)]
        commun = s.index.intersection(o.index)
        if len(commun) < 30:
            continue
        a, b = o.loc[commun].to_numpy(), s.loc[commun].to_numpy()
        fini = np.isfinite(a) & np.isfinite(b)
        if fini.sum() < 30:
            continue
        a, b = a[fini], b[fini]
        lignes.append({"site": k, "n": int(fini.sum()),
                       "observe": float(np.mean(a)), "neisim": float(np.mean(b)),
                       "rapport": float(np.mean(b) / max(np.mean(a), 1e-9)),
                       "r": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan,
                       "erreur": float(np.mean(np.abs(b - a)))})
    if not lignes:
        print("aucun site comparable")
        return 1
    t = pd.DataFrame(lignes)
    print(f"{len(t)} sites comparables, {int(t.n.sum())} couples de valeurs\n")
    print(f"  equivalent en eau moyen : observe {t.observe.median():.0f} mm, "
          f"NEISIM {t.neisim.median():.0f} mm")
    print(f"  rapport NEISIM sur observe : mediane {t.rapport.median():.2f}, "
          f"quartiles {t.rapport.quantile(.25):.2f} a {t.rapport.quantile(.75):.2f}")
    print(f"  correlation par site        : mediane {t.r.median():.2f}, "
          f"part au-dessus de 0,8 : {100 * (t.r > 0.8).mean():.0f} %")
    print(f"  erreur absolue moyenne      : mediane {t.erreur.median():.0f} mm, "
          f"soit {100 * t.erreur.median() / t.observe.median():.0f} % de la moyenne observee")
    print("\nA comparer au modele : son maximum hivernal simule vaut 121 mm sur l'Outaouais")
    print("contre 238 mm mesures, soit la moitie (note du chantier CanSWE).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
