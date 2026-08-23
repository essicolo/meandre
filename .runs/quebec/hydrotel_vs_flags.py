"""L'avance d'Hydrotel en hiver porte-t-elle sur des jours MESURES ou RECONSTRUITS ?

Question posee le 2026-08-21, apres R19. De decembre a mars, la majeure partie du debit
observe n'est pas une lecture de courbe de tarage mais une valeur estimee ou corrigee
pour effet de refoulement par le CEHQ (janvier 85.4 %, fevrier 87.3 % sur OUTV en tenue
de cote). Hydrotel est cale contre EXACTEMENT ces memes series. Deux lectures possibles,
et elles n'ont pas les memes consequences :

  1. Hydrotel est meilleur partout, hiver compris, sur des jours mesures comme
     reconstruits -> son avance est reelle et il faut la comprendre.
  2. Hydrotel est surtout meilleur sur les jours RECONSTRUITS -> une part de son avance
     est un ajustement a la reconstruction, et le classement hivernal ne mesure pas ce
     qu'on croit qu'il mesure.

Ce script tranche SANS faire tourner meandre : il compare l'ensemble Hydrotel aux
observations, en separant les jours par leur drapeau. Aucun GPU, aucune simulation.

    .venv/Scripts/python.exe .runs/quebec/hydrotel_vs_flags.py [region ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import duckdb
import numpy as np
import pandas as pd
import torch
import xarray as xr

from meandre.data.basin_cache import BasinCache
from meandre.data.hydrotel_calib import appariement_provincial
from meandre.utils import paths as _paths
from meandre.utils.metrics import kge as kge_fn

MEMBERS = ["LN24HA", "MG24HA", "MG24HI", "MG24HK", "MG24HQ", "MG24HS"]
HELD_OUT = ("2022-01-01", "2024-12-31")
REGIONS = sys.argv[1:] or ["outv", "mont", "sagu", "slno", "gasp"]


def region_table(region: str):
    """Retourne (obs, reconstruit, dates, station_ids, node_ids_des_stations)."""
    chemin = _paths.data_path("quebec", f"{region}.duckdb")
    if not os.path.exists(chemin):
        return None
    con = duckdb.connect(chemin, read_only=True)
    if "reconstructed" not in [c[0] for c in con.execute("DESCRIBE observations").fetchall()]:
        con.close()
        return None
    d = con.execute(
        "SELECT station_id, date, discharge, reconstructed FROM observations "
        "WHERE discharge IS NOT NULL AND reconstructed IS NOT NULL "
        "AND date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) ORDER BY date",
        list(HELD_OUT)).df()
    sta = con.execute("SELECT station_id, node_idx FROM stations").df()
    con.close()
    if not len(d):
        return None
    d["station_id"] = d.station_id.astype(str)
    sta["station_id"] = sta.station_id.astype(str)
    obs = d.pivot_table(index="date", columns="station_id", values="discharge", aggfunc="first")
    rec = d.pivot_table(index="date", columns="station_id", values="reconstructed",
                        aggfunc="first").reindex(columns=obs.columns)
    n_map = dict(zip(sta.station_id, sta.node_idx))
    ids = [s for s in obs.columns if s in n_map]
    obs, rec = obs[ids], rec[ids]
    return obs, rec, pd.DatetimeIndex(obs.index), ids, [int(n_map[s]) for s in ids]


def hydrotel_series(region, member, node_ids_globaux, node_idx_stations, dates):
    z = xr.open_zarr(f"{_paths.RQH_ROOT}/06_posttraitement/posttraitement_{member}.zarr")
    cols = appariement_provincial(region, [int(node_ids_globaux[i]) for i in node_idx_stations],
                                  np.asarray(z["troncon_id"].values).astype(str))
    q = z["Dis"].sel(time=slice(str(dates[0])[:10], str(dates[-1])[:10])) \
                .transpose("time", "troncon_idx")
    t = pd.DatetimeIndex(q["time"].values).normalize()
    v = q.values
    z.close()
    out = np.full((len(dates), len(cols)), np.nan)
    pos = pd.Series(np.arange(len(t)), index=t).reindex(dates.normalize()).to_numpy()
    ok = np.isfinite(pos)
    for j, c in enumerate(cols):
        if c is None:
            continue
        out[ok, j] = v[pos[ok].astype(int), c]
    return out


def score(obs, sim, garder, min_jours=60):
    """KGE median par station sur les jours retenus par le masque `garder`.

    `min_jours` descend a 30 pour le contraste intramensuel : sur trois ans, un mois ne
    compte que ~93 jours par station, dont a peine la moitie dans chaque classe. Garder
    60 y eliminait TOUTES les stations et rendait un tableau vide, ce qui se lit comme
    une absence de signal alors que c'est une absence de mesure.
    """
    s = []
    for j in range(obs.shape[1]):
        m = np.isfinite(obs[:, j]) & np.isfinite(sim[:, j]) & garder[:, j]
        if m.sum() >= min_jours:
            s.append(float(kge_fn(torch.tensor(obs[m, j]), torch.tensor(sim[m, j]))))
    return (np.median(s), len(s)) if s else (np.nan, 0)


def main():
    print("Hydrotel contre les observations, jours MESURES vs RECONSTRUITS (drapeaux CEHQ)")
    print(f"Tenue de cote {HELD_OUT[0]} .. {HELD_OUT[1]}. Ecart positif = Hydrotel fait")
    print("MIEUX sur les jours reconstruits que sur les jours mesures.\n")
    print(f"{'region':>6s} {'membre':>8s} {'mesures':>9s} {'reconstr':>9s} {'ecart':>8s} "
          f"{'n_mes':>6s} {'n_rec':>6s}")
    lignes = []
    for region in REGIONS:
        t = region_table(region)
        if t is None:
            print(f"{region:>6s}  (pas de drapeaux ou pas d'observations)")
            continue
        obs_df, rec_df, dates, ids, node_idx = t
        cache = BasinCache(_paths.data_path("quebec", f"{region}.duckdb"))
        node_ids = cache.load(device="cpu")["node_ids"]
        o = obs_df.to_numpy(dtype=float)
        r = rec_df.to_numpy()
        mesure = (r == False)   # noqa: E712
        reconstruit = (r == True)   # noqa: E712
        for mb in MEMBERS:
            try:
                sim = hydrotel_series(region, mb, node_ids, node_idx, dates)
            except Exception as e:
                print(f"{region:>6s} {mb:>8s}  ({type(e).__name__}: {e})")
                continue
            km, nm = score(o, sim, mesure)
            kr, nr = score(o, sim, reconstruit)
            lignes.append((region, mb, km, kr))
            print(f"{region:>6s} {mb:>8s} {km:9.4f} {kr:9.4f} {kr-km:+8.4f} {nm:6d} {nr:6d}")
    if lignes:
        d = pd.DataFrame(lignes, columns=["region", "membre", "mesure", "reconstruit"])
        d["ecart"] = d.reconstruit - d.mesure
        print(f"\n  BRUT : mesures {d.mesure.median():.4f} | "
              f"reconstruits {d.reconstruit.median():.4f} | "
              f"ecart median {d.ecart.median():+.4f} sur {len(d)} couples region x membre")
        print("  MAIS CE CONTRASTE EST CONFONDU : les jours reconstruits SONT les jours")
        print("  d'hiver, et l'hiver est intrinsequement plus dur. On compare deux saisons,")
        print("  pas deux qualites de donnee. Le contraste valide est ci-dessous.")


def contraste_intramensuel():
    """Le SEUL contraste propre : dans le MEME mois, jours drapeautes contre non drapeautes.

    Decembre (47 % de reconstruits sur OUTV), mars (61 %) et avril (9.5 %) contiennent les
    deux classes en quantite. A meteorologie et a regime hydrologique comparables, un ecart
    de score entre les deux classes ne peut plus etre mis sur le compte de la saison.
    """
    print("\n\n  CONTRASTE INTRAMENSUEL (a saison EGALE) : KGE sur les jours mesures contre")
    print("  les jours reconstruits, DANS LE MEME MOIS. Ecart positif = Hydrotel colle")
    print("  MIEUX a la reconstruction qu'a la mesure, ce qui trahirait un ajustement.\n")
    print(f"{'region':>6s} {'mois':>5s} {'mesures':>9s} {'reconstr':>9s} {'ecart':>8s} "
          f"{'n_paires':>9s}")
    tout = []
    for region in REGIONS:
        t = region_table(region)
        if t is None:
            continue
        obs_df, rec_df, dates, ids, node_idx = t
        cache = BasinCache(_paths.data_path("quebec", f"{region}.duckdb"))
        node_ids = cache.load(device="cpu")["node_ids"]
        o, r = obs_df.to_numpy(dtype=float), rec_df.to_numpy()
        mois = dates.month.to_numpy()
        sims = []
        for mb in MEMBERS:
            try:
                sims.append(hydrotel_series(region, mb, node_ids, node_idx, dates))
            except Exception:
                pass
        if not sims:
            continue
        sim = np.nanmedian(np.stack(sims), axis=0)   # mediane de l'ensemble
        for m in range(1, 13):
            dans = (mois == m)[:, None] & np.ones_like(o, dtype=bool)
            a = dans & (r == False)   # noqa: E712
            b = dans & (r == True)    # noqa: E712
            # exiger les DEUX classes en quantite : sinon on retombe sur le contraste
            # saisonnier qu'on cherche justement a eliminer
            if a.sum() < 150 or b.sum() < 150:
                continue
            km, nm = score(o, sim, a, min_jours=30)
            kr, nr = score(o, sim, b, min_jours=30)
            if nm < 5 or nr < 5:
                continue
            tout.append((region, m, km, kr))
            print(f"{region:>6s} {m:5d} {km:9.4f} {kr:9.4f} {kr-km:+8.4f} {min(nm, nr):9d}")
    if tout:
        d = pd.DataFrame(tout, columns=["region", "mois", "mesure", "reconstruit"])
        d["ecart"] = d.reconstruit - d.mesure
        print(f"\n  ecart median a saison egale : {d.ecart.median():+.4f} "
              f"sur {len(d)} couples region x mois")
        print("  Proche de zero = le drapeau ne change pas la qualite de l'accord, donc")
        print("  l'avance hivernale d'Hydrotel n'est PAS un ajustement a la reconstruction.")


if __name__ == "__main__":
    main()
    contraste_intramensuel()
