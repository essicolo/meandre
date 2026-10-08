"""Plancher de bruit sur la cible de fonte CanSWE.

R54 a mesure un plafond de R2 = 0.234 sur l'INTENSITE de la fonte journaliere. Avant
d'attribuer ce residu a une physique manquante, il faut borner ce que la cible elle-meme
peut porter. Une variation de masse en un point ponctuel contient l'erreur de mesure,
la redistribution par le vent et la representativite du site.

Methode : deux sites voisins, sous le meme forcage meteo a l'echelle de la maille, sont
deux realisations du meme signal. Leur accord BORNE ce qu'un modele pilote par ce forcage
peut atteindre. Les carottages manuels etant bihebdomadaires, on normalise l'ablation par
la duree de l'intervalle et on apparie les intervalles qui se recouvrent.
"""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from meandre.data.basin_cache import BasinCache
from meandre.utils import paths as p

PLATEFORMES = sys.argv[1:] or ["sagu", "outv", "gasp", "abit", "mont", "slso"]
DIST_MAX = 40.0
DUREE_MAX = 10

lignes = []
for reg in PLATEFORMES:
    try:
        bc = BasinCache(p.data_path("quebec", f"{reg}.duckdb"))
        mes, sites = bc.load_canswe("2000-01-01", "2024-12-31")
    except Exception as e:
        print(f"[{reg}] ignore : {e}")
        continue
    m = mes[mes.swe_mm.notna()].copy()
    m["sid"] = reg + ":" + m.swe_station_id.astype(str)
    m["date"] = pd.to_datetime(m.date)
    s = sites.copy()
    s["sid"] = reg + ":" + s.swe_station_id.astype(str)
    lignes.append((m, s[["sid", "lat", "lon"]]))

m = pd.concat([a for a, _ in lignes], ignore_index=True)
coord = pd.concat([b for _, b in lignes], ignore_index=True).drop_duplicates("sid").set_index("sid")
m = m.sort_values(["sid", "date"])
m["dswe"] = m.groupby("sid").swe_mm.diff()
m["dj"] = m.groupby("sid").date.diff().dt.days
m["fin"] = m.date
m["debut"] = m.date - pd.to_timedelta(m.dj, unit="D")
iv = m[(m.dj >= 1) & (m.dj <= DUREE_MAX) & m.dswe.notna()].copy()
# ablation moyenne sur l'intervalle, en mm/j
iv["abl"] = np.clip(-iv.dswe, 0.0, None) / iv.dj
print(f"{len(iv):,} intervalles sur {iv.sid.nunique()} sites, {len(PLATEFORMES)} plateformes")

ids = [i for i in iv.sid.unique() if i in coord.index]
paires = []
for a in range(len(ids)):
    for b in range(a + 1, len(ids)):
        la, lo = coord.loc[ids[a]]
        lb, lob = coord.loc[ids[b]]
        d = 111.0 * np.hypot(lo - lob, (la - lb) * np.cos(np.radians(la)))
        if d <= DIST_MAX:
            paires.append((ids[a], ids[b], d))
print(f"{len(paires)} paires de sites a moins de {DIST_MAX:.0f} km")

par_sid = {k: v.sort_values("debut") for k, v in iv.groupby("sid")}
rs, ns, ds = [], [], []
for a, b, d in paires:
    A, B = par_sid[a], par_sid[b]
    # recouvrement : intervalles dont le milieu du plus court tombe dans l'autre
    A2 = A.assign(mid=A.debut + (A.fin - A.debut) / 2)
    xs, ys = [], []
    for _, r in A2.iterrows():
        c = B[(B.debut <= r.mid) & (B.fin >= r.mid)]
        if len(c):
            xs.append(r.abl)
            ys.append(float(c.abl.iloc[0]))
    if len(xs) < 40:
        continue
    xs, ys = np.array(xs), np.array(ys)
    if xs.std() < 1e-6 or ys.std() < 1e-6:
        continue
    rs.append(float(np.corrcoef(xs, ys)[0, 1]))
    ns.append(len(xs))
    ds.append(d)

rs = np.array(rs)
if len(rs) == 0:
    print("aucune paire exploitable")
    raise SystemExit(0)
print(f"{len(rs)} paires avec au moins 40 intervalles apparies (median {int(np.median(ns))})")
print(f"correlation entre sites voisins : mediane {np.median(rs):.3f} | "
      f"q25 {np.quantile(rs, .25):.3f} | q75 {np.quantile(rs, .75):.3f}")
print(f"R2 implicite (un site prevoit son voisin) : mediane {np.median(rs) ** 2:.3f} | "
      f"q75 {np.quantile(rs, .75) ** 2:.3f}")
b = pd.cut(np.array(ds), [0, 10, 20, 30, 40])
for k, g in pd.DataFrame({"r": rs, "b": b}).groupby("b", observed=True):
    print(f"  {str(k):>10} km : {len(g):3d} paires | r median {g.r.median():.3f}")
