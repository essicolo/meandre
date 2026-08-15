"""DUEL contre l'ENSEMBLE des 6 calages Hydrotel (règle du projet : ne jamais se
comparer à un seul membre).

Le 2026-08-11 on a découvert que le « 0.82 d'Hydrotel » répété pendant des jours était
une chaîne de caractères jamais mesurée. Mesuré sur le membre LN24HA seul, Hydrotel fait
0.7531 sur OUTV et méandre ancré 0.7389. Ce script étend la comparaison aux 6 membres :
médiane de l'ensemble, MEILLEUR membre, et fraction de stations où méandre passe devant.

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/duel_ensemble.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import tomllib, numpy as np, pandas as pd, torch, xarray as xr
from meandre.data.hydrotel_calib import appariement_provincial
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
BASE = "C:/Users/parse01/documents-locaux/rqh-local/rqh_2026-04/data/06_posttraitement"
MEMBRES = ["LN24HA", "MG24HA", "MG24HI", "MG24HK", "MG24HQ", "MG24HS"]
T0, T1 = "2022-01-01", "2024-12-31"

cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device="cpu")
td = r["train_data"]; ids = r["node_ids"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
m = (tt >= T0) & (tt <= T1)
sid = td.station_idx.numpy()
qo = td.q_obs.numpy()[m]
tro = [int(ids[i]) for i in sid]

def kge(o, s):
    if o.std() < 1e-9 or s.std() < 1e-9 or o.mean() <= 0 or s.mean() <= 0:
        return np.nan
    rr = np.corrcoef(o, s)[0, 1]
    b = s.mean() / o.mean()
    g = (s.std() / s.mean()) / (o.std() / o.mean())
    return 1.0 - np.sqrt((rr - 1) ** 2 + (b - 1) ** 2 + (g - 1) ** 2)

res = {}
for mem in MEMBRES:
    p = f"{BASE}/posttraitement_{mem}.zarr"
    if not os.path.exists(p):
        print(f"  {mem} : absent"); continue
    z = xr.open_zarr(p)
    var = "Dis" if "Dis" in z else list(z.data_vars)[0]
    dimt = [d for d in z[var].dims if "time" in d.lower() or "date" in d.lower()][0]
    dimn = [d for d in z[var].dims if d != dimt][0]
    # Les identifiants ne sont PAS la dimension (`troncon_idx`, un simple rang) mais une
    # COORDONNÉE dédiée `troncon_id`, de la forme "REG#####". Deux appariements ratés
    # avant de le vérifier : entiers contre chaînes (KGE -0.25, absurde), puis chaînes
    # contre le rang (aucun appariement).
    idvar = "troncon_id" if "troncon_id" in z.coords or "troncon_id" in z else dimn
    tz = pd.DatetimeIndex(pd.to_datetime(z[dimt].values))
    # IDENTIFIANTS : le stockage PROVINCIAL nomme les tronçons "REG#####" (chaîne),
    # pas par l'entier local de la région. Comparer les entiers apparie des tronçons
    # sans rapport et donne des KGE absurdes (-0.25 mesuré avant correction, alors que
    # le MÊME membre lu dans le fichier régional donne 0.7531).
    # conversion CENTRALISÉE (voir hydrotel_calib) : lève si rien ne s'apparie, plutôt
    # que de rendre des scores absurdes en silence.
    cols = appariement_provincial(REG, tro, np.asarray(z[idvar].values).astype(str))
    # L'ordre des dimensions varie d'un stockage à l'autre (ici tronçon en PREMIER) :
    # on laisse xarray sélectionner et transposer plutôt que d'indexer à la main.
    sel = z[var].sel({dimt: slice(T0, T1)}).transpose(dimt, dimn)
    Q = sel.values[:, [c if c is not None else 0 for c in cols]]
    z.close()
    n = min(Q.shape[0], qo.shape[0])
    ks = []
    for k in range(len(tro)):
        if cols[k] is None:
            ks.append(np.nan); continue
        o, s = qo[:n, k], Q[:n, k]
        v = np.isfinite(o) & np.isfinite(s)
        ks.append(kge(o[v], s[v]) if v.sum() >= 60 else np.nan)
    res[mem] = np.array(ks, dtype=float)
    print(f"  {mem} : médian {np.nanmedian(res[mem]):.4f} "
          f"({int(np.isfinite(res[mem]).sum())}/{len(tro)} stations)", flush=True)

if not res:
    raise SystemExit("aucun membre exploitable")
M = np.vstack([res[k] for k in res])            # (n_membres, n_stations)
med_par_station = np.nanmedian(M, axis=0)
best_par_station = np.nanmax(M, axis=0)
print(f"\n=== ENSEMBLE {REG.upper()} ({len(res)} membres, {T0}..{T1}) ===")
print(f"  médiane des médianes de membre : {np.nanmedian([np.nanmedian(v) for v in res.values()]):.4f}")
print(f"  ENSEMBLE médian par station    : {np.nanmedian(med_par_station):.4f}")
print(f"  MEILLEUR membre par station    : {np.nanmedian(best_par_station):.4f}")
# PAS DE REPÈRE EN DUR (leçon du 0.82 fantôme, 11 août) : les valeurs de méandre
# dépendent de la région et un chiffre recopié devient une donnée fausse. On lit le
# journal des runs si un résultat existe pour CETTE région, sinon on ne dit rien.
import glob as _gl
_ms = []
for _f in _gl.glob(f"D:/meandre-data/quebec/log-{REG}-socle*.txt"):
    for _l in open(_f, encoding="utf-8", errors="ignore"):
        if "HELD-OUT" in _l and "médian" in _l:
            _ms.append((os.path.basename(_f), _l.split("médian")[1].split("|")[0].strip()))
if _ms:
    print("\n  repères méandre MESURÉS sur cette région :")
    for _n, _v in _ms:
        print(f"    {_n:34s} {_v}")
else:
    print("\n  (aucun run méandre trouvé pour cette région : pas de repère affiché)")
np.savez_compressed(f"D:/meandre-data/quebec/duel_ensemble_{REG}.npz",
                    membres=np.array(list(res.keys())), kge=M,
                    med=med_par_station, best=best_par_station,
                    troncons=np.array(tro))
