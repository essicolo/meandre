"""QUEL EST LE VRAI TAUX DE VIDANGE SOUTERRAIN ? (chantier hiver, 2026-08-19)

Le champion tourne avec k_gw = 0.0645 /j, soit un temps de residence de ~15 jours.
Le diagnostic de fevrier montre que la riviere coule a 13-15 m3/s pendant que la
neige s'accumule : l'eau vient d'une reserve remplie a l'automne, ce qu'une nappe
qui se vide en 15 jours ne peut pas faire.

Ici on MESURE le taux sur les jauges, sans modele : en hiver profond (dec-mars),
sans pluie ni fonte, le debit qui decroit est du drainage pur. La pente de ln(Q)
sur ces segments EST le taux de vidange.

Contrainte de rigueur : on exige un segment stricement decroissant d'au moins
NMIN jours ET une temperature max sous 0 degre sur tout le segment (sinon la fonte
alimente et la recession n'est pas pure).

  PYTHONIOENCODING=utf-8 JOINT_FX_SUFFIX=-hyb python .runs/quebec/recession_hiver.py outv
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd()); sys.path.insert(0, ".runs/quebec")
import numpy as np, pandas as pd, torch
from joint_data import load_region

REG = (sys.argv[1] if len(sys.argv) > 1 else "outv").lower()
NMIN = int(os.environ.get("REC_NMIN", "5"))
import tomllib
cfg = tomllib.load(open(".runs/quebec/config/gasp-v4.toml", "rb"))
r = load_region(REG, dict(cfg["loss"]), device="cpu")
td = r["train_data"]
tt = pd.DatetimeIndex(pd.to_datetime(r["times"])[td.train_slice.start:])
qo = td.q_obs.numpy()
sid = td.station_idx.numpy()
# temperature max du domaine, canal 2 du forcage ; moyenne sur les noeuds amont
tmax = td.forcing[:, :, 2].numpy().mean(axis=1)
mois = tt.month
hiver = np.isin(mois, [12, 1, 2, 3])
gel = tmax < 0.0

print(f"[recession] {REG.upper()} : {qo.shape[1]} stations, {len(tt)} jours, "
      f"segments >= {NMIN} j, hiver dec-mars, Tmax < 0 C", flush=True)

taux_tous = []
for s in range(qo.shape[1]):
    q = qo[:, s]
    ok = np.isfinite(q) & (q > 0) & hiver & gel
    taux_s = []
    i = 0
    while i < len(q) - NMIN:
        if not ok[i]:
            i += 1; continue
        j = i
        while j + 1 < len(q) and ok[j + 1] and q[j + 1] < q[j]:
            j += 1
        if j - i + 1 >= NMIN:
            seg = np.log(q[i:j + 1])
            k = -np.polyfit(np.arange(len(seg)), seg, 1)[0]
            if 0 < k < 1:
                taux_s.append(k)
            i = j + 1
        else:
            i += 1
    if taux_s:
        taux_tous.extend(taux_s)
        print(f"  station {s:2d} : {len(taux_s):3d} recessions, k median {np.median(taux_s):.4f} /j "
              f"(temps de residence {1/np.median(taux_s):5.1f} j)")

t = np.array(taux_tous)
print(f"\n=== ENSEMBLE : {len(t)} recessions hivernales pures ===")
for p in (10, 25, 50, 75, 90):
    v = np.percentile(t, p)
    print(f"  centile {p:2d} : k = {v:.4f} /j  (residence {1/v:6.1f} j)")
print(f"\n  k_gw du champion : 0.0645 /j (residence 15.5 j)")
print(f"  rapport au median mesure : x{0.0645/np.median(t):.2f}")
