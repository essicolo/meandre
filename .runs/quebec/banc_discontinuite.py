"""Le champ provincial saute-t-il aux FRONTIERES DE REGION ?

Mesure du 2026-09-03, a partir des cartes de parametres du rapport. Chaque region porte
son propre champ, entraine separement : deux troncons voisins de part et d'autre d'une
frontiere administrative recoivent leurs parametres de DEUX reseaux differents. La
frontiere n'a aucune existence hydrologique.

Protocole : pour toutes les paires de troncons distants de moins de dix kilometres, on
compare l'ecart relatif du parametre quand les deux appartiennent a la meme region et
quand une frontiere les separe. Le rapport des deux medianes est le saut de frontiere.

Resultat au 2026-09-03 (28 035 troncons, 15 regions, 8 696 paires transfrontalieres) :
K_sat 121 a 178 % d'ecart a la frontiere contre 2 % a l'interieur, soit 57 a 83 fois ;
Z2 286 fois ; Z3 113 fois ; T_melt 12 fois. Le champ provincial est une mosaique de
quinze champs disjoints, pas un champ.

Un champ provincial unique doit ramener ces rapports vers 1.

  python .runs/quebec/banc_discontinuite.py
"""

import glob
import os
import numpy as np

D_KM = 10.0
RT = 6371.0

def charge():
    n, X, R = [], [], []
    P = {}
    for f in sorted(glob.glob("D:/meandre-data/quebec/rapport/rap-*-avec.npz")):
        reg = os.path.basename(f).split("-")[1]
        d = np.load(f)
        c = d["coords"]
        X.append(c); R.append(np.full(len(c), reg))
        for k in d.files:
            if k.startswith("param_"):
                P.setdefault(k, []).append(d[k])
    X = np.vstack(X); R = np.concatenate(R)
    P = {k: np.concatenate(v) for k, v in P.items() if sum(len(x) for x in v) == len(X)}
    return X, R, P

X, R, P = charge()
print(f"{len(X)} troncons, {len(set(R))} regions, {len(P)} champs")

# Paires proches par grille (rapide) : maille de D_KM
lon, lat = X[:, 0], X[:, 1]
mlat = np.cos(np.radians(np.median(lat)))
gx = np.floor(lon * mlat * 111.0 / D_KM).astype(int)
gy = np.floor(lat * 111.0 / D_KM).astype(int)
from collections import defaultdict
cases = defaultdict(list)
for i, (a, b) in enumerate(zip(gx, gy)):
    cases[(a, b)].append(i)

rng = np.random.default_rng(0)
meme, autre = [], []
for (a, b), idx in cases.items():
    vois = idx + cases.get((a + 1, b), []) + cases.get((a, b + 1), [])
    if len(vois) < 2:
        continue
    v = np.array(vois)
    if len(v) > 40:
        v = rng.choice(v, 40, replace=False)
    for p in range(len(v)):
        for q in range(p + 1, len(v)):
            i, j = v[p], v[q]
            dx = (lon[i] - lon[j]) * mlat * 111.0
            dy = (lat[i] - lat[j]) * 111.0
            if dx * dx + dy * dy > D_KM * D_KM:
                continue
            (meme if R[i] == R[j] else autre).append((i, j))
meme = np.array(meme); autre = np.array(autre)
print(f"paires a moins de {D_KM:.0f} km : {len(meme)} dans la meme region, "
      f"{len(autre)} de part et d'autre d'une frontiere\n")

print(f"{'champ':22s} {'ecart meme region':>18s} {'ecart transfrontalier':>22s} {'saut':>7s}")
lignes = []
for k, v in P.items():
    v = v.astype(float)
    def ec(pr):
        if len(pr) == 0: return np.nan
        a, b = v[pr[:, 0]], v[pr[:, 1]]
        return float(np.median(np.abs(a - b) / (np.abs(a + b) / 2 + 1e-12)))
    e1, e2 = ec(meme), ec(autre)
    if not np.isfinite(e1) or e1 <= 0 or not np.isfinite(e2): continue
    lignes.append((e2 / e1, k, e1, e2))
for saut, k, e1, e2 in sorted(lignes, reverse=True)[:12]:
    print(f"{k[6:]:22s} {100*e1:17.1f}% {100*e2:21.1f}% {saut:6.1f}x")
