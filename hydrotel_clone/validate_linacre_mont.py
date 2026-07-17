"""Valide le clone Linacre contre etp.csv du run Hydrotel 4.3.6 MONT (4780 UHRH, 5 ans).
Entrées C++ exactes : tmin_jour/tmax_jour (interp. par UHRH), couvert_nival, albedo_neige.
Statiques : lat/alti (uhrh.csv), params linacre.csv (dont coeff optimisé par UHRH).
  python hydrotel_clone/validate_linacre_mont.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import hydrotel_clone.validate_soil_all_uhrh as VS
from hydrotel_clone.hydrotel_params import load_project
from hydrotel_clone.linacre import linacre_etp, load_linacre_params

torch.set_default_dtype(torch.float64)
MONT = "D:/meandre-data/quebec/mont-hy"
VS.DEL = MONT; VS.RES = MONT + "/simulation/simulation/resultat"

proj = load_project(MONT)
ids = proj["uhrh_ids"]; U = len(ids)
uh = proj["uhrh"]
lat = torch.tensor([uh[u]["lat"] for u in ids])
alti = torch.tensor([uh[u]["altitude"] for u in ids])
t_froid, t_chaud, albedo, coeff = load_linacre_params(MONT + "/simulation/simulation", ids)
print(f"{U} UHRH | coeff optimisé : min {coeff.min():.3f} méd {coeff.median():.3f} max {coeff.max():.3f}")

g = lambda n: VS.read_cpp(n, U)
tmn, tmx = g("tmin_jour"), g("tmax_jour")
couv, albn = g("couvert_nival"), g("albedo_neige")
etpC = g("etp")
NT = min(map(len, [tmn, tmx, couv, albn, etpC]))
etp_s = np.zeros((NT, U))
T = lambda a, i: torch.tensor(a[i])
with torch.no_grad():
    for i in range(NT):
        etp_s[i] = linacre_etp(T(tmn, i), T(tmx, i), lat, alti, T(couv, i), T(albn, i),
                               t_froid=t_froid, t_chaud=t_chaud, albedo=albedo, coeff=coeff).numpy()

rmse = np.sqrt(np.nanmean((etp_s - etpC[:NT]) ** 2, axis=0))
cum_s = etp_s.sum(axis=0).mean() / (NT / 365.25)
cum_c = etpC[:NT].sum(axis=0).mean() / (NT / 365.25)
print(f"ETP annuelle moyenne : clone {cum_s:.1f} | C++ {cum_c:.1f} mm/an")
print(f"RMSE par UHRH : méd {np.median(rmse):.5f}  p95 {np.percentile(rmse, 95):.5f}  max {rmse.max():.5f} mm/j")
worst = np.argsort(-rmse)[:4]
print("pires :", ", ".join(f"UHRH{ids[k]} ({rmse[k]:.4f})" for k in worst))
print("DONE")
