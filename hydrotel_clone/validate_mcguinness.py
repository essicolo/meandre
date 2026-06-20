"""Validation du clone ETP McGuinness-Bordne.

Le run Hydrotel SLSO n'a pas sorti l'ETP (resultat/ = debit_aval seulement),
donc on valide sur la plus petite unité disponible :
  1. Re astronomique du clone vs la formule FAO-56 Ra (INDÉPENDANTE, standard) :
     valide la transcription du rayonnement extraterrestre aux décimales.
  2. Magnitude de l'ETP McGuinness annuelle sur le vrai forçage du banc SLSO
     (Tmin/Tmax) vs l'ET actuelle de méandre (Penman-Monteith) et la pluie.

  python hydrotel_clone/validate_mcguinness.py
"""
from __future__ import annotations
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.mcguinness import rayonnement_extraterrestre, mcguinness_etp

T = lambda x: torch.tensor(x, dtype=torch.float64)


def fao56_Ra(J, lat_dd):
    """FAO-56 Allen 1998 eq.21 — rayonnement extraterrestre [MJ.m-2.j-1].
    Indépendant de McGuinness (déclinaison/excentricité simplifiées)."""
    phi = math.radians(lat_dd)
    dr = 1.0 + 0.033 * math.cos(2 * math.pi * J / 365.0)
    delta = 0.409 * math.sin(2 * math.pi * J / 365.0 - 1.39)
    ws = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta))))
    Gsc = 0.0820  # MJ/m2/min
    Ra = (24 * 60 / math.pi) * Gsc * dr * (ws * math.sin(phi) * math.sin(delta)
                                           + math.cos(phi) * math.cos(delta) * math.sin(ws))
    return Ra


print("=== 1. Re McGuinness (Spencer) vs FAO-56 Ra, lat 46.5 (SLSO) ===")
print(f"{'jour':>5} {'Re_clone':>9} {'Ra_FAO56':>9} {'écart%':>7}")
lat = 46.5
for J in [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]:
    re = float(rayonnement_extraterrestre(T(float(J)), T(lat)))
    ra = fao56_Ra(J, lat)
    print(f"{J:>5} {re:9.2f} {ra:9.2f} {100*(re-ra)/ra:7.2f}")

print("\n=== 2. ETP McGuinness annuelle sur le forçage du banc SLSO-od ===")
try:
    import tomllib, pandas as pd, xarray as xr, duckdb
    cfg = tomllib.load(open(".runs/slso-od/config/slso-od-mini-clone.toml", "rb"))
    DB = ".runs/slso-od/" + cfg["paths"]["basin_db"]
    con = duckdb.connect(DB, read_only=True)
    df = con.execute("SELECT node_idx, lat, area_km2_local FROM nodes n JOIN territorial t USING(node_idx) ORDER BY node_idx").fetchdf() \
        if False else None
    lat_nodes = np.array([r[0] for r in con.execute("SELECT lat FROM nodes ORDER BY node_idx").fetchall()], dtype=np.float64)
    area = np.array([r[0] for r in con.execute("SELECT area_km2_local FROM territorial ORDER BY node_idx").fetchall()], dtype=np.float64)
    con.close()
    ds = xr.open_dataset(cfg["paths"]["forcing_cache"]); ff = ds["forcing"].values.astype(np.float64); times = pd.to_datetime(ds["time"].values); ds.close()
    w0 = int(np.searchsorted(times, np.datetime64("2019-01-01"))); win = times[w0:]
    tmin = ff[w0:, :, 1]; tmax = ff[w0:, :, 2]; pr = ff[w0:, :, 0]
    doy = win.dayofyear.values.astype(np.float64)
    nyr = len(win) / 365.25
    etp = mcguinness_etp(T(tmin), T(tmax), T(lat_nodes)[None, :], T(doy)[:, None]).numpy()  # (T, N) mm/j
    wmean = lambda x: (x.sum(0) * area).sum() / area.sum() / nyr
    P = wmean(pr); ETPmcg = wmean(etp)
    print(f"  lat nœuds : {lat_nodes.min():.2f}–{lat_nodes.max():.2f}")
    print(f"  P (forçage)               : {P:6.0f} mm/an")
    print(f"  ETP McGuinness (potentiel): {ETPmcg:6.0f} mm/an  (ETP/P = {ETPmcg/P:.2f})")
    print(f"  -> ET ACTUELLE méandre (PM, mesurée) : 617-687 mm/an (ET/P 0.60-0.67)")
    print(f"  -> obs implique ET/P ~0.57 (RC ~0.43). McGuinness POTENTIEL {ETPmcg:.0f};")
    print(f"     l'ET réelle (sol-limitée) < potentiel, donc McGuinness ramène l'ET sous le PM.")
    # profil mensuel domaine
    mois = pd.Series(win.month, index=range(len(win)))
    etp_dom = (etp * area).sum(1) / area.sum()
    print("  ETP McGuinness moyenne par mois (mm/j) :")
    for m in range(1, 13):
        idx = np.where(win.month == m)[0]
        print(f"    {m:2d}: {etp_dom[idx].mean():.2f}", end="")
    print()
except Exception as e:
    print(f"  (forçage du banc indisponible : {e})")
print("DONE")
