"""Télécharge le forçage CaSR v3.2 (ECCC HPFX) pour les tuiles couvrant SLSO,
les variables utiles au modèle, 2000-2024. NetCDF tuilés ~44 Mo/fichier.
Reprise/retry ; saute les fichiers déjà présents et les 404.
  python .runs/slso/fetch_casr.py
"""
import os, sys, time, urllib.request, urllib.error
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE = "https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/data/CaSRv3.2/netcdf_tile"
OUT = ".runs/slso/data/casr"
os.makedirs(OUT, exist_ok=True)

# SLSO mosaïque 2x2 : 32% des nœuds (est du bassin) débordaient de la seule colonne
# rlon526-560 -> ajout colonne est rlon561-595 (vérifié contre les coords en pôle tourné).
TILES = ["rlon526-560_rlat351-385", "rlon526-560_rlat386-420",
         "rlon561-595_rlat351-385", "rlon561-595_rlat386-420"]
# (code variable ECCC, nom de fichier) — A_=analyse (obs-contraint), P_=champ modèle
VARS = [
    "A_PR0_SFC",     # précipitation (mm/h, analyse CaPA)
    "A_TT_1.5m",     # température air 1.5m (°C, analyse) -> Tmin/Tmax
    "A_TD_1.5m",     # point de rosée 1.5m (°C, analyse) -> e_a
    "P_FB_SFC",      # flux solaire descendant (W/m2) -> R_n (SW)
    "P_FI_SFC",      # flux infrarouge descendant (W/m2) -> R_n (LW)
    "P_UVC_10m",     # vitesse vent 10m (modulus, m/s) -> u2
]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015",
          "2016-2019", "2020-2023", "2024-2024"]

total = ok = skip = miss = 0
for tile in TILES:
    for var in VARS:
        for ch in CHUNKS:
            total += 1
            fn = f"CaSR_v3.2_{var}_{tile}_{ch}.nc"
            url = f"{BASE}/{tile}/{fn}"
            dst = os.path.join(OUT, fn)
            if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
                skip += 1; continue
            for attempt in range(4):
                try:
                    t0 = time.time()
                    urllib.request.urlretrieve(url, dst)
                    mb = os.path.getsize(dst) / 1048576
                    ok += 1
                    print(f"[ok] {fn}  {mb:.1f} Mo  {time.time()-t0:.0f}s  ({ok}/{total})", flush=True)
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        miss += 1
                        print(f"[404] {fn} (variable/tranche absente)", flush=True)
                        break
                    print(f"[retry {attempt}] {fn} HTTP {e.code}", flush=True); time.sleep(5)
                except Exception as e:
                    print(f"[retry {attempt}] {fn} : {e}", flush=True); time.sleep(5)
            else:
                print(f"[FAIL] {fn} apres 4 essais", flush=True)

print(f"\nCASR FETCH DONE : {ok} telecharges, {skip} deja la, {miss} absents (404), sur {total}", flush=True)
