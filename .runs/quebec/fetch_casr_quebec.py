"""Télécharge les tuiles CaSR v3.2 pour tout le Québec (union tiles_needed.txt).
Sortie D:/meandre-data/casr (C: plein). Saute les fichiers déjà présents ici OU dans
le cache SLSO historique (.runs/slso/data/casr). Reprise/retry, tolère les 404.
  python .runs/quebec/fetch_casr_quebec.py
"""
import os, sys, time, urllib.request, urllib.error
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE = "https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/data/CaSRv3.2/netcdf_tile"
OUT = "D:/meandre-data/casr"
LEGACY = ".runs/slso/data/casr"
os.makedirs(OUT, exist_ok=True)
with open("D:/meandre-data/quebec/tiles_needed.txt") as f:
    TILES = [t.strip().replace("rlon", "rlon").replace("_rlat", "_rlat") for t in f if t.strip()]
VARS = ["A_PR0_SFC", "A_TT_1.5m", "A_TD_1.5m", "P_FB_SFC", "P_FI_SFC", "P_UVC_10m"]
CHUNKS = ["2000-2003", "2004-2007", "2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2024"]

total = ok = skip = miss = err = 0
for tile in TILES:
    tdir = tile  # ex rlon561-595_rlat386-420 ; l'URL utilise le même nom de répertoire
    for var in VARS:
        for ch in CHUNKS:
            total += 1
            fn = f"CaSR_v3.2_{var}_{tile}_{ch}.nc"
            dst = os.path.join(OUT, fn)
            legacy = os.path.join(LEGACY, fn)
            if (os.path.exists(dst) and os.path.getsize(dst) > 1_000_000) or \
               (os.path.exists(legacy) and os.path.getsize(legacy) > 1_000_000):
                skip += 1; continue
            url = f"{BASE}/{tile}/{fn}"
            done = False
            for attempt in range(4):
                try:
                    urllib.request.urlretrieve(url, dst + ".part")
                    os.replace(dst + ".part", dst); ok += 1; done = True
                    print(f"[ok] {fn}", flush=True); break
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        miss += 1; print(f"[404] {fn}", flush=True); done = True; break
                    time.sleep(5 * (attempt + 1))
                except Exception:
                    time.sleep(5 * (attempt + 1))
            if not done:
                err += 1; print(f"[ERR] {fn}", flush=True)
print(f"\ntotal {total} | téléchargés {ok} | déjà là {skip} | 404 {miss} | erreurs {err}")
