"""Récupération ECCC pleine période (2000-2024) par tuiles de 4° x 5 ans, avec cache.
Base du produit météo mixte : l'interpolation depuis les stations bat CaSR jusqu'à ~60 km
(courbe de bascule du 3 août), il faut donc les observations sur toute la fenêtre.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import pandas as pd
from meandre.data.eccc_loader import fetch_daily

BBOX = (-80.0, 44.5, -60.0, 53.5)
FEN = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2024)]
ok = ech = 0
for lon0 in range(-80, -60, 4):
    for y0, y1 in FEN:
        bb = (float(lon0), BBOX[1], float(min(lon0 + 4, -60)), BBOX[3])
        try:
            d = fetch_daily(bb, y0, y1)
            ok += 1
            print(f"  {bb[0]:.0f}..{bb[2]:.0f} {y0}-{y1} : {len(d):,}", flush=True)
        except Exception as ex:
            ech += 1
            print(f"  {bb[0]:.0f}..{bb[2]:.0f} {y0}-{y1} : ECHEC {type(ex).__name__}", flush=True)
print(f"[eccc] terminé : {ok} tuiles OK, {ech} échecs")
