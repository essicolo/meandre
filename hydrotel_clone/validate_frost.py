"""Validation du clone de gel RANKINEN contre Hydrotel C++ sur DELISLE UHRH 1.

DELISLE n'active pas le gel par défaut (TEMPERATURE DU SOL vide) ; la référence
a été générée en relançant Hydrotel-DELISLE en config RANKINEN (sur copie de
sécurité, restaurée ensuite). La sortie profondeur_gel.csv est sauvée dans
hydrotel_clone/_profondeur_gel_delisle.csv.

On pilote le clone avec Tmin/Tmax + la HAUTEUR du couvert nival (m) d'Hydrotel,
et on compare la profondeur de gel (cm) jour par jour.

Pour régénérer la référence :
  cd .../hydrotel/DemoProject/DELISLE/simulation/simulation
  cp simulation.csv simulation.csv.bak ; cp -r resultat resultat.bak
  sed -i 's/^TEMPERATURE DU SOL;/TEMPERATURE DU SOL;RANKINEN/' simulation.csv
  wsl ../../../../gcc/hydrotel ../../../DELISLE.csv
  cp resultat/profondeur_gel.csv <clone>/_profondeur_gel_delisle.csv
  cp simulation.csv.bak simulation.csv ; rm -rf resultat ; mv resultat.bak resultat

  python hydrotel_clone/validate_frost.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.frost import Rankinen, n_intervalles

WDEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE/simulation/simulation/resultat"
REF = os.path.join(os.path.dirname(__file__), "_profondeur_gel_delisle.csv")
UHRH = 1
T = lambda x: torch.tensor([float(x)], dtype=torch.float64)


def rc(path, c=UHRH, h=2):
    L = open(path, encoding="latin-1").read().splitlines()
    return np.array([float(l.split(';')[c]) for l in L[h:] if len(l.split(';')) > c])


tn, tx = rc(f"{WDEL}/tmin.csv"), rc(f"{WDEL}/tmax.csv")
hn = rc(f"{WDEL}/hauteur_neige.csv")           # hauteur couvert nival [m]
gel_h = rc(REF)                                 # profondeur_gel C++ [cm]
N = min(map(len, [tn, tx, hn, gel_h]))

# UHRH1 : Z=(0.1,0.4,1.0) -> profondeur 1.5 m ; params rankinen.csv DELISLE
nd = n_intervalles(1.5, 0.05)
mod = Rankinen(intervalle=0.05, temp_ini_base=4.0, seuil_gel=-0.5, fs=2.35,
               kt=0.8, cs=1.0e6, cice=4.0e6, pas_de_temps=24)

gel_c = np.zeros(N); profil = None
for i in range(N):
    if i == 0:
        profil = mod.init_profil(T(tn[i]), T(tx[i]), T(hn[i]), nd)
    profil, gel = mod(T(tn[i]), T(tx[i]), T(hn[i]), profil, 0.1, 0.4, 1.0)
    gel_c[i] = float(gel)

rmse = float(np.sqrt(np.nanmean((gel_c[:N] - gel_h[:N]) ** 2)))
print(f"UHRH {UHRH} : {N} jours, n_depth={nd}")
print(f"RMSE profondeur_gel = {rmse:.4f} cm   max abs = {np.max(np.abs(gel_c[:N]-gel_h[:N])):.4f} cm")
print(f"gel max : clone {gel_c.max():.2f} vs C++ {gel_h[:N].max():.2f} cm ; "
      f"jours geles : clone {int((gel_c>0).sum())} vs C++ {int((gel_h[:N]>0).sum())}")
print("DONE")
