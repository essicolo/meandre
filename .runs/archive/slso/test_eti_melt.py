"""Validation UNITAIRE de la fonte ETI (radiation réelle) vs degré-jour, sur une
saison de fonte synthétique, AVANT tout branchage dans le modèle. Vérifie :
  1) conservation : les deux finissent par fondre tout le manteau ;
  2) timing : ETI concentre la fonte quand la radiation est forte (printemps tardif),
     donc DÉCALE le barycentre de fonte vs le degré-jour piloté par la température ;
  3) différentiabilité : le gradient remonte vers tf et srf.
  python .runs/slso/test_eti_melt.py
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, torch
from hydrotel_clone.snow import DegreJourModifie, init_ce, init_state

torch.set_default_dtype(torch.float64)
N = 4                      # 4 nœuds identiques
DAYS = 100                 # 1er mars -> début juin
snow = DegreJourModifie(pas_de_temps=24)

# Terrain plat, params Hydrotel par défaut
lat = torch.full((N,), 48.0); pente = torch.zeros(N); orient = torch.zeros(N)
ce1, ce0 = init_ce(lat, pente, orient)
def base_p(extra):
    p = dict(lat=lat, ce1=ce1, ce0=ce0,
             pct_conifers=torch.ones(N), pct_feuillus=torch.zeros(N), pct_autres=torch.zeros(N),
             coeff_fonte_conifers=torch.full((N,), 0.012), coeff_fonte_feuillus=torch.full((N,), 0.014),
             coeff_fonte_decouver=torch.full((N,), 0.016),
             seuil_fonte_conifers=torch.zeros(N), seuil_fonte_feuillus=torch.zeros(N),
             seuil_fonte_decouver=torch.zeros(N),
             taux_fonte_geo=torch.full((N,), 0.5), densite_max=torch.full((N,), 466.0),
             constante_tassement=torch.full((N,), 0.1))
    p.update(extra); return p

# Forçage synthétique : T monte de -8 à +12, SW monte de 80 à 320 W/m² (saison).
jours = torch.arange(1, DAYS + 1)
doy = 60 + jours            # ~1er mars
tmean = -8.0 + 20.0 * (jours.double() / DAYS)
tmin = tmean - 4.0; tmax = tmean + 4.0
sw = 80.0 + 240.0 * (jours.double() / DAYS)          # W/m² incident
SWE0 = 250.0                                          # mm, manteau initial

def run(mode, tf=None, srf=None, requires_grad=False):
    st = init_state(N)
    # dépose tout le SWE au pas 0 comme neige initiale dans le stock conifer
    s, h, c, e = st["conifers"]
    st["conifers"] = (s + SWE0 / 1000.0, h + SWE0 / 1000.0 / 0.3, c, e)
    p = base_p(dict(melt_mode=mode, tf=tf, srf=srf))
    apports = []
    for k in range(DAYS):
        sw_k = sw[k].expand(N) if mode == "eti" else None
        ap, st = snow(tmin[k].expand(N), tmax[k].expand(N),
                      torch.zeros(N), (neige if (neige := torch.zeros(N)) is not None else 0),
                      doy[k].expand(N).double(), st, p, sw_in=sw_k)
        apports.append(ap[0])
    a = torch.stack(apports)                          # (DAYS,) apport nœud 0
    return a

dd = run("degree_day")
tf = torch.tensor(0.0012, requires_grad=True)         # m/°C/j
srf = torch.tensor(0.00020, requires_grad=True)       # m/j par W/m²
eti = run("eti", tf=tf.expand(N), srf=srf.expand(N))

def barycentre(a):
    a = a.detach().numpy(); w = a / (a.sum() + 1e-9)
    return float((np.arange(len(a)) * w).sum())

print(f"total fondu (mm)  : degré-jour {float(dd.sum()):.1f}  eti {float(eti.sum()):.1f}  (SWE0={SWE0})")
print(f"barycentre fonte (jour) : degré-jour {barycentre(dd):.1f}  eti {barycentre(eti):.1f}  "
      f"(ETI {'plus TARDIF' if barycentre(eti) > barycentre(dd) else 'plus précoce'})")
print(f"pic fonte (jour)  : degré-jour {int(dd.argmax())}  eti {int(eti.argmax())}")

# différentiabilité : la fonte TOTALE = SWE0 par conservation (dérivée nulle, normal).
# tf/srf agissent sur le TIMING : perte = 1er moment temporel Σ(jour·fonte), non
# contraint par conservation, = le signal réellement appris via le timing du débit.
print(f"graphe connecté ? eti.requires_grad={eti.requires_grad}  grad_fn={eti.grad_fn is not None}")
loss = (torch.arange(DAYS).double() * eti).sum()
loss.backward()
ok = (tf.grad is not None and tf.grad.abs() > 0 and srf.grad is not None and srf.grad.abs() > 0)
print(f"grad tf {float(tf.grad):.4e}  grad srf {float(srf.grad):.4e}  "
      f"(moment temporel sensible à tf/srf => ETI différentiable) {'OK' if ok else 'ECHEC'}")
