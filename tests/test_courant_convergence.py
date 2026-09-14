"""Test de convergence du sous-pas de Courant : la partition ruissellement/hypodermique
dépend-elle du plafond n_substep ? Reproduction autonome sur la colonne BV3C2 seule.

Le C++ boucle jusqu'à épuisement du temps (while) ; le clone plafonne à n_substep
itérations. Quand l'échelle de Courant descend à DT_H/1152 (gel/saturation), un jour
exige ~1152 sous-pas ; un plafond de 64 ne traite qu'une fraction du jour et verse le
reliquat en ruissellement de surface (fermeture de masse du 2026-08-09).

  python tests/test_courant_convergence.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone, SOIL_TEXTURES, EPAISSEUR, KREC_DEFAULT, CIN_DEFAULT, DT_H

# Ce fichier est un SCRIPT : son corps s'execute a la collecte de pytest, faute de
# fonction de test. La double precision qu'il pose doit donc etre rendue a la fin, sinon
# elle fuit vers tous les modules charges ensuite et en casse seize.
_DTYPE_INITIAL = torch.get_default_dtype()
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

N = 200
z1, z2, z3 = EPAISSEUR
tex = SOIL_TEXTURES["silt_loam"]

def params():
    z = lambda v: torch.full((N,), float(v))
    return dict(
        thetas1=z(tex["thetas"]), thetas2=z(tex["thetas"]), thetas3=z(tex["thetas"]),
        thetacc1=z(tex["thetacc"]), thetacc2=z(tex["thetacc"]), thetacc3=z(tex["thetacc"]),
        thetapf1=z(tex["thetapf"]), thetapf2=z(tex["thetapf"]), thetapf3=z(tex["thetapf"]),
        ks1=z(tex["ks"]), ks2=z(tex["ks"]), ks3=z(tex["ks"]),
        psis1=z(tex["psis"]), psis2=z(tex["psis"]), psis3=z(tex["psis"]),
        b1=z(tex["lam"]), b2=z(tex["lam"]), b3=z(tex["lam"]),
        omegpi1=z((1 + 2 * tex["lam"]) / (2 + 2 * tex["lam"])),
        omegpi2=z((1 + 2 * tex["lam"]) / (2 + 2 * tex["lam"])),
        omegpi3=z((1 + 2 * tex["lam"]) / (2 + 2 * tex["lam"])),
        mm1=z(0.1), mm2=z(0.1), mm3=z(0.1), nn1=z(0.1), nn2=z(0.1), nn3=z(0.1),
        krec=z(KREC_DEFAULT), cin=z(CIN_DEFAULT),
        z1=z(z1), z2=z(z2), z3=z(z3),
        slope=z(0.03), fsa=z(1.0), fse=z(0.0), fsi=z(0.0),
        coef_recharge=z(0.0),
    )

print("Scenario A : sol pres de saturation (0.95 thetas), SANS gel, pluie 40 mm/j, 3 jours")
print("(saturation seule : porte binaire pinf=0 quand t1=thetas, l echelle fine vient des flux)")
print()
print("sous-pas | surf mm | hypo mm | L3 mm | total mm | fraction surface")

for n_sub in [24, 64, 128, 256, 512, 1152]:
    clone = BV3C2Clone(n_substep=n_sub, static=True)
    p = params()
    z = lambda v: torch.full((N,), float(v))
    t1, t2, t3 = z(0.95 * tex["thetas"]), z(0.95 * tex["thetas"]), z(0.95 * tex["thetas"])
    frozen_depth = z(0.0); swe = z(0.0)
    tot_r = tot_h = tot_b = 0.0
    for jour in range(3):
        out = clone(t1, t2, t3, z(40.0), z(0.1), frozen_depth, swe, p,
                    etr1_mm=z(0.0))
        prod_surf, prod_hypo, prod_base, rech, thetas_new, diag = out
        t1, t2, t3 = thetas_new
        tot_r += float(prod_surf.mean()); tot_h += float(prod_hypo.mean()); tot_b += float(prod_base.mean())
    tot = tot_r + tot_h + tot_b
    frac = tot_r / tot * 100 if tot > 1e-9 else 0.0
    print(f"{n_sub:8d} | {tot_r:7.2f} | {tot_h:7.2f} | {tot_b:6.2f} | {tot:7.2f} | {frac:5.1f} %")

print()
print("Scenario B : sol MOYEN (0.80 thetas), sans gel, pluie 40 mm/j, 3 jours")
for n_sub in [24, 64, 128, 256, 512, 1152]:
    clone = BV3C2Clone(n_substep=n_sub, static=True)
    p = params()
    z = lambda v: torch.full((N,), float(v))
    t1, t2, t3 = z(0.80 * tex["thetas"]), z(0.80 * tex["thetas"]), z(0.80 * tex["thetas"])
    frozen_depth = z(0.0); swe = z(0.0)
    tot_r = tot_h = tot_b = 0.0
    for jour in range(3):
        out = clone(t1, t2, t3, z(40.0), z(0.1), frozen_depth, swe, p,
                    etr1_mm=z(0.0))
        prod_surf, prod_hypo, prod_base, rech, thetas_new, diag = out
        t1, t2, t3 = thetas_new
        tot_r += float(prod_surf.mean()); tot_h += float(prod_hypo.mean()); tot_b += float(prod_base.mean())
    tot = tot_r + tot_h + tot_b
    frac = tot_r / tot * 100 if tot > 1e-9 else 0.0
    print(f"{n_sub:8d} | {tot_r:7.2f} | {tot_h:7.2f} | {tot_b:6.2f} | {tot:7.2f} | {frac:5.1f} %")

torch.set_default_dtype(_DTYPE_INITIAL)
