"""Comparaison CONTRÔLÉE de la génération de ruissellement : sol fidèle Hydrotel
(bv3c_hydrotel) vs sol actuel de méandre (soil.py), MÊME forçage, MÊMES
paramètres calibrés, gel activé équitablement des deux côtés.

Quantifie la divergence diagnostiquée : combien d'eau de freshet et d'orage
méandre absorbe en stockage là où Hydrotel la fait ruisseler. Forçage : année
boréale synthétique (accumulation hiver, fonte printanière sur sol gelé =
freshet, orages d'été).

  python .runs/slso-od/compare_soil_generation.py
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch

from meandre.vertical.bv3c_hydrotel import BV3CHydrotel, make_params, SOIL_TEXTURES, EPAISSEUR, KREC_DEFAULT
from meandre.vertical.soil import SoilModule
from meandre.vertical.frost import FrostModule

dev = "cpu"
torch.manual_seed(0)

# ── Paramètres calibrés Hydrotel SLSO (silt_loam surface, loam dessous) ──
tx1, tx2, tx3 = SOIL_TEXTURES["silt_loam"], SOIL_TEXTURES["loam"], SOIL_TEXTURES["loam"]
z1, z2, z3 = EPAISSEUR
def T(x): return torch.tensor(float(x), device=dev)

# ── Forçage : année boréale (apport = pluie+fonte au sol, mm/j ; T_air degC) ──
N = 365
apport = np.zeros(N); tair = np.zeros(N)
doy = np.arange(N)
# Température air : sinusoïde, min -18 en janvier, max +20 en juillet.
tair = -18.0 + 19.0 * (1 - np.cos(2*np.pi*(doy-15)/365)) * 0.5 + 19.0*0.0
tair = 1.0 - 19.0*np.cos(2*np.pi*(doy-200)/365)   # ~ -18 hiver, +20 été
# Hiver (j 0-90) : précip tombe en neige, peu d'apport au sol (stockée).
# Freshet (j 95-120) : fonte du couvert, gros apport sur sol encore gelé.
rng = np.random.default_rng(0)
apport[95:120] = rng.uniform(15, 38, 25)          # fonte printanière concentrée
# Été (j 150-270) : orages épars.
storm_days = rng.choice(np.arange(150, 280), size=18, replace=False)
apport[storm_days] = rng.uniform(8, 55, 18)
# Pluie d'automne légère.
apport[290:330] += rng.uniform(0, 6, 40)

apport_t = torch.tensor(apport, dtype=torch.float32)
tair_t = torch.tensor(tair, dtype=torch.float32)

# ── Modules ──
faithful = BV3CHydrotel(n_substeps_max=48)
p_faith = make_params("silt_loam", "loam", "loam", slope=0.04, krec=KREC_DEFAULT, device=dev)
meandre = SoilModule(z1=z1, vg_alpha=1.0, use_infiltration_excess=True, vsa_b=2.5)
frost = FrostModule()

# Paramètres méandre mappés sur Hydrotel (Campbell ks -> K_sat m/j).
ks1_md = T(tx1["ks"]*24); ks2_md = T(tx2["ks"]*24); ks3_md = T(tx3["ks"]*24)
por = (T(tx1["thetas"]), T(tx2["thetas"]), T(tx3["thetas"]))
fc = (T(tx1["thetacc"]), T(tx2["thetacc"]), T(tx3["thetacc"]))
wp = (T(tx1["thetapf"]), T(tx2["thetapf"]), T(tx3["thetapf"]))
fvert = (T(0.5), T(0.5), T(0.5))
frost_alpha = T(0.95); alpha_T = T(0.15)

# ── États initiaux (humidité de départ identique) ──
th0 = (0.35, 0.30, 0.30)
tf1, tf2, tf3 = T(th0[0]), T(th0[1]), T(th0[2])
tm1, tm2, tm3 = T(th0[0]), T(th0[1]), T(th0[2])
tsoil = T(2.0); S_uz = None

ro_f = np.zeros(N); ro_m = np.zeros(N); frz = np.zeros(N, bool)
for t in range(N):
    ta = tair_t[t]; ap = apport_t[t]
    # ET demande : ~2.5 mm/j en été (T>10), 0 l'hiver.
    et = torch.clamp((ta - 5.0) / 8.0, 0.0, 1.0) * 2.5
    # Gel : T_sol via frost module (les deux côtés le partagent).
    ks_eff, tsoil = frost(ta.unsqueeze(0), tsoil.unsqueeze(0), ks1_md.unsqueeze(0),
                          frost_alpha.unsqueeze(0), alpha_T.unsqueeze(0))
    ks_eff = ks_eff.squeeze(0); tsoil = tsoil.squeeze(0)
    frozen = tsoil < 0.0
    frz[t] = bool(frozen)

    # ── Fidèle Hydrotel : porte gel binaire interne ──
    rof, itf, baf, ref, (tf1, tf2, tf3), _ = faithful(
        tf1, tf2, tf3, ap, et, frozen, T(0.0), p_faith)
    ro_f[t] = rof.item()

    # ── méandre : gel via K_sat réduit (ks_eff), même apport ──
    out = meandre(ap, et, T(0.0), T(0.0), tm1, tm2, tm3,
                  ks_eff, ks2_md, ks3_md, por[0], por[1], por[2],
                  fc[0], fc[1], fc[2], wp[0], wp[1], wp[2],
                  fvert[0], fvert[1], fvert[2], z2=T(z2), z3=T(z3), S_uz=S_uz)
    tm1, tm2, tm3, R_surf, inter, base, S_uz = out
    ro_m[t] = R_surf.item()

# ── Bilan ──
fresh = slice(95, 125)
summer = slice(150, 285)
print(f"{'':18} | {'Hydrotel fidèle':>16} | {'méandre soil.py':>16}")
print(f"{'apport total':18} | {apport.sum():16.1f} | {apport.sum():16.1f}  (mm/an)")
print(f"{'RUNOFF total':18} | {ro_f.sum():16.1f} | {ro_m.sum():16.1f}")
print(f"{'coeff ruiss. an':18} | {ro_f.sum()/apport.sum():16.2f} | {ro_m.sum()/apport.sum():16.2f}")
print(f"{'-- FRESHET (avr) --':18} |")
print(f"{'  apport freshet':18} | {apport[fresh].sum():16.1f} | {apport[fresh].sum():16.1f}")
print(f"{'  runoff freshet':18} | {ro_f[fresh].sum():16.1f} | {ro_m[fresh].sum():16.1f}")
print(f"{'  pic runoff jour':18} | {ro_f[fresh].max():16.1f} | {ro_m[fresh].max():16.1f}")
print(f"{'-- ÉTÉ (orages) --':18} |")
print(f"{'  apport été':18} | {apport[summer].sum():16.1f} | {apport[summer].sum():16.1f}")
print(f"{'  runoff été':18} | {ro_f[summer].sum():16.1f} | {ro_m[summer].sum():16.1f}")
print(f"{'  pic runoff jour':18} | {ro_f[summer].max():16.1f} | {ro_m[summer].max():16.1f}")
print(f"\njours gelés (T_sol<0) : {frz.sum()} / {N}")
print(f"freshet absorbée par méandre vs Hydrotel : "
      f"{(ro_f[fresh].sum()-ro_m[fresh].sum()):.1f} mm de moins en ruissellement")
