"""Validation de la COLONNE COMPLÈTE (sol BV3C2 + milieu humide isolé) sur les 495
UHRH de DELISLE vs C++ Hydrotel. Découverte clé : production_surf/hypo/base.csv est
POST-milieu-humide (bv3c2.cpp l.838-895). Le sol seul est fidèle (theta à la
décimale) ; il faut chaîner le milieu humide isolé pour comparer aux sorties.

Chaîne fidèle (l.851-889) :
  prodOld = surf+hypo+base (sol, post-recharge)
  apport_wet = apport * wetfr   (wetfr = wet_a/uhrh_a, l.2777+862+867)
  CalculMilieuHumideIsole : prod = prodOld*(1-wetdrafr) + wetprod  (l.1483+1517)
  repartition : surf = surfOld/prodOld * prod  (idem hypo, base)
Volume du milieu humide ÉVOLUÉ depuis l'état initial (etats/bilan_vertical...csv,
MH WETVOL), comme le C++ (SetWetvol l.190).

  python hydrotel_clone/validate_column_full.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone
from hydrotel_clone.milieu_humide import init_wetland_geom, calcul_milieu_humide_isole
from hydrotel_clone.hydrotel_params import load_project
import hydrotel_clone.validate_soil_all_uhrh as VS

torch.set_default_dtype(torch.float64)
DEL = VS.DEL
SIM = DEL + "/simulation/simulation"
ETAT = DEL + "/etats/bilan_vertical_2020010100.csv"


def load_wetlands(path, ids):
    """milieux_humides_isoles.csv → dict uhrh → params (None si absent)."""
    out = {}
    for ln in open(path, encoding="latin-1").read().splitlines():
        c = ln.split(";")
        if len(c) >= 11 and c[0].strip().isdigit():
            uid = int(c[0]); v = [float(x) for x in c[1:11]]
            uhrh_a, wet_a, wet_dra_fr, frac, wetdnor, wetdmax, ksat_bs, c_ev, c_prod, _ = v
            A, B, wetnvol, wetmxvol = init_wetland_geom(wet_a, wetdmax, frac, wetdnor)
            out[uid] = dict(wetfr=wet_a / uhrh_a, wetdrafr=wet_dra_fr, hru_ha=uhrh_a * 100.0,
                            A=A, B=B, wetnvol=wetnvol, wetmxvol=wetmxvol,
                            ksat_bs=ksat_bs, c_ev=c_ev, c_prod=c_prod)
    return out


def load_init_wetvol(path, ids):
    """etat bilan vertical → wetvol initial par UHRH (m3)."""
    out = {}
    for ln in open(path, encoding="latin-1").read().splitlines():
        c = ln.split(";")
        if len(c) >= 5 and c[0].strip().isdigit():
            out[int(c[0])] = float(c[4])
    return out


def main():
    proj = load_project(DEL)
    ids = proj["uhrh_ids"]; U = len(ids)
    wl = load_wetlands(SIM + "/milieux_humides_isoles.csv", ids)
    wv0 = load_init_wetvol(ETAT, ids)
    has_wet = np.array([u in wl for u in ids])
    print(f"{U} UHRH, {has_wet.sum()} avec milieu humide isolé")

    g = lambda n: VS.read_cpp(n, U)
    apC, etpC = g("apport"), g("etp")
    e1C, e2C, e3C = g("etr1"), g("etr2"), g("etr3")
    th1C, th2C, th3C = g("theta1"), g("theta2"), g("theta3")
    psC, phC, pbC = g("production_surf"), g("production_hypo"), g("production_base")
    couv = g("couvert_nival")
    NT = min(map(len, [apC, th1C, couv]))

    # tenseurs wetland par UHRH (NaN si pas de MH ; on masque)
    def wt(key, default=0.0):
        return torch.tensor([wl[u][key] if u in wl else default for u in ids])
    wetfr = wt("wetfr"); wetdrafr = wt("wetdrafr"); hru_ha = wt("hru_ha", 1.0)
    A = wt("A", 1.0); B = wt("B"); wetnvol = wt("wetnvol", 1.0); wetmxvol = wt("wetmxvol", 1.0)
    ksat_bs = wt("ksat_bs"); c_ev = wt("c_ev"); c_prod = wt("c_prod", 1.0)
    wmask = torch.tensor(has_wet)
    wet_vol = torch.tensor([wv0.get(u, 0.0) for u in ids])

    P = VS.build_psoil(proj, ids)
    soil = BV3C2Clone(n_substep=2000)
    t1 = torch.tensor(th1C[0]); t2 = torch.tensor(th2C[0]); t3 = torch.tensor(th3C[0])
    z = torch.zeros(U)
    T = lambda a, i: torch.tensor(a[i])
    ps_s = np.zeros((NT, U)); ph_s = np.zeros((NT, U)); pb_s = np.zeros((NT, U))
    with torch.no_grad():
        for i in range(1, NT):
            surf, hyp, base, rech, (t1, t2, t3), _ = soil(
                t1, t2, t3, T(apC, i), T(etpC, i), z, T(couv, i), P,
                etr1_mm=T(e1C, i), etr2_mm=T(e2C, i), etr3_mm=T(e3C, i))
            prodOld = surf + hyp + base
            ap_wet = T(apC, i) * wetfr
            evp = T(etpC, i)
            wet_vol_new, sep, flwi, flwo, wetprod = calcul_milieu_humide_isole(
                torch.clamp(wet_vol, min=1e-9), ap_wet, evp, prodOld, hru_ha, wetdrafr,
                A, B, wetnvol, wetmxvol, ksat_bs, c_ev, c_prod, pdt=24)
            prod_new = prodOld * (1.0 - wetdrafr) + wetprod
            safe = prodOld > 0
            scale = torch.where(safe, prod_new / torch.clamp(prodOld, min=1e-12), torch.ones_like(prodOld))
            surf_w = torch.where(wmask, surf * scale, surf)
            hyp_w = torch.where(wmask, hyp * scale, hyp)
            base_w = torch.where(wmask, base * scale, base)
            wet_vol = torch.where(wmask, wet_vol_new, wet_vol)
            ps_s[i] = surf_w.numpy(); ph_s[i] = hyp_w.numpy(); pb_s[i] = base_w.numpy()

    sl = slice(1, NT)
    def rmse_per(a, b): return np.sqrt(np.nanmean((a[sl] - b[sl]) ** 2, axis=0))
    rps, rph, rpb = rmse_per(ps_s, psC), rmse_per(ph_s, phC), rmse_per(pb_s, pbC)
    def cum(a): return a[sl].sum(axis=0).mean()
    print(f"\n=== BILAN CUMULÉ moyen {U} UHRH (mm) colonne | C++ ===")
    print(f"  prod_surf : {cum(ps_s):8.1f} | {cum(psC):8.1f}")
    print(f"  prod_hypo : {cum(ph_s):8.1f} | {cum(phC):8.1f}")
    print(f"  prod_base : {cum(pb_s):8.1f} | {cum(pbC):8.1f}")
    w = has_wet; nw = ~has_wet
    print(f"\n=== RMSE prod_surf par UHRH (médiane / p95 / max) ===")
    print(f"  TOUS         méd {np.median(rps):.4f}  p95 {np.percentile(rps,95):.4f}  max {np.max(rps):.4f}")
    print(f"  avec MH ({w.sum():3d}) méd {np.median(rps[w]):.4f}  p95 {np.percentile(rps[w],95):.4f}  max {np.max(rps[w]):.4f}")
    print(f"  sans MH ({nw.sum():3d}) méd {np.median(rps[nw]):.4f}  p95 {np.percentile(rps[nw],95):.4f}  max {np.max(rps[nw]):.4f}")
    print(f"  prod_hypo TOUS méd {np.median(rph):.4f}  p95 {np.percentile(rph,95):.4f}")
    # UHRH 376 focus
    k = ids.index(376)
    print(f"\nUHRH376 (MH 66%): cumul surf colonne={ps_s[sl,k].sum():.1f} C++={psC[sl,k].sum():.1f} | "
          f"jour68 surf={ps_s[68,k]:.3f} C++={psC[68,k]:.3f}  hypo={ph_s[68,k]:.3f} C++={phC[68,k]:.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
