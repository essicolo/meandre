"""Validation du sol BV3C2 cloné sur LES 495 UHRH de DELISLE (pas 1 cherry-pickée),
contre C++ Hydrotel. Params 100% lus dans les fichiers du projet (hydrotel_params),
zéro hardcode, zéro NeRF.

Isolation du sol : on PILOTE BV3C2 avec les intermédiaires C++ comme entrées
(apport, etp, etr1/2/3, couvert), theta partant du theta C++ jour 0, gel=0 (comme
validate_chain, pas de profondeur_gel.csv en sortie). On compare production_surf/
hypo/base + theta1/2/3 à C++, par UHRH, sur toute la trajectoire.

Si ça colle partout → le sol cloné est prouvé sur la population complète. Si des
UHRH divergent → on les localise (texture, occupation) et on corrige la FIDÉLITÉ,
sans compensation.

  python hydrotel_clone/validate_soil_all_uhrh.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from hydrotel_clone.bv3c2 import BV3C2Clone
from hydrotel_clone.hydrotel_params import load_project, uhrh_fractions

torch.set_default_dtype(torch.float64)
DEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE"
RES = DEL + "/simulation/simulation/resultat"


def read_cpp(name, n_uhrh):
    """Lit resultat/<name>.csv → (NT, n_uhrh). 2 lignes d'entête, col k = UHRH k."""
    lines = open(f"{RES}/{name}.csv", encoding="latin-1").read().splitlines()[2:]
    rows = []
    for ln in lines:
        c = ln.split(";")
        if len(c) >= 1 + n_uhrh:
            rows.append([float(c[1 + k]) for k in range(n_uhrh)])
    return np.array(rows)


def build_psoil(proj, ids):
    """Construit le dict psoil vectorisé (tenseurs (U,)) depuis les fichiers."""
    U = len(ids)
    g = lambda f: torch.tensor([f(u) for u in ids])
    tex = [proj["sol"][proj["texture"][u]] for u in ids]
    thetas = torch.tensor([t["thetas"] for t in tex])
    ks = torch.tensor([t["ks"] for t in tex])
    psis = torch.tensor([t["psis"] for t in tex])
    lam = torch.tensor([t["lam"] for t in tex])
    b = 1.0 / lam
    omegpi = (1.0 + 2.0 * b) / (2.0 + 2.0 * b)
    A = omegpi
    psi_i = psis * A.pow(-b)
    dpsi_i = -psis * b * A.pow(-b - 1.0)
    r = psi_i / dpsi_i
    nn = (A * A - A - 2.0 * r * A + r) / (A - 1.0 - r)
    mm = -dpsi_i / (2.0 * A - nn - 1.0)
    frac = [uhrh_fractions(proj, u) for u in ids]
    fsa = torch.tensor([f[0] for f in frac]); fse = torch.tensor([f[1] for f in frac])
    fsi = torch.tensor([f[2] for f in frac])
    bv = proj["bv3c"]
    p = dict(slope=g(lambda u: proj["uhrh"][u]["slope"]),
             krec=g(lambda u: bv[u]["krec"]), cin=g(lambda u: bv[u]["cin"]),
             coef_recharge=g(lambda u: bv[u]["recharge"]),
             fsa=fsa, fse=fse, fsi=fsi)
    for i in (1, 2, 3):
        p[f"z{i}"] = g(lambda u, i=i: bv[u][f"z{i}"])
        p[f"thetas{i}"] = thetas.clone(); p[f"ks{i}"] = ks.clone(); p[f"psis{i}"] = psis.clone()
        p[f"b{i}"] = b.clone(); p[f"omegpi{i}"] = omegpi.clone(); p[f"mm{i}"] = mm.clone(); p[f"nn{i}"] = nn.clone()
    return p


def main():
    proj = load_project(DEL)
    ids = proj["uhrh_ids"]; U = len(ids)
    print(f"{U} UHRH, params lus des fichiers")

    apC = read_cpp("apport", U); etpC = read_cpp("etp", U)
    e1C, e2C, e3C = read_cpp("etr1", U), read_cpp("etr2", U), read_cpp("etr3", U)
    psC, phC, pbC = read_cpp("production_surf", U), read_cpp("production_hypo", U), read_cpp("production_base", U)
    th1C, th2C, th3C = read_cpp("theta1", U), read_cpp("theta2", U), read_cpp("theta3", U)
    couv = read_cpp("couvert_nival", U)
    NT = min(map(len, [apC, etpC, e1C, psC, th1C, couv]))
    print(f"{NT} jours C++ chargés")

    p = build_psoil(proj, ids)
    soil = BV3C2Clone(n_substep=1500)
    t1 = torch.tensor(th1C[0]); t2 = torch.tensor(th2C[0]); t3 = torch.tensor(th3C[0])
    T = lambda a, i: torch.tensor(a[i])
    z = torch.zeros(U)
    ps_s = np.zeros((NT, U)); ph_s = np.zeros((NT, U)); pb_s = np.zeros((NT, U))
    t1_s = np.zeros((NT, U)); t2_s = np.zeros((NT, U)); t3_s = np.zeros((NT, U))
    t1_s[0], t2_s[0], t3_s[0] = th1C[0], th2C[0], th3C[0]
    with torch.no_grad():
        for i in range(1, NT):
            ps, ph, pb, rech, (t1, t2, t3), _ = soil(
                t1, t2, t3, T(apC, i), T(etpC, i), z, T(couv, i), p,
                etr1_mm=T(e1C, i), etr2_mm=T(e2C, i), etr3_mm=T(e3C, i))
            ps_s[i] = ps.numpy(); ph_s[i] = ph.numpy(); pb_s[i] = pb.numpy()
            t1_s[i] = t1.numpy(); t2_s[i] = t2.numpy(); t3_s[i] = t3.numpy()

    sl = slice(1, NT)
    def rmse_per(a, b): return np.sqrt(np.nanmean((a[sl] - b[sl]) ** 2, axis=0))   # (U,)
    rmse_ps = rmse_per(ps_s, psC); rmse_t1 = rmse_per(t1_s, th1C)
    rmse_ph = rmse_per(ph_s, phC); rmse_t2 = rmse_per(t2_s, th2C); rmse_t3 = rmse_per(t3_s, th3C)

    # bilans cumulés bassin (moyenne UHRH)
    def cum(a): return a[sl].sum(axis=0).mean()
    print(f"\n=== BILAN CUMULÉ moyen sur {U} UHRH (mm) clone | C++ ===")
    print(f"  prod_surf : {cum(ps_s):8.1f} | {cum(psC):8.1f}")
    print(f"  prod_hypo : {cum(ph_s):8.1f} | {cum(phC):8.1f}")
    print(f"  prod_base : {cum(pb_s):8.1f} | {cum(pbC):8.1f}")
    print(f"\n=== RMSE par UHRH (médiane / p95 / max sur {U} UHRH) ===")
    for nm, r in [("prod_surf", rmse_ps), ("prod_hypo", rmse_ph),
                  ("theta1", rmse_t1), ("theta2", rmse_t2), ("theta3", rmse_t3)]:
        print(f"  {nm:10s} méd {np.median(r):.4f}  p95 {np.percentile(r,95):.4f}  max {np.max(r):.4f}")

    # pires UHRH sur theta1 + leur texture/occupation
    worst = np.argsort(rmse_t1)[::-1][:8]
    print(f"\n=== 8 pires UHRH (RMSE theta1) ===")
    print(f"{'uhrh':>5} {'rmse_t1':>8} {'rmse_pS':>8} {'texture':>10} {'fsa':>5} {'fse':>5} {'fsi':>5}")
    for k in worst:
        u = ids[k]; f = uhrh_fractions(proj, u)
        tx = proj["sol"][proj["texture"][u]]["name"]
        print(f"{u:>5} {rmse_t1[k]:8.4f} {rmse_ps[k]:8.4f} {tx:>10} {f[0]:5.2f} {f[1]:5.2f} {f[2]:5.2f}")

    # par texture
    print(f"\n=== RMSE theta1 médian par texture ===")
    txa = np.array([proj["sol"][proj["texture"][u]]["name"] for u in ids])
    for t in sorted(set(txa)):
        m = txa == t
        print(f"  {t:10s} n={m.sum():3d}  rmse_t1 méd {np.median(rmse_t1[m]):.4f}  rmse_pS méd {np.median(rmse_ps[m]):.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
