"""Port SCALAIRE ligne-à-ligne de CalculeUHRH/TriCoucheOct97/CalculeRuisselement
(bv3c2.cpp) pour UN uhrh, UN jour. But : isoler si l'algorithme fidèle reproduit
la production C++ d'un jour donné, en partant du theta C++ de la veille + forçage
C++ du jour. Sonde de débogage, jetable."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import hydrotel_clone.validate_soil_all_uhrh as V
from hydrotel_clone.hydrotel_params import load_project

PAS = 24.0
FDTCMIN = PAS / (24.0 * 60.0 * 60.0 * 1000.0)


def calc_k(theta, thetas, ks, b):
    omega = max(theta / thetas, 0.05)
    return ks * omega ** (2.0 * b + 3.0)


def calc_psi(theta, thetas, psis, b, omegpi, mm, nn):
    omega = max(theta / thetas, 0.05)
    if omega < omegpi:
        return psis * omega ** (-b)
    return -mm * (omega - nn) * (omega - 1.0)


def calc_ruis(apport, theta1, thetas1, ks1):
    """CalculeRuisselement l.2175. gel=0. Retourne pinf, ruis (m/h)."""
    if apport > 0.0:
        prec = apport / (1000.0 * PAS)
        if theta1 == thetas1:
            pinf = 0.0
        elif prec > ks1:
            pinf = ks1
        else:
            pinf = prec
        return pinf, prec - pinf
    return 0.0, 0.0


def tricouche(P, th, pinf, ruis, dtc_in, etr):
    """TriCoucheOct97 l.1841, gel=0. P params, th=[t1,t2,t3], retourne ruis,q2,q3,dtc,th."""
    pte = P["slope"]; z11, z22, z33 = P["z1"], P["z2"], P["z3"]
    krec = P["krec"]; cin = P["cin"]
    theta1, theta2, theta3 = th
    k1 = calc_k(theta1, P["thetas1"], P["ks1"], P["b1"])
    k2 = calc_k(theta2, P["thetas2"], P["ks2"], P["b2"])
    k3 = calc_k(theta3, P["thetas3"], P["ks3"], P["b3"])
    psi1 = calc_psi(theta1, P["thetas1"], P["psis1"], P["b1"], P["omegpi1"], P["mm1"], P["nn1"])
    psi2 = calc_psi(theta2, P["thetas2"], P["psis2"], P["b2"], P["omegpi2"], P["mm2"], P["nn2"])
    psi3 = calc_psi(theta3, P["thetas3"], P["psis3"], P["b3"], P["omegpi3"], P["mm3"], P["nn3"])
    k12 = max(k1, k2); k23 = max(k2, k3)
    qq12 = k12 * (2.0 * (psi2 - psi1) / (z11 + z22) + 1.0)
    qq23 = k23 * (2.0 * (psi3 - psi2) / (z22 + z33) + 1.0)
    q2 = k2 * math.sin(math.atan(pte)) * z22
    q3 = krec * z33 * theta3
    v_etr1, v_etr2, v_etr3 = [e / 1000.0 / PAS for e in etr]
    dtc = dtc_in
    if pinf > 0.0 and dtc > 1.0:
        dtc = 1.0
    q12z = qq12 / z11; q23z = qq23 / z22; q2s = q2 / z22
    # ── dtc selection (l.1954-2034) ──
    dtcTemp = dVal1 = dVal2 = 0.0
    bDtcMod = False
    if (abs(q12z * dtc) >= cin * theta1) or (abs((q23z + q2s) * dtc) >= cin * theta2):
        if theta1 != 0.0 and q12z != 0.0:
            dVal1 = cin * theta1 / abs(q12z)
        if theta2 != 0.0 and (q23z + q2s) != 0.0:
            dVal2 = cin * theta2 / abs(q23z + q2s)
        if dVal1 != 0.0 and dVal2 != 0.0:
            dtcTemp = min(dVal1, dVal2)
        else:
            dtcTemp = dVal1 if dVal1 != 0.0 else dVal2
        if dtcTemp != 0.0:
            if dtcTemp < FDTCMIN:
                dtcTemp = FDTCMIN
            iVal = int(PAS / dtcTemp)
            if iVal % 2 == 0:
                dtcTemp = PAS / (iVal + 2.0)
            else:
                dtcTemp = PAS / (iVal + 1.0)
            dtcTemp = min(dtc, dtcTemp)
            bDtcMod = True
    if (abs(q12z * dtc) >= cin * theta1) or (abs((q23z + q2s) * dtc) >= cin * theta2):
        dtc = PAS / 48.0
        if (abs(q12z * dtc) >= cin * theta1) or (abs((q23z + q2s) * dtc) >= cin * theta2):
            dtc = PAS / 288.0
            if (abs(q12z * dtc) >= cin * theta1) or (abs((q23z + q2s) * dtc) >= cin * theta2):
                dtc = PAS / 1152.0
    if bDtcMod:
        dtc = min(dtc, dtcTemp)
    # ── integration (l.2036-2038) ──
    theta1 += dtc * (pinf - qq12 - v_etr1) / z11
    theta2 += (dtc / z22) * (qq12 - qq23 - v_etr2 - q2)
    theta3 += (dtc / z33) * (qq23 - q3 - v_etr3)
    # ── saturation cascade (l.2046-2116) ──
    s1, s2, s3 = P["thetas1"], P["thetas2"], P["thetas3"]
    if theta3 > s3 or theta2 > s2 or theta1 > s1:
        if theta3 > s3:
            sur = (theta3 - s3) * z33; qq23 -= sur / dtc; theta2 += sur / z22; theta3 = s3
        if theta2 > s2 and theta3 < s3:
            sur = (theta2 - s2) * z22; qq23 += sur / dtc; theta3 += sur / z33; theta2 = s2
            if theta3 > s3:
                sur = (theta3 - s3) * z33; qq23 -= sur / dtc; theta2 += sur / z22; theta3 = s3
        if theta2 > s2:
            sur = (theta2 - s2) * z22; qq12 -= sur / dtc; theta1 += sur / z11; theta2 = s2
        if theta1 > s1 and (theta2 < s2 or theta3 < s3):
            sur = (theta1 - s1) * z11; qq12 += sur / dtc; theta2 += sur / z22; theta1 = s1
            if theta2 > s2 and theta3 < s3:
                sur = (theta2 - s2) * z22; qq23 += sur / dtc; theta3 += sur / z33; theta2 = s2
                if theta3 > s3:
                    sur = (theta3 - s3) * z33; qq23 -= sur / dtc; theta2 += sur / z22; theta3 = s3
            if theta2 > s2:
                sur = (theta2 - s2) * z22; qq12 -= sur / dtc; theta1 += sur / z11; theta2 = s2
        if theta1 > s1:
            ruis += (theta1 - s1) * z11 / dtc; theta1 = s1
    # ── negativity (l.2118-2158) ──
    if theta1 < 0.0:
        qq12 += theta1 * z11 / dtc; theta2 += theta1 * z11 / z22; theta1 = 0.0
    if theta2 < 0.0:
        qq23 += theta2 * z22 / dtc; theta3 += theta2 * z22 / z33; theta2 = 0.0
    if theta3 < 0.0:
        if abs(z33 * theta3) < z22 * theta2:
            qq23 -= theta3 * z33 / dtc; theta2 += theta3 * z33 / z22; theta3 = 0.0
        elif abs(z33 * theta3) < z22 * theta2 + z11 * theta1:
            theta1 += theta2 * z22 / z11 + theta3 * z33 / z11; theta2 = 0.0; theta3 = 0.0
        else:
            theta1 = theta2 = theta3 = 0.0
    return ruis, q2, q3, dtc, [theta1, theta2, theta3]


def calc_uhrh(P, th, apport, etr, verbose=False):
    """CalculeUHRH l.751, gel=0. Retourne prod_surf/hypo/base (mm), th_new."""
    lprec = leau = lruis = lhyp = lbase = 0.0
    tr = PAS
    nit = 0
    while tr > 0.0:
        pinf, ruis = calc_ruis(apport, th[0], P["thetas1"], P["ks1"])
        dtc = tr
        ruis, q2, q3, dtc, th = tricouche(P, th, pinf, ruis, dtc, etr)
        tr -= dtc
        prec = apport / (1000.0 * PAS)
        pres = max(0.0, prec)  # fse=0 ici (sol pur) → leau non utilisé
        lruis += ruis * dtc; lprec += prec * dtc; leau += pres * dtc
        lhyp += q2 * dtc; lbase += q3 * dtc
        nit += 1
        if verbose and nit <= 6:
            print(f"   it{nit:3d} dtc={dtc:.5f} pinf={pinf*1000*PAS:.3f} ruis={ruis*1000*PAS:.4f} "
                  f"t1={th[0]:.5f} t2={th[1]:.5f} t3={th[2]:.5f} lruis={lruis*1000:.3f}")
        if nit > 5000:
            print("   !! cap 5000 it, tr=", tr); break
    fsa, fse, fsi = P["fsa"], P["fse"], P["fsi"]
    prod_surf = lruis * fsa + leau * fse + lprec * fsi
    prod_hypo = lhyp * fsa
    prod_base = lbase * fsa
    cr = P["coef_recharge"]
    prod_hypo -= prod_hypo * cr; prod_base -= prod_base * cr
    return prod_surf * 1000, prod_hypo * 1000, prod_base * 1000, th, nit


def main():
    proj = load_project(V.DEL)
    ids = proj["uhrh_ids"]; U = len(ids); k = ids.index(376)
    import torch
    torch.set_default_dtype(torch.float64)
    pv = V.build_psoil(proj, ids)
    P = {key: (float(val[k]) if hasattr(val, "__len__") else float(val)) for key, val in pv.items()}
    g = lambda n: V.read_cpp(n, U)
    apC, etpC = g("apport"), g("etp")
    e1C, e2C, e3C = g("etr1"), g("etr2"), g("etr3")
    th1C, th2C, th3C = g("theta1"), g("theta2"), g("theta3")
    psC, phC, pbC = g("production_surf"), g("production_hypo"), g("production_base")

    print(f"UHRH376 params: z={P['z1']}/{P['z2']}/{P['z3']} thetas={P['thetas1']:.3f} "
          f"ks={P['ks1']:.4f} cin={P['cin']} krec={P['krec']:.2e} slope={P['slope']:.5f} "
          f"fsa={P['fsa']:.3f} fse={P['fse']:.3f} fsi={P['fsi']:.3f}")
    for day in [68, 72, 30]:
        th0 = [th1C[day - 1, k], th2C[day - 1, k], th3C[day - 1, k]]
        etr = [e1C[day, k], e2C[day, k], e3C[day, k]]
        ps, ph, pb, thn, nit = calc_uhrh(P, list(th0), apC[day, k], etr, verbose=(day == 68))
        print(f"\nDAY {day}: apport={apC[day,k]:.2f}  th_start={th0[0]:.4f}/{th0[1]:.4f}/{th0[2]:.4f}  ({nit} it)")
        print(f"  SCALAR  prod_surf={ps:.3f} hypo={ph:.3f} base={pb:.3f}  th_end={thn[0]:.4f}/{thn[1]:.4f}/{thn[2]:.4f}")
        print(f"  C++     prod_surf={psC[day,k]:.3f} hypo={phC[day,k]:.3f} base={pbC[day,k]:.3f}  th_end={th1C[day,k]:.4f}/{th2C[day,k]:.4f}/{th3C[day,k]:.4f}")


if __name__ == "__main__":
    main()
