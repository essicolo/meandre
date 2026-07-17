"""Charge les paramètres de sol CALIBRÉS d'Hydrotel (projet PHYSITEL) et les
agrège UHRH→troncon, pour ANCRER la colonne fidèle sur la calibration existante.

OBJECTIF (temporaire) : reproduire Hydrotel — la colonne BV3C2 a été validée à la
décimale quand on la nourrit de ces params (bv3c.csv + proprietehydrolique.sol).
Ici on les fournit au modèle méandre comme point de départ, AU LIEU de l'init
littérature générique du NeRF. C'est un PRIOR OPTIONNEL (un flag), pas une
dépendance permanente : l'objectif ultime reste de découpler méandre de
Hydrotel/PHYSITEL (params appris depuis les features, NeRF). Voir
[[project_meandre_reproduce_then_modernize]].

Source : hydrotel_clone.hydrotel_params (lit uhrh.csv, type_sol.cla,
proprietehydrolique.sol, bv3c.csv, occupation). Agrégation = moyenne pondérée par
l'aire UHRH (cohérente avec _build_territorial), par troncon dans l'ordre node_ids.
"""
from __future__ import annotations
from pathlib import Path
import torch

from hydrotel_clone.hydrotel_params import load_project, uhrh_fractions
from meandre.data.physitel_loader import _parse_troncon


def _spline_coeffs(b, psis):
    """omegpi/mm/nn du raccord C1 de psi (identique à BV3C2/make_params)."""
    omegpi = (1.0 + 2.0 * b) / (2.0 + 2.0 * b)
    A = omegpi
    psi_i = psis * A.pow(-b)
    dpsi_i = -psis * b * A.pow(-b - 1.0)
    r = psi_i / dpsi_i
    nn = (A * A - A - 2.0 * r * A + r) / (A - 1.0 - r)
    mm = -dpsi_i / (2.0 * A - nn - 1.0)
    return omegpi, mm, nn


def load_calibrated_soil(project_dir, node_ids, z1_fixed,
                         sim_subdir="simulation/simulation",
                         device="cpu", dtype=torch.float64) -> dict:
    """Retourne le dict p_soil par NŒUD (troncon) attendu par HydrotelColumn :
    z1/z2/z3, thetas1/2/3, ks1/2/3, b/psis/omegpi/mm/nn 1/2/3, krec, slope, cin,
    fsa/fse/fsi, coef_recharge — calibrés Hydrotel, agrégés UHRH→troncon.

    z1_fixed : épaisseur couche 1 du modèle (self.z1) ; on garde z2/z3 d'Hydrotel
    mais z1 reste la valeur méandre (la colonne empile z1+z2+z3). NB : bv3c.csv
    donne z1/z2/z3 d'Hydrotel ; on utilise z2_h+z3_h et z1_fixed pour cohérence
    avec la colonne (z1 fixe, z2/z3 = profondeurs Hydrotel)."""
    proj = load_project(str(project_dir), sim_subdir)
    troncons = _parse_troncon(Path(project_dir) / "physitel" / "troncon.trl")
    t2u = {t["id"]: t["uhrh_ids"] for t in troncons}
    uhrh, sol, tex, bv = proj["uhrh"], proj["sol"], proj["texture"], proj["bv3c"]

    def up(u):
        tx = sol[tex[u]]; b = bv[u]
        fsa, fse, fsi, _ = uhrh_fractions(proj, u)
        return dict(thetas=tx["thetas"], ks=tx["ks"], psis=tx["psis"], lam=tx["lam"],
                    z1=b["z1"], z2=b["z2"], z3=b["z3"], krec=b["krec"], cin=b["cin"],
                    recharge=b["recharge"], slope=uhrh[u]["slope"],
                    fsa=fsa, fse=fse, fsi=fsi, area=max(uhrh[u]["area_km2"], 1e-9))

    keys = ("thetas", "ks", "psis", "lam", "z1", "z2", "z3", "krec", "cin", "recharge",
            "slope", "fsa", "fse", "fsi")
    cols = {k: [] for k in keys}
    n_missing = 0
    for tid in node_ids:
        uids = [u for u in t2u.get(int(tid), []) if u in tex and u in bv]
        ps = [up(u) for u in uids]
        if not ps:
            n_missing += 1
            # défaut neutre (loam) si troncon sans UHRH calibré
            cols["thetas"].append(0.434); cols["ks"].append(0.0132); cols["psis"].append(0.40)
            cols["lam"].append(0.252); cols["z1"].append(0.21941); cols["z2"].append(0.15725); cols["z3"].append(2.65)
            cols["krec"].append(1.2869e-7); cols["cin"].append(0.03); cols["recharge"].append(0.0)
            cols["slope"].append(0.04); cols["fsa"].append(1.0); cols["fse"].append(0.0); cols["fsi"].append(0.0)
            continue
        w = torch.tensor([p["area"] for p in ps]); w = w / w.sum()
        for k in keys:
            cols[k].append(float((w * torch.tensor([p[k] for p in ps])).sum()))
    if n_missing:
        print(f"[hydrotel_calib] {n_missing}/{len(node_ids)} troncons sans UHRH calibre -> defaut loam")

    T = lambda k: torch.tensor(cols[k], dtype=dtype, device=device)
    thetas, ks, psis, lam = T("thetas"), T("ks"), T("psis"), T("lam")
    b = 1.0 / lam
    omegpi, mm, nn = _spline_coeffs(b, psis)
    p = dict(z1=T("z1"), z2=T("z2"), z3=T("z3"),   # z CALIBRÉS Hydrotel (pas z1_fixed)
             slope=torch.clamp(T("slope"), min=1e-4), krec=T("krec"), cin=T("cin"),
             fsa=T("fsa"), fse=T("fse"), fsi=T("fsi"), coef_recharge=T("recharge"))
    for i in (1, 2, 3):
        p[f"thetas{i}"] = thetas.clone(); p[f"ks{i}"] = ks.clone(); p[f"psis{i}"] = psis.clone()
        p[f"b{i}"] = b.clone(); p[f"omegpi{i}"] = omegpi.clone()
        p[f"mm{i}"] = mm.clone(); p[f"nn{i}"] = nn.clone()
    return p


def load_linacre_nodes(project_dir, node_ids, sim_subdir="simulation/simulation",
                       device="cpu", dtype=None):
    """Params Linacre par NŒUD (troncon) : lat/alti (uhrh.csv) + linacre.csv
    (t_froid, t_chaud, albedo, COEFF MULTIPLICATIF OPTIMISATION = calage régional
    d'ETP des plateformes LN24HA), agrégés UHRH→troncon pondérés par superficie."""
    from pathlib import Path
    dtype = dtype or torch.get_default_dtype()
    proj = load_project(str(project_dir), sim_subdir)
    troncons = _parse_troncon(Path(project_dir) / "physitel" / "troncon.trl")
    t2u = {t["id"]: t["uhrh_ids"] for t in troncons}
    uhrh = proj["uhrh"]
    lin = {}
    for ln in open(f"{project_dir}/{sim_subdir}/linacre.csv", encoding="latin-1").read().splitlines():
        c = ln.split(";")
        if len(c) >= 5 and c[0].strip().isdigit():
            lin[int(c[0])] = [float(x) for x in c[1:5]]
    cols = {k: [] for k in ("lat", "alti", "tf", "tc", "alb", "coeff")}
    n_missing = 0
    for tid in node_ids:
        uids = [u for u in t2u.get(int(tid), []) if u in uhrh and u in lin]
        if not uids:
            n_missing += 1
            cols["lat"].append(46.0); cols["alti"].append(200.0); cols["tf"].append(-10.0)
            cols["tc"].append(20.0); cols["alb"].append(0.23); cols["coeff"].append(0.45)
            continue
        w = torch.tensor([max(uhrh[u]["area_km2"], 1e-9) for u in uids]); w = w / w.sum()
        agg = lambda vals: float((w * torch.tensor(vals)).sum())
        cols["lat"].append(agg([uhrh[u]["lat"] for u in uids]))
        cols["alti"].append(agg([uhrh[u]["altitude"] for u in uids]))
        cols["tf"].append(agg([lin[u][0] for u in uids]))
        cols["tc"].append(agg([lin[u][1] for u in uids]))
        cols["alb"].append(agg([lin[u][2] for u in uids]))
        cols["coeff"].append(agg([lin[u][3] for u in uids]))
    if n_missing:
        print(f"[linacre] {n_missing}/{len(node_ids)} troncons sans UHRH -> défauts")
    T = lambda k: torch.tensor(cols[k], dtype=dtype, device=device)
    return T("lat"), T("alti"), T("tf"), T("tc"), T("alb"), T("coeff")
