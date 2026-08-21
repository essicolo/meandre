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


def imposed_retention_curve(cs: dict, use_aquifer: bool) -> dict:
    """Sous-ensemble du sol calibré à IMPOSER au modèle (mode d'ancrage sauf_ks).

    Tout est imposé SAUF les conductivités et porosités, qui restent au champ appris
    (loi des ancrages : ancrer les processus scalaires, jamais figer le sol en bloc).

    ET, quand l'aquifère restituant est actif, SAUF krec et coef_recharge. Le calage
    Hydrotel porte une recharge quasi nulle (~1e-7 m/h) parce que chez lui c'est une
    fuite jamais restituée, que son calage étrangle. L'imposer à notre aquifère revient
    à brancher le réservoir en aval d'un robinet fermé : la recharge tombe à zéro et
    toute variation de krec devient sans effet.

    Cette fonction existe pour que le PILOTE d'entraînement et les DIAGNOSTICS ne
    puissent plus diverger sur ce point. Ils avaient divergé (2026-08-19) : le balayage
    recharge x vidange mesurait un champion à 0.7591 au lieu de 0.7880, avec une
    recharge nulle partout et un krec inopérant, parce que le diagnostic imposait krec
    et le pilote non. Sixième occurrence de « la recette d'exécution ne se déduit pas
    du point de reprise ».
    """
    exclus_prefixes = ("ks", "thetas")
    exclus_cles = ("krec", "coef_recharge") if use_aquifer else ()
    return {k: v for k, v in cs.items()
            if not k.startswith(exclus_prefixes) and k not in exclus_cles}


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


def load_melt_nodes(project_dir, node_ids, sim_subdir="simulation/simulation",
                    device="cpu", dtype=None):
    """Params fonte RÉGIONAUX par nœud depuis degre_jour_modifie.csv (calage
    plateforme) : seuils de fonte par classe (C), taux de fonte par classe
    (mm/j/C), taux fonte géothermique, densité max, constante tassement.
    Agrégés UHRH→troncon pondérés par superficie."""
    from pathlib import Path
    dtype = dtype or torch.get_default_dtype()
    proj = load_project(str(project_dir), sim_subdir)
    troncons = _parse_troncon(Path(project_dir) / "physitel" / "troncon.trl")
    t2u = {t["id"]: t["uhrh_ids"] for t in troncons}
    uhrh = proj["uhrh"]
    dj = {}
    for ln in open(f"{project_dir}/{sim_subdir}/degre_jour_modifie.csv", encoding="latin-1").read().splitlines():
        c = ln.split(";")
        if len(c) >= 11 and c[0].strip().isdigit():
            dj[int(c[0])] = [float(x) for x in c[1:11]]
    # colonnes : 0 taux_geo, 1 densite_max, 2 tassement, 3-5 seuils c/f/d, 6-8 taux c/f/d, 9 seuil albedo
    names = ["taux_geo", "dens_max", "tasse", "seuil_c", "seuil_f", "seuil_d",
             "taux_c", "taux_f", "taux_d", "seuil_alb"]
    cols = {k: [] for k in names}
    n_missing = 0
    for tid in node_ids:
        uids = [u for u in t2u.get(int(tid), []) if u in uhrh and u in dj]
        if not uids:
            n_missing += 1
            for k, v in zip(names, [0.5, 466.0, 0.1, 0.0, 0.0, 0.0, 12.0, 14.0, 16.0, 1.0]):
                cols[k].append(v)
            continue
        w = torch.tensor([max(uhrh[u]["area_km2"], 1e-9) for u in uids]); w = w / w.sum()
        for j, k in enumerate(names):
            cols[k].append(float((w * torch.tensor([dj[u][j] for u in uids])).sum()))
    if n_missing:
        print(f"[melt] {n_missing}/{len(node_ids)} troncons sans UHRH -> défauts")
    return {k: torch.tensor(cols[k], dtype=dtype, device=device) for k in names}


def load_passage_pluie_neige(project_dir, defaut: float = 0.0) -> float:
    """Seuil de PARTITION PLUIE/NEIGE calibré du projet Hydrotel (thiessen.csv,
    colonne PASSAGE PLUIE NEIGE, °C). Uniforme par région dans les projets du Québec
    (OUTV/SAGU/SLNO -2.2168, GASP -3.0672).

    BIFURCATION TROUVÉE LE 2026-08-10 : méandre codait 0 °C en dur pour toutes les
    régions. Avec un seuil à 0 au lieu de -2.2, toute précipitation entre les deux
    tombe en PLUIE chez méandre et en NEIGE chez Hydrotel — d'où un excès d'écoulement
    en février-mars (rapports 1.33 et 1.43 contre Hydrotel) et une crue de fonte
    amputée en avril-mai (0.89 et 0.83), signature mesurée sur OUTV à intrants
    identiques et paramètres figés.
    """
    from pathlib import Path
    p = Path(project_dir) / "simulation" / "simulation" / "thiessen.csv"
    if not p.exists():
        return defaut
    import pandas as _pd
    import numpy as _np
    d = _pd.read_csv(p, sep=";", skiprows=4)
    d.columns = [c.strip() for c in d.columns]
    col = [c for c in d.columns if "PASSAGE" in c.upper()]
    if not col:
        return defaut
    return float(_np.median(d[col[0]].values))


def load_occupation_sol(project_dir, node_ids, device="cpu"):
    """OCCUPATION DU SOL brute de PHYSITEL (physitel/occupation_sol.cla), agrégée par
    tronçon au prorata des aires d'UHRH, en FRACTIONS (somme = 1).

    BIFURCATION TROUVÉE LE 2026-08-10, la plus lourde de la série. Les colonnes
    d'occupation de la base du Québec sont CENTRÉES-RÉDUITES (f_forest va de -3.67 à
    +1.27, moyenne nulle) et les colonnes brutes `f_*_raw` n'ont jamais été écrites par
    le constructeur de régions. `get_physical("f_forest_raw")` renvoie donc None et la
    colonne retombe sur son défaut 0.0 : méandre simulait l'Outaouais comme 100 % de sol
    nu DÉCOUVERT, sans forêt, sans eau libre, sans imperméable — là où Hydrotel a 67.7 %
    de forêt (35.8 feuillus + 31.9 conifères) et 9.4 % d'eau. Or le découvert est la
    classe de neige qui fond le plus vite : Hydrotel y porte 52 mm d'équivalent en eau
    contre 136 sous conifères. D'où un manteau deux fois trop maigre, une fonte
    d'un mois trop précoce (rapports 1.37 et 1.47 contre Hydrotel en février-mars) et
    une crue de printemps amputée (0.86 et 0.79 en avril-mai).

    Retourne un dict de tenseurs (n_nodes,) prêt pour ``HydrotelColumn.set_land_cover``.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from meandre.data.physitel_loader import _parse_troncon

    proj = Path(project_dir)
    lignes = [l.split() for l in
              (proj / "physitel" / "occupation_sol.cla").read_text(encoding="latin-1").splitlines()
              if l.strip()]
    noms = [x.strip('"') for x in lignes[2][1:]]
    ids = np.array([int(r[0]) for r in lignes[3:]])
    A = np.array([[float(x) for x in r[1:]] for r in lignes[3:]], dtype=np.float64)
    tot = np.clip(A.sum(axis=1, keepdims=True), 1e-9, None)
    frac = A / tot                                   # fraction de pixels par UHRH
    par_uhrh = {int(i): frac[k] for k, i in enumerate(ids)}

    def col(motifs):
        return [k for k, nm in enumerate(noms) if any(m in nm.lower() for m in motifs)]
    idx = {
        "f_water_raw": col(["eau"]),
        "f_urban_raw": col(["impermeable"]),
        "f_wetland_raw": col(["tourbiere", "milieu_humide"]),
        "f_forest_conifer_raw": col(["conifere"]),
        "f_forest_deciduous_raw": col(["feuillu"]),
        "f_agriculture_raw": col(["agricole"]),
        "f_bare_raw": col(["sol_nu"]),
    }
    tr = {t["id"]: t for t in _parse_troncon(proj / "physitel" / "troncon.trl")}
    uh = pd.read_csv(proj / "physitel" / "uhrh.csv", sep=";", skiprows=1)
    col_aire = next((c for c in uh.columns if "superficie" in c.lower() or "aire" in c.lower()), None)
    aires = dict(zip(uh[uh.columns[0]].astype(int), uh[col_aire].astype(float)))

    n = len(node_ids)
    out = {k: np.zeros(n, dtype=np.float32) for k in idx}
    for j, nid in enumerate(node_ids):
        t = tr.get(int(nid))
        if t is None:
            continue
        acc = np.zeros(len(noms)); wtot = 0.0
        for uid in t["uhrh_ids"]:
            f = par_uhrh.get(abs(int(uid)))
            if f is None:
                continue
            w = aires.get(abs(int(uid)), 1.0)
            acc += w * f; wtot += w
        if wtot <= 0:
            continue
        acc /= wtot
        for k, cols in idx.items():
            out[k][j] = float(acc[cols].sum()) if cols else 0.0
    res = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in out.items()}
    res["f_forest_raw"] = res["f_forest_conifer_raw"] + res["f_forest_deciduous_raw"]
    res["f_forest_mixed_raw"] = torch.zeros_like(res["f_forest_raw"])
    return res


def load_milieux_humides(project_dir, node_ids, device="cpu"):
    """MILIEUX HUMIDES ISOLÉS du projet Hydrotel
    (simulation/simulation/milieux_humides_isoles.csv), agrégés par tronçon.

    BUG TROUVÉ LE 2026-08-10 : `_wetland_from_territorial` renvoie None dès que
    `wet_a_raw` manque, et cette colonne n'existe dans AUCUN cache du Québec. Le module
    de milieu humide n'a donc JAMAIS été instancié, sans le moindre message : aucun
    laminage nulle part, pour 7.6 % de superficie humide moyenne.

    Aires sommées sur les UHRH du tronçon, paramètres moyennés au prorata des aires.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import torch
    from meandre.data.physitel_loader import _parse_troncon

    proj = Path(project_dir)
    f = proj / "simulation" / "simulation" / "milieux_humides_isoles.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, sep=";")
    d.columns = [c.strip() for c in d.columns]
    col = {c.split("(")[0].strip().lower(): c for c in d.columns}
    uid = d[col["uhrhid"]].astype(int).values
    par = {}
    for k, nm in (("wet_a", "wet_a"), ("wet_dra_fr", "wet_dra_fr"), ("frac", "frac"),
                  ("wetdnor", "wetdnor"), ("wetdmax", "wetdmax"), ("ksat_bs", "ksat_bs"),
                  ("c_ev", "c_ev"), ("c_prod", "c_prod")):
        if nm in col:
            par[k] = dict(zip(uid, pd.to_numeric(d[col[nm]], errors="coerce").fillna(0.0).values))
    if "wet_a" not in par:
        return {}

    tr = {t["id"]: t for t in _parse_troncon(proj / "physitel" / "troncon.trl")}
    n = len(node_ids)
    out = {k: np.zeros(n, dtype=np.float32) for k in par}
    for j, nid in enumerate(node_ids):
        t = tr.get(int(nid))
        if t is None:
            continue
        ids = [abs(int(u)) for u in t["uhrh_ids"] if abs(int(u)) in par["wet_a"]]
        if not ids:
            continue
        a = np.array([par["wet_a"][u] for u in ids], dtype=np.float64)
        out["wet_a"][j] = float(a.sum())          # les AIRES s'additionnent
        w = a if a.sum() > 0 else np.ones_like(a)  # les PARAMÈTRES se moyennent
        for k in par:
            if k == "wet_a":
                continue
            v = np.array([par[k][u] for u in ids], dtype=np.float64)
            out[k][j] = float((v * w).sum() / w.sum())
    return {f"{k}_raw": torch.tensor(v, dtype=torch.float32, device=device)
            for k, v in out.items()}


def load_phenologie(project_dir):
    """PHÉNOLOGIE du projet Hydrotel : indice foliaire (physio/ind_fol.def) et
    profondeur racinaire (physio/pro_rac.def), par classe d'occupation et par jour.

    BUG TROUVÉ LE 2026-08-10 : méandre code ces profils EN DUR (_LEAF/_ROOT, repris du
    bassin DELISLE) alors qu'Hydrotel les lit dans le projet, et les écarts sont
    importants sur OUTV : conifères 1.531 m de racines contre 1.0 chez nous, milieux
    humides 1.531 contre 0.75, agriculture 0.108 contre jusqu'à 0.8 ; feuillus à indice
    foliaire NUL en hiver et culminant à 6, contre un plancher à 3 chez nous. Des
    racines trop courtes prélèvent moins d'eau : moins d'ETR, plus d'écoulement, ce qui
    correspond à l'excès d'été mesuré sur les deux régions (1.25 à 1.39).

    Retourne {classe_meandre: (jours, indice_foliaire, profondeur_racinaire)}.
    """
    from pathlib import Path
    proj = Path(project_dir) / "physio"

    def lire(f):
        li = [l for l in (proj / f).read_text(encoding="latin-1").splitlines() if l.strip()]
        noms = li[3].split()[1:]
        jours, vals = [], []
        for l in li[4:]:
            t = l.split()
            if not t[0].lstrip("-").isdigit():
                continue
            jours.append(int(t[0])); vals.append([float(x) for x in t[1:]])
        return noms, jours, vals

    try:
        nf, jf, vf = lire("ind_fol.def")
        nr, jr, vr = lire("pro_rac.def")
    except (FileNotFoundError, IndexError):
        return {}
    # Les deux fichiers n'ont PAS la même grille de jours (OUTV : 1/158/188/299/365 pour
    # le feuillage, 1/160/190/260/365 pour les racines). On interpole linéairement les
    # deux sur la grille RÉUNIE, ce que fait aussi le C++ entre ses points de rupture.
    import numpy as _np
    jours = sorted(set(jf) | set(jr))

    def col(noms, motifs):
        return [k for k, nm in enumerate(noms) if any(m in nm.lower() for m in motifs)]

    corresp = {"feuillus": ["feuillu"], "conifers": ["conifere"],
               "agri": ["agricole"], "humides": ["tourbiere", "milieu_humide"],
               "ouverts": ["sol_nu"]}
    out = {}
    for cl, motifs in corresp.items():
        cf, cr = col(nf, motifs), col(nr, motifs)
        if not cf or not cr:
            continue
        lai_pts = [sum(v[k] for k in cf) / len(cf) for v in vf]
        rac_pts = [sum(v[k] for k in cr) / len(cr) for v in vr]
        lai = list(_np.interp(jours, jf, lai_pts))
        rac = list(_np.interp(jours, jr, rac_pts))
        out[cl] = (list(jours), lai, rac)
    return out


# ── IDENTITÉ DES TRONÇONS ───────────────────────────────────────────────────
# Le dépôt manipule TROIS numérotations et chaque script refaisait la conversion à la
# main. Le 2026-08-11 cela a produit trois appariements ratés d'affilée, dont un qui
# rendait des KGE de -0.25 : des nombres faux, pas une erreur. D'où ces fonctions,
# à utiliser partout plutôt que de reconstruire l'appariement sur place.
#
#   1. ENTIER LOCAL      : identifiant du tronçon dans le troncon.trl de la région
#                          (1..n_troncons). C'est ce que porte `node_ids`.
#   2. CHAÎNE PROVINCIALE: "REG#####" (ex "OUTV00123"), coordonnée `troncon_id` des
#                          stockages provinciaux de post-traitement.
#   3. RANG              : position dans le tableau provincial (`troncon_idx`, 0..28034).
#                          N'a AUCUN sens hydrologique, c'est un indice de stockage.

def id_provincial(region: str, troncon_local: int) -> str:
    """Entier local -> chaîne provinciale ("OUTV", 123) -> "OUTV00123"."""
    return f"{region.upper()}{int(troncon_local):05d}"


def id_local(id_prov: str) -> tuple[str, int]:
    """Chaîne provinciale -> (région, entier local). "OUTV00123" -> ("OUTV", 123)."""
    s = str(id_prov)
    return s[:4].upper(), int(s[4:])


def appariement_provincial(region: str, troncons_locaux, ids_provinciaux):
    """Rangs, dans un tableau provincial, des tronçons locaux demandés.

    `ids_provinciaux` : la coordonnée `troncon_id` du stockage (chaînes "REG#####").
    Retourne une liste de rangs, `None` pour les tronçons absents. LÈVE une erreur si
    AUCUN ne s'apparie : un appariement vide est toujours un bug de convention, jamais
    un résultat, et le laisser passer produit des scores absurdes silencieux.
    """
    pos = {str(t): k for k, t in enumerate(ids_provinciaux)}
    rangs = [pos.get(id_provincial(region, t)) for t in troncons_locaux]
    if all(r is None for r in rangs):
        raise ValueError(
            f"aucun tronçon apparié pour {region.upper()} : les identifiants fournis "
            f"ressemblent à {str(ids_provinciaux[0])!r}, on cherchait "
            f"{id_provincial(region, troncons_locaux[0])!r}")
    return rangs


def load_mcguinness_nodes(project_dir, node_ids, sim_subdir="simulation/simulation",
                          device="cpu"):
    """Coefficient multiplicatif d'optimisation de l'ETP McGuinness
    (etp-mc-guiness.csv), agrégé UHRH -> tronçon au prorata des superficies.

    TROUVÉ LE 2026-08-16, remarque d'Essi : CINQ plateformes sur six utilisent
    McGuinness et une seule Linacre — les préfixes l'encodent (MG contre LN). Or notre
    socle était ancré sur LN24HA, la seule Linacre ET la moins bonne des six (0.7531
    contre 0.830 pour MG24HK). Pire, `mcguinness_etp` était appelé SANS ce coefficient,
    qui vaut 0.600 sur MG24HK et 0.850 sur MG24HS : notre ETP McGuinness était donc
    jusqu'à 67 % trop forte. Neuvième fichier de calage présent et jamais lu.
    """
    from pathlib import Path
    import numpy as np
    import torch as _t
    proj = load_project(str(project_dir), sim_subdir)
    troncons = _parse_troncon(Path(project_dir) / "physitel" / "troncon.trl")
    t2u = {t["id"]: t["uhrh_ids"] for t in troncons}
    uhrh = proj["uhrh"]
    f = Path(project_dir) / sim_subdir / "etp-mc-guiness.csv"
    if not f.exists():
        return None
    coef = {}
    for ln in f.read_text(encoding="latin-1").splitlines():
        t = [x.strip() for x in ln.split(";")]
        if len(t) >= 2 and t[0].isdigit():
            coef[int(t[0])] = float(t[1])
    if not coef:
        return None
    out = np.ones(len(node_ids), dtype=np.float32)
    for j, nid in enumerate(node_ids):
        vs, ws = [], []
        for u in t2u.get(int(nid), []):
            u = abs(int(u))
            if u in coef:
                vs.append(coef[u]); ws.append(max(uhrh.get(u, {}).get("area_km2", 1.0), 1e-9))
        if vs:
            out[j] = float(np.average(vs, weights=ws))
    return _t.tensor(out, dtype=_t.float32, device=device)
