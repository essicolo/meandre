"""Loader des paramètres Hydrotel PAR UHRH, lus directement dans les fichiers du
projet (physitel/, physio/, simulation/). AUCUNE valeur hardcodée, AUCUN NeRF :
c'est la source de vérité pour cloner Hydrotel fidèlement sur tout le bassin.

Fichiers lus (projet DELISLE, idem SLSO) :
  physitel/uhrh.csv               lat, lon, pente (ratio), orientation par UHRH
  physitel/type_sol.cla           classe de texture par UHRH (index 0-based dans .sol)
  physitel/proprietehydrolique.sol  table texture → thetas/thetacc/thetapf/ks/psis/lambda/alpha
  physitel/occupation_sol.cla     pixels par classe d'occupation, par UHRH
  physio/ind_fol.def              indice foliaire (cycle annuel) par classe d'occupation
  physio/pro_rac.def              profondeur racinaire (cycle annuel) par classe d'occupation
  simulation/<sim>/bv3c.csv       z1/z2/z3, humidité rel. init, extinction, krec, assèch, cin, recharge
                                  + classes intégrées imperméable/eau

Le test de régression (python hydrotel_clone/hydrotel_params.py) reconstruit UHRH1
et vérifie qu'il reproduit les valeurs connues de validate_chain (validé décimale vs C++).
"""
from __future__ import annotations
import os


def _read_lines(path, encoding="latin-1"):
    with open(path, encoding=encoding) as f:
        return f.read().splitlines()


# ── proprietehydrolique.sol : table texture (index 0-based = classe .cla) ──
def load_sol(path):
    """Retourne la liste des textures dans l'ordre du fichier (index 0-based =
    classe de type_sol.cla). Chaque entrée : dict(name, thetas, thetacc, thetapf,
    ks_m_per_h, psis, lam, alpha)."""
    lines = _read_lines(path)
    # entête : "3" / "19 7" / titre / colonnes ; les textures suivent
    out = []
    for ln in lines[4:]:
        toks = ln.split()
        if len(toks) < 8:
            continue
        name = toks[0]
        v = [float(x) for x in toks[1:8]]
        out.append(dict(name=name, thetas=v[0], thetacc=v[1], thetapf=v[2],
                        ks=v[3], psis=v[4], lam=v[5], alpha=v[6]))
    return out


# ── type_sol.cla : classe de texture par UHRH ──
def load_type_sol(path):
    lines = _read_lines(path)
    out = {}
    for ln in lines[1:]:                 # 1re ligne = entête ("1")
        t = ln.split()
        if len(t) >= 2:
            out[int(t[0])] = int(t[1])
    return out


# ── uhrh.csv : géométrie / position ──
def load_uhrh(path):
    lines = _read_lines(path)
    out = {}
    for ln in lines:
        c = ln.split(";")
        if len(c) >= 9 and c[0].strip().isdigit():
            uid = int(c[0])
            out[uid] = dict(altitude=float(c[2]), slope=float(c[3]),
                            orientation=float(c[4]), nb_pixel=int(c[5]),
                            area_km2=float(c[6]), lon=float(c[7]), lat=float(c[8]))
    return out


# ── occupation_sol.cla : pixels par classe d'occupation, par UHRH ──
def load_occupation(path):
    lines = _read_lines(path)
    # "1" / "10" / entête noms entre guillemets / data : uhrh c1..c10
    import re
    names = re.findall(r'"([^"]+)"', lines[2])
    out = {}
    for ln in lines[3:]:
        t = ln.split()
        if len(t) >= 1 + len(names) and t[0].isdigit():
            uid = int(t[0])
            cnts = [float(x) for x in t[1:1 + len(names)]]
            out[uid] = dict(zip(names, cnts))
    return names, out


# ── ind_fol.def / pro_rac.def : cycles annuels par classe d'occupation ──
def load_cycle(path):
    """Retourne (jours_bp, {nom_classe: [valeurs]}) du cycle annuel."""
    import re
    lines = _read_lines(path)
    names = re.findall(r'"([^"]+)"', lines[3])
    jbp, cols = [], {n: [] for n in names}
    for ln in lines[4:]:
        t = ln.split()
        if len(t) >= 1 + len(names):
            jbp.append(int(float(t[0])))
            for k, n in enumerate(names):
                cols[n].append(float(t[1 + k]))
    return jbp, cols


# ── bv3c.csv : params de calage BV3C2 par UHRH ──
def load_bv3c(path):
    lines = _read_lines(path)
    integ_imperm, integ_eau = [], []
    out = {}
    for ln in lines:
        if ln.upper().startswith("CLASSE INTEGRE IMPERMEABLE"):
            integ_imperm = [int(x) for x in ln.split(";")[1:] if x.strip().isdigit()]
        elif ln.upper().startswith("CLASSE INTEGRE EAU"):
            integ_eau = [int(x) for x in ln.split(";")[1:] if x.strip().isdigit()]
        c = ln.split(";")
        if len(c) >= 12 and c[0].strip().isdigit():
            uid = int(c[0]); v = [float(x) for x in c[1:12]]
            out[uid] = dict(z1=v[0], z2=v[1], z3=v[2], hri1=v[3], hri2=v[4], hri3=v[5],
                            extinction=v[6], krec=v[7], assech=v[8], cin=v[9], recharge=v[10])
    return out, integ_imperm, integ_eau


# ── Projet complet ──
def load_project(project_dir, sim_subdir="simulation/simulation"):
    P = lambda *a: os.path.join(project_dir, *a)
    sol = load_sol(P("physitel", "proprietehydrolique.sol"))
    tex = load_type_sol(P("physitel", "type_sol.cla"))
    uhrh = load_uhrh(P("physitel", "uhrh.csv"))
    occ_names, occ = load_occupation(P("physitel", "occupation_sol.cla"))
    leaf_jbp, leaf = load_cycle(P("physio", "ind_fol.def"))
    root_jbp, root = load_cycle(P("physio", "pro_rac.def"))
    bv3c, imperm, eau = load_bv3c(P(sim_subdir, "bv3c.csv"))
    return dict(sol=sol, texture=tex, uhrh=uhrh, occ_names=occ_names, occ=occ,
                leaf_jbp=leaf_jbp, leaf=leaf, root_jbp=root_jbp, root=root,
                bv3c=bv3c, integ_imperm=imperm, integ_eau=eau,
                uhrh_ids=sorted(uhrh.keys()))


def uhrh_fractions(proj, uid):
    """fsa/fse/fsi (fractions de superficie) depuis l'occupation + classes intégrées."""
    names = proj["occ_names"]; cnts = proj["occ"][uid]
    tot = sum(cnts.values())
    if tot <= 0:
        return 1.0, 0.0, 0.0, tot
    # classes intégrées : 1-based dans bv3c.csv → noms via l'ordre occ_names
    imp_names = [names[i - 1] for i in proj["integ_imperm"] if 1 <= i <= len(names)]
    eau_names = [names[i - 1] for i in proj["integ_eau"] if 1 <= i <= len(names)]
    fsi = sum(cnts[n] for n in imp_names) / tot
    fse = sum(cnts[n] for n in eau_names) / tot
    fsa = max(1.0 - fsi - fse, 0.0)
    return fsa, fse, fsi, tot


def uhrh_texture(proj, uid):
    """Texture (dict du .sol) de l'UHRH : classe .cla = index 0-based dans .sol."""
    cls = proj["texture"][uid]
    return proj["sol"][cls]


if __name__ == "__main__":
    # ── Régression UHRH1 DELISLE vs valeurs connues de validate_chain ──
    DEL = r"C:/Users/parse01/documents-locaux/GitHub/hydrotel/DemoProject/DELISLE"
    proj = load_project(DEL)
    print(f"projet chargé : {len(proj['uhrh_ids'])} UHRH, {len(proj['sol'])} textures")

    u = proj["uhrh"][1]
    fsa, fse, fsi, tot = uhrh_fractions(proj, 1)
    tx = uhrh_texture(proj, 1)
    b = proj["bv3c"][1]
    print(f"\nUHRH1 : lat={u['lat']:.5f} slope={u['slope']:.6f} orient={u['orientation']:.0f}")
    print(f"  texture (classe {proj['texture'][1]}) = {tx['name']} : thetas={tx['thetas']} "
          f"ks={tx['ks']} alpha={tx['alpha']}")
    print(f"  fractions : fsa={fsa:.3f} fse={fse:.3f} fsi={fsi:.3f} (tot {tot:.0f} px)")
    print(f"  bv3c : z={b['z1']}/{b['z2']}/{b['z3']} hri={b['hri1']} krec={b['krec']:.2e} "
          f"cin={b['cin']} assech={b['assech']} des={b['extinction']}")

    # assertions vs validate_chain (UHRH1, validé décimale vs C++)
    assert abs(u["lat"] - 45.29459) < 1e-4, u["lat"]
    assert abs(u["slope"] - 0.026023) < 1e-5, u["slope"]
    assert tx["name"] == "sandy_loam", tx["name"]
    assert abs(tx["thetacc"] - 0.207) < 1e-6 and abs(tx["thetapf"] - 0.095) < 1e-6
    assert abs(tx["alpha"] - 4.5) < 1e-6, tx["alpha"]
    assert abs(fse - 119 / 1754) < 1e-4 and abs(fsi - (1111 + 280) / 1754) < 1e-4, (fse, fsi)
    assert abs(fsa - 0.139) < 1e-3, fsa
    assert (b["z1"], b["z2"], b["z3"]) == (0.1, 0.4, 1.0)
    assert abs(b["krec"] - 1e-6) < 1e-12 and b["cin"] == 0.3 and b["hri1"] == 0.9
    print("\nREGRESSION OK — loader reproduit UHRH1 (validate_chain) depuis les fichiers")

    # distribution textures sur tout le bassin
    from collections import Counter
    dist = Counter(proj["sol"][c]["name"] for c in proj["texture"].values())
    print("textures du bassin :", dict(dist))
