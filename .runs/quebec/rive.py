"""RIVE du Saint-Laurent par tronçon, dérivée du RÉSEAU et non d'un découpage administratif.

Idée d'Essi (2026-08-30) : le champ spatial devrait pouvoir être discontinu, entre bassins
de dimensions différentes et surtout de part et d'autre du fleuve. Les hydrologues du
Québec krigent leurs erreurs de modélisation en deux zones distinctes selon la rive, en
reconstruction historique. Ce n'est pas une convention administrative : c'est la
reconnaissance opérationnelle que la structure d'erreur diffère.

POURQUOI CETTE DISCONTINUITÉ EST PHYSIQUE. Deux tronçons distants de cinq kilomètres de
part et d'autre du fleuve sont voisins pour un encodage euclidien et infiniment éloignés
pour l'eau : ils ne partagent aucun exutoire. Et la géologie diffère franchement, Bouclier
au nord (till mince sur roc, réponse rapide), Appalaches et basses-terres au sud (dépôts
meubles profonds, agriculture, drainage). Un champ continu est forcé de moyenner les deux
dans la bande centrale, précisément là où se trouvent MONT et SLSO.

MÉTHODE. On classe l'EXUTOIRE de chaque tronçon, pas le tronçon lui-même : un affluent
appartient à la rive de son fleuve, quelle que soit sa position. L'axe du Saint-Laurent
est approché par une ligne entre deux points connus, et le signe du produit vectoriel
donne la rive. C'est de la géométrie sur le réseau, pas un fichier de régions.

RÉSERVE ASSUMÉE. Une droite est une approximation grossière d'un fleuve qui serpente.
Elle suffit pour un fanion binaire, elle ne suffirait pas pour une distance au fleuve.
Les bassins qui ne drainent pas vers le Saint-Laurent (baie James, Ungava) tombent d'un
côté ou de l'autre sans que ça veuille dire grand-chose ; ils sont marqués `hors`.
"""
import numpy as np
import torch

# AXE DU SAINT-LAURENT en POLYLIGNE, et non en droite. Une droite unique ne peut pas
# separer les deux rives a la fois dans le troncon fluvial (oriente est-nord-est) et dans
# l'estuaire, qui s'elargit et tourne franchement vers l'est. Premiere version avec une
# droite Montreal - Pointe-des-Monts : la Cote-Nord tombait au SUD et la Monteregie au
# NORD. Points pris sur le cours du fleuve, d'amont en aval.
AXE = [(-74.60, 45.10),   # entree du lac Saint-Francois
       (-73.55, 45.50),   # Montreal
       (-72.55, 46.35),   # Trois-Rivieres
       (-71.21, 46.81),   # Quebec
       (-69.72, 47.85),   # Riviere-du-Loup / Saint-Simeon
       (-67.38, 49.32),   # Pointe-des-Monts
       (-64.20, 49.60),   # detroit de Jacques-Cartier
       (-61.00, 49.90)]   # golfe, au nord d'Anticosti
# Au-dela de cette distance a l'axe, le bassin ne draine pas vers le fleuve. Seuil
# MESURE et non suppose : les bassins du Saint-Laurent sont a 17 a 233 km de l'axe,
# abit a 489 et labi a 553, tous deux tributaires de la baie James. 350 les separe
# avec une marge de 116 km des deux cotes.
DIST_MAX_KM = 350.0


def _km(lon, lat, lat0):
    return np.asarray(lon) * 111.32 * np.cos(np.radians(lat0)), np.asarray(lat) * 110.574


def _cote(lon, lat):
    """Distance SIGNEE a la polyligne, en km. Positif au nord-ouest, negatif au sud-est.

    On prend le segment le plus PROCHE de chaque point, puis le signe du produit
    vectoriel sur ce segment. Une polyligne suit le fleuve la ou une droite le coupe.
    """
    lat0 = 47.5
    px, py = _km(lon, lat, lat0)
    px = np.atleast_1d(px); py = np.atleast_1d(py)
    meilleure = np.full(px.shape, np.inf)
    signee = np.zeros(px.shape)
    for (ax_, ay_), (bx_, by_) in zip(AXE[:-1], AXE[1:]):
        ax, ay = _km(ax_, ay_, lat0)
        bx, by = _km(bx_, by_, lat0)
        ux, uy = bx - ax, by - ay
        L2 = ux * ux + uy * uy
        t = np.clip(((px - ax) * ux + (py - ay) * uy) / L2, 0.0, 1.0)
        cx, cy = ax + t * ux, ay + t * uy
        d = np.hypot(px - cx, py - cy)
        croix = (ux * (py - ay) - uy * (px - ax)) / np.sqrt(L2)
        prend = d < meilleure
        meilleure = np.where(prend, d, meilleure)
        signee = np.where(prend, croix, signee)
    return signee, meilleure


def exutoires(edge_index, n_noeuds):
    """Pour chaque nœud, l'indice du nœud terminal vers lequel il s'écoule.

    Le graphe est une union par blocs de réseaux disjoints : il y a donc plusieurs
    exutoires, un par bassin. On descend le graphe jusqu'à un nœud sans successeur.
    """
    suiv = np.full(n_noeuds, -1, dtype=np.int64)
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    suiv[src] = dst                           # un seul aval par tronçon (arbre)
    out = np.arange(n_noeuds, dtype=np.int64)
    for _ in range(4096):                     # borne large, le réseau est un arbre
        nxt = np.where(suiv[out] >= 0, suiv[out], out)
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out


def rive_par_noeud(node_coords, edge_index):
    """Retourne (rive, distance_km) où rive vaut +1 (nord-ouest), -1 (sud-est), 0 (hors).

    La rive est celle de l'EXUTOIRE, héritée par tout le bassin amont.
    """
    c = node_coords.detach().cpu().numpy()
    n = c.shape[0]
    ex = exutoires(edge_index, n)
    # CENTROIDE DU BASSIN, pas position de l'exutoire. Dette 18 du registre : les
    # topologies cotieres de PHYSITEL raccordent des bassins independants a un exutoire
    # ARTIFICIEL dont la position ne veut rien dire (le noeud 1 de gasp recoit 167 aretes
    # de 127 km medians). Classer sur l'exutoire mettait la Cote-Nord au sud.
    lon_c = np.zeros(n); lat_c = np.zeros(n); cnt = np.zeros(n)
    np.add.at(lon_c, ex, c[:, 0]); np.add.at(lat_c, ex, c[:, 1]); np.add.at(cnt, ex, 1.0)
    cnt = np.maximum(cnt, 1.0)
    d_bassin, dist = _cote(lon_c[ex] / cnt[ex], lat_c[ex] / cnt[ex])
    rive = np.sign(d_bassin).astype(np.float32)
    rive[dist > DIST_MAX_KM] = 0.0
    return torch.from_numpy(rive), torch.from_numpy(dist.astype(np.float32))


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, ".")
    from meandre.data.basin_cache import BasinCache

    # On lit les caches DIRECTEMENT : ce calcul n'a besoin que des coordonnees et du
    # graphe. Passer par load_domain chargerait 5.6 Go de forcage pour rien, et
    # echouerait sur les regions dont une variante de forcage manque (vaud n'a pas de
    # -budyko en local).
    noms = (os.environ.get("RIVE_REGIONS")
            or "outv,gasp,mont,sagu,slno,abit,slso,cnda,cndb,cndc,cndd,cnde,labi,vaud").split(",")
    noms = [n.strip() for n in noms if n.strip()]
    cs, es, tranches, off = [], [], {}, 0
    for n in noms:
        h = BasinCache(f"D:/meandre-data/quebec/{n}.duckdb").load(device="cpu")
        c, g = h["node_coords"], h["graph"]
        cs.append(c)
        if g.edge_index.numel():
            es.append(g.edge_index + off)
        tranches[n] = (off, off + c.shape[0])
        off += c.shape[0]
    coords = torch.cat(cs)
    ei = torch.cat(es, dim=1) if es else torch.zeros(2, 0, dtype=torch.long)
    r, d = rive_par_noeud(coords, ei)
    print("")
    print(f"{len(r):,} troncons, {len(noms)} regions")
    for v, nom in ((1.0, "nord-ouest"), (-1.0, "sud-est"), (0.0, "hors bassin fleuve")):
        m = r == v
        print(f"  {nom:20s} {int(m.sum()):6,d} troncons ({100 * float(m.float().mean()):.1f} %)")
    print("")
    print("par region :")
    for nom, (a, b) in tranches.items():
        sub = r[a:b]
        print(f"  {nom:6s} nord {int((sub > 0).sum()):5d} | sud {int((sub < 0).sum()):5d} | "
              f"hors {int((sub == 0).sum()):5d}")
    import numpy as _np
    _np.savez_compressed("D:/meandre-data/quebec/rive-troncons.npz",
                         regions=_np.array(noms), rive=r.numpy(), dist_km=d.numpy(),
                         debut=_np.array([tranches[n][0] for n in noms]))
    print("")
    print("ecrit : D:/meandre-data/quebec/rive-troncons.npz")
