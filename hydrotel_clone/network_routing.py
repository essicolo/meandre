"""Routeur RÉSEAU fidèle Hydrotel (onde cinématique modifiée), assemblage des
solveurs validés (transfert_riviere, transfert_lac) avec l'accumulation
topologique et les sous-pas globaux. Porté du C++ (ONDE_CINEMATIQUE_MODIFIEE::
Calcule, l.572). Sous-projet PROPRE.

CONÇU GÉNÉRIQUE (note Essi) : prend une représentation de réseau ABSTRAITE
(tronçons = dicts type/géométrie, tableau aval, ordre topologique), pas le format
PHYSITEL. Le .trl n'est qu'un adaptateur de validation (delisle_network). En open
data, méandre construira la même structure depuis son graphe + territorial.

Orchestration (par pas journalier) :
  1. nt global depuis le Courant (max célérité des rivières), dt>=1800s.
  2. nt sous-pas : reset qamont=0 ; balayage topologique amont→aval ; chaque
     tronçon route (rivière/lac) avec son état _ocm (pas précédent) ; qaval
     propagé au qamont du tronçon aval ; état mis à jour ; débit aval MOYEN.
"""
from __future__ import annotations
from hydrotel_clone.routing import celerite, transfert_riviere


def _transfert_lac_scalar(dt, aire, c, k, qa, ql, qb, qc, qm):
    from hydrotel_clone.routing import MAXITER, EPSILON
    hb = (qb / c) ** (1.0 / k) if qb > 0 else 0.0
    haut = max(0.0, hb + ((qa + qc) / 2.0 - qb + (ql + qm) / 2.0) * dt / aire)
    for _ in range(MAXITER):
        f0 = haut - hb + (qb + c * haut ** k - qa - qc - ql - qm) * dt / aire / 2.0
        f1 = 1.0 + c * k * haut ** (k - 1.0) * dt / aire / 2.0
        step = f0 / f1
        haut = haut - step
        if abs(step) < EPSILON:
            break
    return c * max(haut, 0.0) ** k


def route_network(reaches, downstream, topo_order, apport_lateral, pdts=86400):
    """Route un réseau sur T pas journaliers.

    reaches : liste de dicts par tronçon. Rivière : {type:'river', lng, lrg, pte,
        man}. Lac : {type:'lake', surface_m2, c, k}.
    downstream : liste, downstream[i] = index du tronçon aval (ou -1 si exutoire).
    topo_order : liste d'indices amont→aval (sources d'abord).
    apport_lateral : (T, nReach) apport latéral [m3/s] (constant sur les sous-pas).
    Retourne debit_aval : (T, nReach) débit aval moyen journalier [m3/s]."""
    import numpy as np
    nR = len(reaches)
    T = len(apport_lateral)
    # état _ocm par tronçon : qamont, qaval, qapportlat (pas précédent)
    st = [{"qa": 0.0, "qb": 0.0, "ql": 0.0} for _ in range(nR)]
    out = np.zeros((T, nR))
    rivers = [i for i, r in enumerate(reaches) if r["type"] == "river"]

    for day in range(T):
        apl = apport_lateral[day]
        # nt global depuis la célérité des rivières (Courant)
        ntmax = 1
        for i in rivers:
            r = reaches[i]
            c = celerite(r["lng"], r["lrg"], r["pte"], r["man"], st[i]["qa"], st[i]["qb"])
            if c > 0:
                nt_i = int(pdts / int(r["lng"] / c + 1.0)) + 1
                ntmax = max(ntmax, nt_i)
        if pdts / ntmax < 1800:
            dt = 1800
            nt = max(1, int(pdts / dt))
        else:
            nt = ntmax
            dt = int(pdts / nt)

        qaval_moy = [0.0] * nR
        for t in range(nt):
            qamont_acc = [0.0] * nR
            # le qamont du sous-pas vient de l'accumulation ; on balaye amont→aval
            for i in topo_order:
                r = reaches[i]
                qc = qamont_acc[i]
                qm = float(apl[i])
                qa, qb, ql = st[i]["qa"], st[i]["qb"], st[i]["ql"]
                if r["type"] == "river":
                    qd = transfert_riviere(dt, r["lng"], r["lrg"], r["pte"], r["man"],
                                           qa, ql, qb, qc, qm)
                else:
                    qd = _transfert_lac_scalar(dt, r["surface_m2"], r["c"], r["k"],
                                               qa, ql, qb, qc, qm)
                qd = max(0.0, qd)
                d = downstream[i]
                if d >= 0:
                    qamont_acc[d] += qd
                st[i] = {"qa": qc, "qb": qd, "ql": qm}
                qaval_moy[i] = qd if t == 0 else (qaval_moy[i] * t + qd) / (t + 1.0)
        out[day] = qaval_moy
    return out


def delisle_network(del_dir):
    """Adaptateur de validation : parse troncon.trl + noeuds.nds de DELISLE en
    (reaches, downstream, topo_order, id->index). NON utilisé en production."""
    import numpy as np
    # altitudes des nœuds
    nlines = open(f"{del_dir}/physitel/noeuds.nds", encoding="latin-1").read().split()
    # format : "1 / 200 / Noeuds" puis par nœud id x y alt 0
    # on parse en flux après l'entête (3 tokens : '1','200','Noeuds')
    toks = nlines
    # trouve le début des données (après 'Noeuds')
    k0 = toks.index("Noeuds") + 1
    Z = {}
    i = k0
    while i + 4 < len(toks):
        try:
            nid = int(toks[i]); alt = float(toks[i + 3])
            Z[nid] = alt; i += 5
        except (ValueError, IndexError):
            break

    # troncons (flux de tokens)
    t = open(f"{del_dir}/physitel/troncon.trl", encoding="latin-1").read().split()
    p = t.index("TRONCONS") + 1
    nb = int(t[p - 2])  # le nombre de troncons est avant 'TRONCONS'
    reaches = [None] * nb
    aval_node = [0] * nb
    amont_nodes = [[] for _ in range(nb)]
    idx_of_id = {}
    i = p
    for k in range(nb):
        tid = int(t[i]); typ = int(t[i + 1]); navd = int(t[i + 2]); i += 3
        idx_of_id[tid] = k
        aval_node[k] = navd
        if typ == 1:   # rivière (type-1 → 0)
            namont = int(t[i]); lng = float(t[i + 1]); lrg = float(t[i + 2]); man = float(t[i + 3]); i += 4
            amont_nodes[k] = [namont]
            pte = max((Z.get(namont, 0) - Z.get(navd, 0)) / lng, 0.0025)
            reaches[k] = {"type": "river", "lng": lng, "lrg": lrg, "pte": pte, "man": man}
        else:          # lac (type-2 → 1)
            nna = int(t[i]); i += 1
            ams = [int(t[i + j]) for j in range(nna)]; i += nna
            lng = float(t[i]); surf = float(t[i + 1]); c = float(t[i + 2]); k_ = float(t[i + 3]); i += 4
            amont_nodes[k] = ams
            reaches[k] = {"type": "lake", "surface_m2": surf * 1e6, "c": c, "k": k_}
        # LectureZoneAmont : nb_zones + zone ids
        nz = int(t[i]); i += 1 + nz
        # ordre de shreve (fichier type 2) : 1 token
        i += 1

    # connectivité : downstream[T] = troncon dont amont contient aval_node[T]
    node_to_amont_troncon = {}
    for k in range(nb):
        for n in amont_nodes[k]:
            node_to_amont_troncon[n] = k
    downstream = [node_to_amont_troncon.get(aval_node[k], -1) for k in range(nb)]

    # ordre topologique amont→aval (Kahn : sources d'abord)
    indeg = [0] * nb
    for d in downstream:
        if d >= 0:
            indeg[d] += 1
    from collections import deque
    q = deque(i for i in range(nb) if indeg[i] == 0)
    topo = []
    while q:
        u = q.popleft(); topo.append(u)
        d = downstream[u]
        if d >= 0:
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    return reaches, downstream, topo, idx_of_id


def topo_levels(downstream, nb):
    """Niveaux topologiques : level[i] = 0 pour les sources, sinon max(level amont)+1.
    Les tronçons d'un même niveau sont indépendants (vectorisables). Retourne une
    liste de listes d'indices, par niveau croissant (amont→aval)."""
    upstream = [[] for _ in range(nb)]
    for i, d in enumerate(downstream):
        if d >= 0:
            upstream[d].append(i)
    level = [-1] * nb
    # propage en ordre topologique (Kahn déjà disponible via topo, mais recompute simple)
    from collections import deque
    indeg = [0] * nb
    for d in downstream:
        if d >= 0:
            indeg[d] += 1
    q = deque(i for i in range(nb) if indeg[i] == 0)
    for i in q:
        level[i] = 0
    while q:
        u = q.popleft()
        d = downstream[u]
        if d >= 0:
            level[d] = max(level[d], level[u] + 1)
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    nlev = max(level) + 1
    groups = [[] for _ in range(nlev)]
    for i in range(nb):
        groups[level[i]].append(i)
    return groups
