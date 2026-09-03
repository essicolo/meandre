"""Banc des DEFAUTS connus : reproduire en secondes ce que les simulations montrent.

Pourquoi ce banc (2026-09-03, demande d'Essi). Le banc de physique dit que la colonne
est juste ; ce n'est pas la question. Les deux defauts qui rendent le modele non livrable
sont AILLEURS : le champ spatial se comporte comme un champ de trois parametres, et la
simulation arrive un a trois jours en retard, d'autant plus que le bassin est grand. Les
constater demandait jusqu'ici un entrainement ou une simulation provinciale, soit des
heures. Ce banc les reproduit sans colonne, sans forcage et sans GPU, en quelques
secondes, pour qu'un correctif se juge immediatement.

Chaque banc rend un NOMBRE qui doit bouger quand le defaut est corrige.

  RETARD DU ROUTAGE. On injecte une impulsion d'apport lateral dans le VRAI reseau et on
  mesure le decalage du centre de masse a l'exutoire. Aucune colonne n'intervient : le
  retard mesure est purement celui du routage. Le temps de parcours physique du reseau
  sert de reference (somme des longueurs sur le chemin le plus long, divisee par la
  celerite). Un routage sain doit en etre proche ; le defaut se lit dans l'ecart.

  EFFONDREMENT DU CHAMP. On evalue le champ d'un point de reprise sur les coordonnees
  reelles et on compte : sorties sans gradient (cablees sur rien), sorties rembourrees
  a zero parce que le point de reprise est plus ancien que le champ, et sorties dont
  l'etalement spatial est sous 5 %, donc des constantes deguisees.

  python .runs/quebec/banc_defauts.py                     retard sur outv, champ sur outv
  python .runs/quebec/banc_defauts.py --region gasp
  python .runs/quebec/banc_defauts.py --retard-seul
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

_DATA = os.environ.get("MEANDRE_DATA", "D:/meandre-data")


# ---------------------------------------------------------------------------
# Banc 1 : le retard du routage
# ---------------------------------------------------------------------------

def _exutoire(graph):
    """Le noeud sans lien sortant qui draine le plus d'amont."""
    n = graph.is_lake.shape[0]
    src = graph.edge_index[0].numpy()
    sortants = np.zeros(n, dtype=bool)
    sortants[src] = True
    candidats = np.flatnonzero(~sortants)
    if candidats.size == 1:
        return int(candidats[0])
    # Plusieurs racines : on prend celle qui a le plus de noeuds en amont.
    dst = graph.edge_index[1].numpy()
    amont = np.bincount(dst, minlength=n)
    return int(candidats[np.argmax(amont[candidats])])


def _longueur_chemin_max(graph):
    """Somme des longueurs de tronçon sur le plus long chemin vers l'exutoire (m)."""
    n = graph.is_lake.shape[0]
    src = graph.edge_index[0].numpy()
    dst = graph.edge_index[1].numpy()
    L = graph.edge_attr[:, 0].numpy().astype(float)
    dist = np.zeros(n)
    for i in graph.topo_order.numpy():
        pass
    ordre = graph.topo_order.numpy()
    pos = {int(v): k for k, v in enumerate(ordre)}
    liens = sorted(range(len(src)), key=lambda e: pos.get(int(src[e]), 0))
    for e in liens:
        s, d = int(src[e]), int(dst[e])
        if dist[s] + L[e] > dist[d]:
            dist[d] = dist[s] + L[e]
    return float(dist.max())


def retard_routage(reg, k_heures, x_musk=0.2, n_substeps=64, jours=90, jour_pulse=10,
                   celerite_ms=1.0, sans_lacs=True):
    """Retard du centre de masse a l'exutoire, en jours, pour un K donne.

    Les lacs sont neutralises par defaut : on isole ainsi l'effet de la constante de
    Muskingum, seul parametre que le banc fait varier. Le delai propre au mode de
    routage (un jour par lac traverse en mode lagged) se mesure separement.
    """
    from meandre.data.basin_cache import BasinCache
    from meandre.routing.message_passing import RoutingLayer
    from meandre.routing.operator_routing import (
        build_operator_topo, build_operator_state, route_operator,
    )

    d = BasinCache(f"{_DATA}/quebec/{reg}.duckdb").load(device=torch.device("cpu"))
    g = d["graph"]
    n = g.is_lake.shape[0]
    if sans_lacs:
        g.is_lake = torch.zeros_like(g.is_lake)

    couche = RoutingLayer(use_travel_time_attention=False,
                          routing_mode="operator-lagged", routing_substeps=n_substeps)
    topo = build_operator_topo(g, lagged=True)
    K = torch.full((n,), float(k_heures) * 3600.0)
    X = torch.full((n,), float(x_musk))
    c01, c2 = couche.muskingum.precompute_coefficients(K, X)
    op = build_operator_state(topo, c01, c2, n_substeps)

    ex = _exutoire(g)
    zero = torch.zeros(n)
    Q_prev = torch.zeros(n)
    serie = np.zeros(jours)
    with torch.no_grad():
        for t in range(jours):
            q_lat = torch.full((n,), 1.0) if t == jour_pulse else zero
            Q = route_operator(couche, topo, op, q_lat, Q_prev, zero, None, None,
                               None, t, True)
            Q_prev = Q
            serie[t] = float(Q[ex])
    if serie.sum() <= 0:
        return None
    jours_axe = np.arange(jours)
    centre = float((serie * jours_axe).sum() / serie.sum())
    pic = int(np.argmax(serie))
    lmax = _longueur_chemin_max(g)
    physique_j = lmax / celerite_ms / 86400.0
    return {"retard_centre_j": centre - jour_pulse, "retard_pic_j": pic - jour_pulse,
            "physique_j": physique_j, "chemin_km": lmax / 1000.0, "n": n,
            "n_lacs": int(d["graph"].is_lake.sum()) if not sans_lacs else 0}


# ---------------------------------------------------------------------------
# Banc 2 : l'effondrement du champ
# ---------------------------------------------------------------------------

def etat_champ(reg, chemin_ck):
    """Compte les sorties mortes, rembourrees et constantes d'un point de reprise."""
    from meandre.spatial.field_network import SpatialFieldNetwork, SpatialParams
    from meandre.data.basin_cache import BasinCache

    noms = [f for f in SpatialParams.__dataclass_fields__][:SpatialParams.N_PARAMS]
    N = SpatialParams.N_PARAMS
    sd = torch.load(chemin_ck, map_location="cpu")["state_dict"]
    k = sd["spatial_encoder.fc_out.weight"].shape[0]
    rembourrees = noms[k:N]

    d = BasinCache(f"{_DATA}/quebec/{reg}.duckdb").load(device=torch.device("cpu"))
    terr = d["territorial"]
    coords = d["node_coords"]
    reseau = SpatialFieldNetwork(n_territorial=terr.n_features)
    # On ne charge que le champ, et seulement les cles compatibles.
    prefixe = "spatial_encoder."
    partiel = {}
    for cle, v in sd.items():
        if not cle.startswith(prefixe):
            continue
        court = cle[len(prefixe):]
        cible = dict(reseau.named_parameters()).get(court)
        if cible is None or cible.shape == v.shape:
            partiel[court] = v
        elif cible.dim() == v.dim() and cible.shape[1:] == v.shape[1:]:
            bourre = torch.zeros_like(cible)
            bourre[:v.shape[0]] = v
            partiel[court] = bourre
    reseau.load_state_dict(partiel, strict=False)
    reseau.eval()
    with torch.no_grad():
        p = reseau(coords, terr.to_tensor())

    n_noeuds = coords.shape[0]
    etals = {}
    for nom in noms:
        v = getattr(p, nom, None)
        if not torch.is_tensor(v) or v.shape[:1] != (n_noeuds,):
            continue
        a = v.numpy().astype(float)
        med = np.median(a)
        etals[nom] = 100.0 * (np.quantile(a, .9) - np.quantile(a, .1)) / (abs(med) + 1e-12)
    constantes = [k2 for k2, e in etals.items() if e < 5.0]
    return {"sorties_ck": k, "sorties_champ": N, "rembourrees": rembourrees,
            "constantes": constantes, "etals": etals, "n_mesurees": len(etals)}


# ---------------------------------------------------------------------------
# Banc 3 : la platitude de l'hydrogramme simule
# ---------------------------------------------------------------------------

def platitude(chemin_npz, seuil=0.01):
    """Part de jours ou le debit simule varie de moins de `seuil` d'un jour a l'autre.

    Constat du 2026-09-03, a partir des hydrogrammes du rapport : le simule reste plat
    32 a 55 % du temps selon la region, contre 3 a 18 % pour l'observe, avec des suites
    de 54 a 123 jours consecutifs commencant en hiver. Le modele descend alors une
    recession exponentielle sans qu'aucun evenement ne s'y ajoute. Le classement des
    regions par platitude reproduit celui de leur echec (outm, abit, sagu, gasp en tete,
    mont en queue), ce qui en fait un indicateur utile : un correctif des processus
    hivernaux doit le faire baisser.
    """
    d = np.load(chemin_npz, allow_pickle=True)
    S, O = d["q_sim"].astype(float), d["q_obs"].astype(float)
    mois = np.array([int(str(x)[5:7]) for x in d["dates"]])
    hiver = np.isin(mois[:-1], [12, 1, 2, 3])
    ps, po, ph, pl = [], [], [], []
    for j in range(S.shape[1]):
        s, o = S[:, j], O[:, j]
        m = np.isfinite(s) & np.isfinite(o)
        if m.sum() < 200:
            continue
        cs = np.abs(np.diff(s)) / np.maximum(s[:-1], 1e-9) < seuil
        co = np.abs(np.diff(o)) / np.maximum(o[:-1], 1e-9) < seuil
        v = np.isfinite(s[:-1]) & np.isfinite(o[:-1])
        ps.append(cs[v].mean())
        po.append(co[v].mean())
        if (v & hiver).any():
            ph.append(cs[v & hiver].mean())
        n = mx = 0
        for c in cs:
            n = n + 1 if c else 0
            mx = max(mx, n)
        pl.append(mx)
    if not ps:
        return None
    return {"n_stations": len(ps), "plat_sim": 100 * float(np.median(ps)),
            "plat_obs": 100 * float(np.median(po)),
            "plat_hiver": 100 * float(np.nanmedian(ph)) if ph else float("nan"),
            "suite_max_j": int(np.median(pl))}


# ---------------------------------------------------------------------------

RETENUS = {"outv": "best-outv-etl-canon", "gasp": "best-gasp-etl-socle30",
           "sagu": "best-sagu-etl-socle30", "slno": "best-slno-etl-socle30",
           "mont": "best-mont-etl-gen1", "slso": "best-slso-etl-casr"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="outv")
    ap.add_argument("--retard-seul", action="store_true")
    ap.add_argument("--champ-seul", action="store_true")
    ap.add_argument("--ks", default="24,16,8,3,1",
                    help="constantes de Muskingum a balayer, en heures")
    a = ap.parse_args()
    reg = a.region.lower()
    echecs = []

    if not a.champ_seul:
        print(f"=== RETARD DU ROUTAGE ({reg}, impulsion dans le reseau reel, sans lacs) ===")
        print(f"{'K (h)':>7s} {'retard du centre':>17s} {'retard du pic':>14s}")
        ref = None
        for kh in [float(x) for x in a.ks.split(",")]:
            r = retard_routage(reg, kh)
            if r is None:
                print(f"{kh:7.1f}   pas de reponse a l'exutoire"); continue
            if ref is None:
                ref = r
                print(f"  reseau : {r['n']} troncons | plus long chemin "
                      f"{r['chemin_km']:.0f} km | temps de parcours physique a 1 m/s "
                      f"{r['physique_j']:.2f} j")
                print(f"{'K (h)':>7s} {'retard du centre':>17s} {'retard du pic':>14s}")
            print(f"{kh:7.1f} {r['retard_centre_j']:15.2f} j {r['retard_pic_j']:12d} j")
        if ref is not None:
            defaut = retard_routage(reg, 24.0)
            exces = defaut["retard_centre_j"] - defaut["physique_j"]
            print(f"  -> a K=24 h (l'initialisation du champ), le routage ajoute "
                  f"{exces:.2f} j au temps de parcours physique")
            if exces > 1.0:
                echecs.append("retard du routage")

    if not a.retard_seul:
        ck = f".runs/quebec/checkpoints/{RETENUS.get(reg, 'best-outv-etl-canon')}.pt"
        print(f"\n=== ETAT DU CHAMP ({reg}, {os.path.basename(ck)}) ===")
        e = etat_champ(reg, ck)
        print(f"  sorties du point de reprise : {e['sorties_ck']} pour "
              f"{e['sorties_champ']} attendues")
        if e["rembourrees"]:
            print(f"  REMBOURREES A ZERO ({len(e['rembourrees'])}), donc figees au milieu "
                  f"de leurs bornes : {', '.join(e['rembourrees'])}")
            echecs.append("sorties rembourrees")
        print(f"  constantes deguisees (etalement sous 5 %) : {len(e['constantes'])} sur "
              f"{e['n_mesurees']} mesurees")
        vivantes = sorted(((v, k) for k, v in e["etals"].items()), reverse=True)[:5]
        print("  les cinq sorties les plus variables : "
              + ", ".join(f"{k} {v:.0f} %" for v, k in vivantes))
        if len(e["constantes"]) > e["n_mesurees"] // 2:
            echecs.append("champ effondre")

    if not a.retard_seul and not a.champ_seul:
        npz = f"{_DATA}/quebec/rapport/rap-{reg}-q.npz"
        print(f"\n=== PLATITUDE DE L'HYDROGRAMME ({reg}) ===")
        if not os.path.exists(npz):
            print(f"  {npz} absent : lancer d'abord une passe avec ETL_DUMP_Q")
        else:
            p = platitude(npz)
            print(f"  {p['n_stations']} stations | simule plat {p['plat_sim']:.1f} % du "
                  f"temps contre {p['plat_obs']:.1f} % pour l'observe")
            print(f"  en hiver (decembre a mars) : {p['plat_hiver']:.1f} % | plus longue "
                  f"suite plate, mediane des stations : {p['suite_max_j']} jours")
            if p["plat_sim"] > 2.5 * p["plat_obs"]:
                echecs.append("hydrogramme plat")

    print()
    if echecs:
        print(f"DEFAUTS REPRODUITS : {', '.join(echecs)}")
        return 1
    print("Aucun des defauts connus ne se reproduit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
