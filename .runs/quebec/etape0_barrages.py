"""ETAPE 0 du dossier barrages : la regularisation contamine-t-elle les parametres appris ?

Question posee par le dossier notebooks/barrages, et c'est une PORTE : si aucune signature
ne ressort, le chantier reservoirs n'a pas de justification empirique et on s'arrete, quel
qu'en soit l'interet conceptuel.

L'HYPOTHESE, precisement. Vingt-sept des 178 stations d'entrainement portent plus de quinze
jours de debit moyen en stockage amont (`stations-regulees.csv`, script 08 du dossier).
Le signal de lachage est donc DEJA dans l'hydrogramme observe contre lequel meandre s'ajuste,
sans qu'aucun organe du modele puisse le porter. Le champ spatial n'a alors qu'une facon de
le reproduire : deformer les parametres qui gouvernent le temps de reponse du bassin. Une
retenue ecrete les pics et soutient l'etiage ; pour imiter ca sans reservoir, il faut rendre
le bassin plus lent et plus tampon. On attend donc, sur les troncons amont de ces stations :

  K_sat_1 plus BAS      (moins d'infiltration rapide, donc moins de pointe)
  manning_n plus HAUT   (canal plus rugueux, donc onde plus lente)
  f_vert_1 plus HAUT    (l'eau descend au lieu de partir en ruissellement lateral)
  interception plus HAUT

CE QUE LE TEST NE PEUT PAS FAIRE SEUL. Les bassins regularises ne sont pas un tirage au
hasard : ils sont plus grands, plus lacustres, et concentres dans SLNO et OUTV. Comparer
brutalement les 27 aux 151 autres mesurerait surtout cette difference de population. On
apparie donc chaque station regularisee a des temoins de la MEME plateforme et de superficie
comparable, et on ne lit que l'ecart intra-paire.

Sortie : un tableau par parametre, avec l'ecart apparie et son signe attendu.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".runs/quebec")
sys.path.insert(0, ".")

BARRAGES = os.environ.get("BARRAGES_DATA", "D:/meandre-data/barrages/data")
CKPT = os.environ.get("ETAPE0_CKPT", "D:/meandre-data/quebec/runpod/best-province.pt")
SEUIL_JOURS = float(os.environ.get("ETAPE0_SEUIL", "15"))
# parametre -> sens attendu si la regularisation est absorbee par le champ
ATTENDU = {"K_sat_1": "-", "K_sat_2": "-", "manning_n": "+", "f_vert_1": "+",
           "interception_capacity": "+", "C_f": "-", "krec": "+", "f_wetland": "+"}


def amont(edge_index, n_nodes, cibles):
    """Ensemble des noeuds en amont de chaque cible, par remontee du graphe."""
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    pred = {}
    for s, d in zip(src, dst):
        pred.setdefault(int(d), []).append(int(s))
    out = {}
    for c in cibles:
        vus, pile = set(), [int(c)]
        while pile:
            k = pile.pop()
            if k in vus:
                continue
            vus.add(k)
            pile.extend(pred.get(k, []))
        out[int(c)] = vus
    return out


def main():
    from domain_data import load_domain
    from meandre.model import HydroModel

    plats = (os.environ.get("ETAPE0_PLATEFORMES")
             or "outv,gasp,mont,sagu,slno,abit,slso,cnda,cndb,cndc,cndd,cnde,labi,vaud")
    noms = [p.strip() for p in plats.split(",") if p.strip()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dom = load_domain(noms, {}, device=dev)

    td = dom["train_data"]
    graph = td.graph
    n_nodes = int(dom["n_nodes"])
    st_idx = td.station_idx.cpu().numpy()
    st_ids = dom["station_ids"]
    node_coords = dom["node_coords"]
    terr = dom["territorial"]          # TerritorialFeatures, pas un tenseur
    territorial = terr.data
    print(f"[etape0] {n_nodes:,} troncons | {len(st_idx)} jauges | {len(noms)} plateformes")

    # ── le champ appris du champion ────────────────────────────────────────
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    modele = HydroModel(
        n_territorial=int(terr.n_features), n_nodes=n_nodes,
        use_latent_codes=True, spatial_melt=True).to(dev)
    modele.load_checkpoint(ck) if hasattr(modele, "load_checkpoint") else None
    sd = ck.get("state_dict", ck)
    own = modele.state_dict()
    mism = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    for k in mism:
        sd.pop(k)
    modele.load_state_dict(sd, strict=False)
    if mism:
        print(f"[etape0] {len(mism)} cles de forme incompatible ignorees : {mism[:3]}")
    modele.eval()
    with torch.no_grad():
        sp = modele.spatial_encoder(node_coords, territorial)
    champs = {k: getattr(sp, k).detach().cpu().numpy() for k in ATTENDU if hasattr(sp, k)}
    print(f"[etape0] champs lus : {', '.join(champs)}")

    # ── qui est regularise ─────────────────────────────────────────────────
    sr = pd.read_csv(f"{BARRAGES}/stations-regulees.csv")
    sr["sid"] = sr.station_id.astype(float).astype(int).astype(str)
    reg = dict(zip(sr.sid, sr.jours_de_debit))
    aire = dict(zip(sr.sid, sr.aire_km2))

    lignes = []
    for j, nidx in enumerate(st_idx):
        brut = st_ids[j] if st_ids else str(j)
        plat, _, s = str(brut).partition(":")
        s = str(int(float(s))) if s.replace(".", "").isdigit() else s
        if s not in reg:
            continue
        lignes.append({"j": j, "node": int(nidx), "plat": plat, "sid": s,
                       "jours": float(reg[s]), "aire": float(aire[s])})
    st = pd.DataFrame(lignes)
    if st.empty:
        print("[etape0] aucune jauge appariee au tableau des stations regularisees")
        return
    st["regule"] = st.jours > SEUIL_JOURS
    print(f"[etape0] {len(st)} jauges appariees | {int(st.regule.sum())} regularisees "
          f"(plus de {SEUIL_JOURS:.0f} jours de stockage amont)")
    print(st.groupby(["plat", "regule"]).size().unstack(fill_value=0).to_string())

    # ── moyenne des parametres sur le bassin amont de chaque station ───────
    ens = amont(graph.edge_index, n_nodes, st.node.tolist())
    for k, v in champs.items():
        st[k] = [float(np.mean(v[sorted(ens[int(n)])])) for n in st.node]

    # ── APPARIEMENT : meme plateforme, superficie la plus proche ───────────
    # Sans lui on mesurerait la difference de POPULATION : les bassins regularises sont
    # plus grands et concentres dans SLNO et OUTV.
    paires = []
    for _, r in st[st.regule].iterrows():
        cand = st[(~st.regule) & (st.plat == r.plat)]
        if cand.empty:
            continue
        c = cand.iloc[(np.log(cand.aire) - np.log(r.aire)).abs().argsort()[:3]]
        paires.append((r, c))
    print(f"\n[etape0] {len(paires)} stations regularisees appariees a leurs 3 temoins "
          f"les plus proches en superficie, sur la MEME plateforme")

    print(f"\n{'parametre':>22} | {'attendu':>7} | {'ecart median':>12} | "
          f"{'signe conforme':>14} | {'p (signe)':>9}")
    print("-" * 78)
    from scipy import stats as _st
    for k in champs:
        ecarts = [float(r[k]) - float(c[k].median()) for r, c in paires]
        e = np.array(ecarts)
        rel = np.median(e) / (np.median([float(c[k].median()) for _, c in paires]) + 1e-12)
        signe = "+" if np.median(e) > 0 else "-"
        conf = "oui" if signe == ATTENDU[k] else "non"
        p = _st.wilcoxon(e).pvalue if len(e) > 5 and np.any(e != 0) else float("nan")
        print(f"{k:>22} | {ATTENDU[k]:>7} | {100 * rel:>11.1f}% | "
              f"{conf:>14} | {p:>9.3f}")

    print("\nLECTURE. La colonne attendu porte le signe que la contamination PREDIT. Un ecart")
    print("conforme ET significatif sur plusieurs parametres du temps de reponse ouvre la")
    print("porte ; des signes disperses ou des p au-dessus de 0.05 la ferment, et le")
    print("chantier reservoirs perd sa justification empirique.")


if __name__ == "__main__":
    main()
