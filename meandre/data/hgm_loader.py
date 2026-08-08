"""Chargeur du cache HGM d'Hydrotel (hydrogramme géomorphologique par UHRH).

Le fichier `<projet>/hgm/hydrogramme_24H_*.hgm` contient, pour chaque UHRH, les poids
DISTRI 1..L de l'hydrogramme unitaire précalculé par l'onde cinématique pixel par pixel
d'Hydrotel (onde_cinematique.cpp::CalculeHgm). C'est le noyau d'étalement temporel que
le C++ applique à TOUTE la production (surface + hypodermique + base). Méandre livrait
l'eau du jour même : les têtes de bassin décorrélaient (r 0.27-0.31 quotidien contre
Hydrotel, test réseau 2026-08-08).

`lire_hgm(project_dir, node_ids)` retourne un noyau (n_nodes, L) normalisé ligne à
ligne : agrégation UHRH -> tronçon pondérée par l'aire (même convention que
hydrotel_calib), les lignes vides (UHRH-lacs, réponse instantanée dans le C++)
deviennent un Dirac au jour 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def lire_hgm(project_dir: str, node_ids, sim_subdir: str = "simulation/simulation"):
    proj = Path(project_dir)
    fics = sorted((proj / "hgm").glob("*.hgm"))
    if not fics:
        raise FileNotFoundError(f"aucun .hgm dans {proj / 'hgm'}")
    lignes = [l.strip() for l in fics[0].read_text(encoding="latin-1").splitlines()]
    # section DISTRI : après la ligne d'en-tête "UHRH;DISTRI 1;..."
    i0 = next(i for i, l in enumerate(lignes) if l.startswith("UHRH;DISTRI"))
    distri = {}
    L = 0
    for l in lignes[i0 + 1:]:
        if not l or ";" not in l:
            continue
        t = l.split(";")
        try:
            uid = int(t[0])
        except ValueError:
            break
        v = np.array([float(x) for x in t[1:] if x != ""], dtype=np.float64)
        distri[uid] = v
        L = max(L, len(v))

    # appartenance UHRH -> tronçon + aires (mêmes fichiers que hydrotel_calib)
    from meandre.data.physitel_loader import _parse_troncon
    tr = _parse_troncon(proj / "physitel" / "troncon.trl")
    uh = pd.read_csv(proj / "physitel" / "uhrh.csv", sep=";", skiprows=1)
    col_id = uh.columns[0]
    col_aire = next((c for c in uh.columns if "aire" in c.lower() or "area" in c.lower()), None)
    aires = dict(zip(uh[col_id].astype(int), uh[col_aire].astype(float))) if col_aire else {}

    par_id = {t["id"]: t for t in tr}
    n = len(node_ids)
    K = np.zeros((n, L), dtype=np.float32)
    for j, nid in enumerate(node_ids):
        t = par_id.get(int(nid))
        acc = np.zeros(L)
        wtot = 0.0
        if t is not None:
            for uid in t["uhrh_ids"]:
                d = distri.get(abs(int(uid)))
                if d is None or d.sum() <= 0:
                    continue
                w = aires.get(abs(int(uid)), 1.0)
                acc[:len(d)] += w * (d / d.sum())
                wtot += w
        if wtot > 0:
            K[j] = (acc / wtot).astype(np.float32)
        else:
            K[j, 0] = 1.0   # UHRH-lac ou absent : réponse instantanée, comme le C++
    return K
