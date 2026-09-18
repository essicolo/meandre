"""Colonnes de descripteurs pour l'entrée du champ spatial, à partir des tables auxiliaires.

Une source ingérée donne une table par tronçon, avec une couverture par variable. Ici elle
devient des colonnes d'entrée du champ, selon deux règles.

Les compositions, groupes de colonnes qui partagent un préfixe et somment à un, passent en
log-ratios isométriques avec remplacement multiplicatif des zéros. Les variables simples sont
centrées et réduites sur les tronçons couverts de la province. Dans les deux cas la valeur est
multipliée par la couverture, si bien qu'un tronçon non cartographié reçoit la valeur neutre 0,
et la couverture elle-même est ajoutée comme masque.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from meandre.utils import paths as _paths


def _lire(source: str) -> tuple[pd.DataFrame, dict]:
    f = f"{_paths.DERIVED_ROOT}/auxiliaires/{source}-troncons.parquet"
    meta = pq.read_schema(f).metadata or {}
    info = json.loads(meta.get(b"meandre", b"{}") or b"{}")
    return pd.read_parquet(f), info


def _ilr(parts: np.ndarray) -> np.ndarray:
    import nuee

    return nuee.ilr(nuee.multiplicative_replacement(nuee.closure(parts)))


def columns_for_nodes(source: str, region: str, node_ids, compositions: dict[str, list[str]] | None = None, variables: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Colonnes d'une source pour les nœuds d'une région, dans l'ordre des nœuds.

    compositions : {préfixe: [classes]} ; les colonnes `<préfixe>_<classe>` forment une
    composition et leur couverture commune est `couv_<préfixe>_<première classe>`.
    variables : colonnes simples, chacune avec sa couverture `couv_<nom>`.
    Sans argument, tout ce que la table contient est pris : les préfixes à plusieurs classes
    comme compositions, le reste comme variables.
    """
    t, info = _lire(source)
    cols = [c for c in t.columns if not c.startswith("couv_") and c not in ("region", "troncon", "area_m2")]
    if compositions is None and variables is None:
        # Les compositions se lisent dans la recette de la source : une couche de classes
        # par entrée de `ingestion.couches`. Le reste de la table est pris en variables simples.
        from meandre.data.auxiliary.catalog import load_source

        ing = load_source(source).ingestion
        compositions = {}
        for couche in ing.get("couches", []):
            classes = [r["classe"] for r in couche.get("regles", [])] + [couche.get("classe_defaut", "autre")]
            compositions[couche["nom"]] = [k for k in classes if f"{couche['nom']}_{k}" in cols]
        variables = [c for c in cols if not any(c.startswith(p + "_") for p in compositions)]
    compositions = compositions or {}
    variables = variables or []
    m = t[t.region == region].set_index("troncon").reindex([int(i) for i in node_ids])
    sorties, noms = [], []
    for prefixe, classes in compositions.items():
        cs = [f"{prefixe}_{k}" for k in classes]
        couv = m[f"couv_{cs[0]}"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        parts = m[cs].fillna(0.0).to_numpy(dtype=float)
        ok = parts.sum(axis=1) > 0
        z = np.zeros((len(m), len(cs) - 1), dtype=np.float32)
        if ok.any():
            z[ok] = _ilr(parts[ok])
        # Centrage sur les tronçons couverts de la province, pour que 0 reste la valeur neutre.
        ref = t[[f"couv_{cs[0]}"]].to_numpy().ravel() >= 0.5
        if ref.any():
            zr = _ilr(np.clip(t.loc[ref, cs].fillna(0.0).to_numpy(dtype=float), 0, None) + 1e-12)
            z[ok] = (z[ok] - zr.mean(axis=0)) / np.maximum(zr.std(axis=0), 1e-6)
        sorties.append(z * couv[:, None])
        noms += [f"{source}_{prefixe}_ilr_{k + 1}" for k in range(len(cs) - 1)]
        sorties.append(couv[:, None].astype(np.float32))
        noms.append(f"{source}_{prefixe}_couverture")
    for v in variables:
        ref = t.loc[(t[f"couv_{v}"] >= 0.5) & t[v].notna(), v]
        couv = m[f"couv_{v}"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        z = ((m[v] - ref.mean()) / max(float(ref.std()), 1e-6)).fillna(0.0).to_numpy(dtype=np.float32)
        sorties.append((z * couv)[:, None])
        sorties.append(couv[:, None].astype(np.float32))
        noms += [f"{source}_{v}", f"{source}_{v}_couverture"]
    return np.hstack(sorties).astype(np.float32), noms
