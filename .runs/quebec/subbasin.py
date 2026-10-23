"""Sous-bassin en amont d'une station : restriction des données d'un territoire.

Le pilote et le banc de sous-bassin construisaient deux modèles différents : le banc gardait
l'ancienne loi des ancrages (ETP de Linacre calée, calage de sol d'Hydrotel) quand le modèle
gelé apprend sa demande évaporative sur CaSR brut. Plutôt que d'entretenir deux
constructeurs, le pilote se restreint ici aux tronçons en amont d'une station : même
fichier de configuration, même construction, moins de tronçons (décision d'Essi,
2026-10-23).

`ETL_SUBBASIN=<territoire>:<station>` est un paramètre de lancement, comme ETL_REGION.
"""
from __future__ import annotations

import os
from collections import defaultdict, deque

import numpy as np
import torch

from meandre.routing.graph import RiverGraph
from meandre.routing.withdrawals import WithdrawalData
from meandre.spatial.territorial import TerritorialFeatures


def requested(reg: str) -> str | None:
    """Station demandée pour ce territoire, ou None."""
    v = os.environ.get("ETL_SUBBASIN")
    if not v:
        return None
    if ":" not in v:
        raise SystemExit(f"ETL_SUBBASIN={v!r} : la forme attendue est <territoire>:<station>")
    terr, station = v.split(":", 1)
    return station if terr.strip().lower() == reg.lower() else None


def upstream(graph: RiverGraph, outlet: int) -> np.ndarray:
    """Indices, triés, des tronçons en amont d'un tronçon, lui compris."""
    src = graph.edge_index[0].cpu().numpy()
    dst = graph.edge_index[1].cpu().numpy()
    parents = defaultdict(list)
    for s, t in zip(src, dst):
        parents[int(t)].append(int(s))
    seen, todo = {int(outlet)}, deque([int(outlet)])
    while todo:
        u = todo.popleft()
        for v in parents[u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return np.array(sorted(seen), dtype=np.int64)


def restrict_graph(graph: RiverGraph, keep: np.ndarray) -> RiverGraph:
    """Graphe réindexé sur les tronçons gardés, ordre topologique d'origine conservé."""
    rank = {int(v): k for k, v in enumerate(keep)}
    src = graph.edge_index[0].cpu().numpy()
    dst = graph.edge_index[1].cpu().numpy()
    kept = np.array([(int(s) in rank) and (int(t) in rank) for s, t in zip(src, dst)], dtype=bool)
    dev = graph.edge_index.device
    e_src = [rank[int(s)] for s in src[kept]]
    e_dst = [rank[int(t)] for t in dst[kept]]
    topo = [rank[int(v)] for v in graph.topo_order.cpu().numpy() if int(v) in rank]
    m = torch.as_tensor(kept, device=dev)
    return RiverGraph(
        edge_index=torch.tensor([e_src, e_dst], dtype=torch.long, device=dev).reshape(2, -1),
        edge_attr=graph.edge_attr[m],
        topo_order=torch.tensor(topo, dtype=torch.long, device=graph.topo_order.device),
        is_lake=graph.is_lake[torch.as_tensor(keep, device=graph.is_lake.device)],
        travel_time_days=graph.travel_time_days[m],
    )


def restrict_territorial(terr: TerritorialFeatures, keep: np.ndarray) -> TerritorialFeatures:
    n0 = terr.data.shape[0]
    k = torch.as_tensor(keep, device=terr.data.device)
    return TerritorialFeatures(
        data=terr.data[k],
        columns=list(terr.columns),
        physical={name: v[k.to(v.device)] for name, v in terr.physical.items() if torch.is_tensor(v) and v.shape[:1] == (n0,)},
    )


def restrict_withdrawals(w: WithdrawalData, keep: np.ndarray) -> WithdrawalData:
    k = torch.as_tensor(keep, device=w.net.device)
    return WithdrawalData(net=w.net[:, k], net_gw=(w.net_gw[:, k] if w.net_gw is not None else None))


def restrict_nodes(x, keep: np.ndarray):
    """Dernier axe d'une série (T, N) restreint aux tronçons gardés ; None reste None."""
    if x is None:
        return None
    return x[:, torch.as_tensor(keep, device=x.device)]
