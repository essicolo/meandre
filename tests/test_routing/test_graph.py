"""Tests for RiverGraph construction."""

import pytest
import torch

from meandre.routing.graph import synthetic_linear_graph, RiverGraph
from meandre.data.graph_builder import _topological_sort


def test_linear_graph_topo_order():
    graph = synthetic_linear_graph(5)
    assert list(graph.topo_order.numpy()) == [0, 1, 2, 3, 4]


def test_linear_graph_upstream():
    graph = synthetic_linear_graph(4)
    assert graph.upstream_nodes(0) == []
    assert graph.upstream_nodes(1) == [0]
    assert graph.upstream_nodes(3) == [2]


def test_topological_sort_simple():
    # 0->2, 1->2, 2->3
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long)
    order = _topological_sort(edge_index, 4)
    order_list = order.tolist()
    assert order_list.index(2) > order_list.index(0)
    assert order_list.index(2) > order_list.index(1)
    assert order_list.index(3) > order_list.index(2)


# ---------------------------------------------------------------------------
# RiverGraph cycle / consistency validation (via __post_init__)
# ---------------------------------------------------------------------------

def test_graph_raises_on_cycle():
    """A graph with a cycle must raise ValueError."""
    # 0->1->2->0 (cycle)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.ones(3, 3)
    topo_order = torch.tensor([0, 1, 2], dtype=torch.long)
    is_lake = torch.zeros(3, dtype=torch.bool)
    travel_time_days = torch.ones(3, dtype=torch.long)

    with pytest.raises(ValueError, match="cycle"):
        RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)


def test_graph_raises_on_out_of_range_node():
    """edge_index referencing a node >= n_nodes must raise ValueError."""
    # node 99 does not exist
    edge_index = torch.tensor([[0, 99], [1, 2]], dtype=torch.long)
    edge_attr = torch.ones(2, 3)
    topo_order = torch.tensor([0, 1, 2], dtype=torch.long)
    is_lake = torch.zeros(3, dtype=torch.bool)
    travel_time_days = torch.ones(2, dtype=torch.long)

    with pytest.raises(ValueError, match=">= n_nodes"):
        RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)


def test_graph_raises_on_negative_node():
    """edge_index with negative indices must raise ValueError."""
    edge_index = torch.tensor([[0, -1], [1, 2]], dtype=torch.long)
    edge_attr = torch.ones(2, 3)
    topo_order = torch.tensor([0, 1, 2], dtype=torch.long)
    is_lake = torch.zeros(3, dtype=torch.bool)
    travel_time_days = torch.ones(2, dtype=torch.long)

    with pytest.raises(ValueError, match="negative"):
        RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)


def test_graph_raises_on_wrong_topo_order():
    """A graph with mismatched topo_order must raise ValueError."""
    # Linear graph 0->1->2 but reversed topo_order [2, 1, 0]
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.ones(2, 3)
    topo_order = torch.tensor([2, 1, 0], dtype=torch.long)
    is_lake = torch.zeros(3, dtype=torch.bool)
    travel_time_days = torch.ones(2, dtype=torch.long)

    with pytest.raises(ValueError, match="topo_order is inconsistent"):
        RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)


def test_graph_accepts_valid_graph():
    """A valid DAG must be accepted silently."""
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.ones(2, 3)
    topo_order = torch.tensor([0, 1, 2], dtype=torch.long)
    is_lake = torch.zeros(3, dtype=torch.bool)
    travel_time_days = torch.ones(2, dtype=torch.long)

    graph = RiverGraph(edge_index, edge_attr, topo_order, is_lake, travel_time_days)
    assert graph.n_nodes == 3
    assert graph.n_edges == 2