"""Tests for RiverGraph construction."""

import torch
from meandre.routing.graph import synthetic_linear_graph
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
    # 0 and 1 must come before 2, 2 before 3
    order_list = order.tolist()
    assert order_list.index(2) > order_list.index(0)
    assert order_list.index(2) > order_list.index(1)
    assert order_list.index(3) > order_list.index(2)
