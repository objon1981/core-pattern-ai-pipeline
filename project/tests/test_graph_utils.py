# tests/test_graph_utils.py

import unittest
import networkx as nx
from src import graph_utils

class TestGraphUtils(unittest.TestCase):

    def setUp(self):
        self.G = nx.Graph()
        self.G.add_nodes_from([
            (0, {"value": 1.0}),
            (1, {"value": 2.0}),
            (2, {"value": 3.0})
        ])
        self.G.add_edges_from([(0, 1), (1, 2)])

    def test_normalize_node_attributes(self):
        normed = graph_utils.normalize_node_attributes(self.G, "value")
        self.assertIn("value_norm", normed.nodes[0])

    def test_prune_low_degree_nodes(self):
        pruned = graph_utils.prune_low_degree_nodes(self.G.copy(), min_degree=2)
        self.assertTrue(1 in pruned.nodes)
        self.assertFalse(0 in pruned.nodes)

    def test_convert_to_dgl(self):
        dgl_graph = graph_utils.convert_to_dgl(self.G)
        self.assertTrue(hasattr(dgl_graph, 'ndata'))

    def test_convert_to_pyg(self):
        pyg_graph = graph_utils.convert_to_pyg(self.G)
        self.assertTrue(hasattr(pyg_graph, 'edge_index'))

    def test_compute_graph_statistics(self):
        stats = graph_utils.compute_graph_statistics(self.G)
        self.assertEqual(stats["num_nodes"], 3)
        self.assertEqual(stats["num_edges"], 2)
        self.assertIn("avg_degree", stats)

if __name__ == '__main__':
    unittest.main()