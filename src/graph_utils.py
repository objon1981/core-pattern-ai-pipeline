# Graph construction, compression/expansion

# src/graph_utils.py

import networkx as nx
import numpy as np
from torch_geometric.utils import from_networkx
import dgl

def normalize_node_attributes(G: nx.Graph, attr_name: str):
    """Normalize a given node attribute across all nodes."""
    values = np.array([G.nodes[n].get(attr_name, 0.0) for n in G.nodes])
    if values.std() == 0:
        return G
    norm_values = (values - values.mean()) / values.std()
    for idx, node in enumerate(G.nodes):
        G.nodes[node][attr_name + "_norm"] = norm_values[idx]
    return G

def prune_low_degree_nodes(G: nx.Graph, min_degree: int = 2):
    """Remove nodes below a certain degree threshold."""
    to_remove = [n for n in G.nodes if G.degree[n] < min_degree]
    G.remove_nodes_from(to_remove)
    return G

def convert_to_dgl(G: nx.Graph):
    """Convert networkx graph to DGL graph."""
    return dgl.from_networkx(G)

def convert_to_pyg(G: nx.Graph):
    """Convert networkx graph to PyTorch Geometric graph."""
    return from_networkx(G)

def compute_graph_statistics(G: nx.Graph):
    """Return basic statistics about the graph."""
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "avg_degree": float(np.mean([d for n, d in G.degree()])),
        "is_connected": nx.is_connected(G) if G.number_of_nodes() > 0 else False,
    }
