import torch
import networkx as nx
import pytest
from src.core_pattern_learner import CorePatternLearner

def create_sample_graph():
    G = nx.Graph()
    G.add_node(0, features=torch.tensor([1.0, 0.0]))
    G.add_node(1, features=torch.tensor([0.0, 1.0]))
    G.add_edge(0, 1)
    return G

def test_model_forward():
    model = CorePatternLearner(in_features=2, hidden_dim=4, num_classes=2)
    G = create_sample_graph()

    x = torch.stack([G.nodes[n]["features"] for n in G.nodes()])
    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    out = model(x, edge_index)

    assert out.shape == (2, 2), "Output shape should match (num_nodes, num_classes)"
    assert torch.is_tensor(out), "Output should be a torch tensor"

def test_attention_weights_extraction():
    model = CorePatternLearner(in_features=2, hidden_dim=4, num_classes=2)
    G = create_sample_graph()

    x = torch.stack([G.nodes[n]["features"] for n in G.nodes()])
    edge_index = torch.tensor(list(G.edges())).t().contiguous()

    _ = model(x, edge_index)
    attn_weights = model.get_attention_weights()

    assert attn_weights is not None, "Attention weights should not be None"
    assert isinstance(attn_weights, torch.Tensor), "Attention weights should be a tensor"
