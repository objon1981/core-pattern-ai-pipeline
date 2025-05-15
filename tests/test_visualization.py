import torch
import networkx as nx
from src.visualization import (
    visualize_graph_nx,
    visualize_tensor_heatmap,
    visualize_attention_matrix,
)

def test_visualize_graph_nx():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    visualize_graph_nx(G, title="Test Graph")

def test_visualize_tensor_heatmap():
    tensor = torch.randn(10, 5)
    visualize_tensor_heatmap(tensor, title="Test Heatmap")

def test_visualize_attention_matrix():
    attn = torch.rand(5, 5)
    visualize_attention_matrix(attn, title="Test Attention Matrix")
