# Visualization functions for graphs & attention


import networkx as nx
import matplotlib.pyplot as plt
import torch
from typing import Optional

def visualize_graph_nx(graph: nx.Graph, title: str = "Graph"):
    """
    Visualize a NetworkX graph using Matplotlib.
    """
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def visualize_tensor_heatmap(tensor: torch.Tensor, title: str = "Tensor Heatmap"):
    """
    Visualize a PyTorch tensor as a heatmap.
    """
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # Convert to 2D
    plt.figure(figsize=(10, 4))
    plt.imshow(tensor.detach().cpu().numpy(), aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Samples" if tensor.size(0) > 1 else "")
    plt.show()

def visualize_attention_matrix(attn_matrix: torch.Tensor, title: str = "Attention Matrix"):
    """
    Visualize attention matrix from transformer models.
    """
    assert attn_matrix.dim() == 2, "Attention matrix must be 2D"
    plt.figure(figsize=(6, 5))
    plt.imshow(attn_matrix.detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.show()
