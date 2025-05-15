# src/universal_graph_converter.py

"""
UniversalGraphConverter:
A modular class to convert different data types into graph representations
Supports: tabular, text, image, audio, video, and knowledge graphs.
"""

import networkx as nx
import numpy as np
from nltk import word_tokenize
from nltk.util import bigrams
from skimage.segmentation import slic
from torch_geometric.utils import from_networkx
import pywt
import dgl


class UniversalGraphConverter:
    def __init__(self, data_type: str, config: dict = None):
        self.data_type = data_type
        self.config = config or {}

    def to_graph(self, data):
        if self.data_type == "tabular":
            return self._convert_tabular(data)
        elif self.data_type == "text":
            return self._convert_text(data)
        elif self.data_type == "image":
            return self._convert_image(data)
        elif self.data_type == "audio":
            return self._convert_audio(data)
        elif self.data_type == "video":
            return self._convert_video(data)
        elif self.data_type == "knowledge":
            return self._convert_knowledge(data)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    # --- A. Tabular Data ---
    def _convert_tabular(self, df):
        G = nx.Graph()
        for idx, row in df.iterrows():
            G.add_node(idx, **row.to_dict())

        # Optional edge construction via similarity
        if self.config.get("connect_similar"):
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(df.values)
            for i in range(len(df)):
                for j in range(i + 1, len(df)):
                    if similarities[i][j] > self.config.get("similarity_threshold", 0.9):
                        G.add_edge(i, j, weight=similarities[i][j])
        return G

    # --- B. Text Data ---
    def _convert_text(self, text):
        words = word_tokenize(text.lower())
        G = nx.Graph()
        for w1, w2 in bigrams(words):
            G.add_edge(w1, w2)
        return G

    # --- C. Image Data ---
    def _convert_image(self, image_array):
        segments = slic(image_array, n_segments=self.config.get("segments", 100))
        G = nx.grid_2d_graph(*segments.shape)
        for (i, j), node in np.ndenumerate(segments):
            G.nodes[(i, j)]["segment"] = int(node)
            G.nodes[(i, j)]["color"] = image_array[i, j].tolist()
        return G

    # --- D. Audio Data ---
    def _convert_audio(self, signal_array):
        coeffs = pywt.wavedec(signal_array, 'db1', level=self.config.get("level", 3))
        G = nx.Graph()
        for i, coeff in enumerate(coeffs):
            G.add_node(i, values=coeff.tolist())
            if i > 0:
                G.add_edge(i - 1, i)
        return G

    # --- E. Video Placeholder ---
    def _convert_video(self, video_data):
        # Placeholder: Extract frames, compute similarities, build graph
        G = nx.Graph()
        return G

    # --- F. Knowledge Graph Placeholder ---
    def _convert_knowledge(self, triples):
        # Expects triples like [(head, relation, tail)]
        G = nx.MultiDiGraph()
        for h, r, t in triples:
            G.add_edge(h, t, relation=r)
        return G

    # --- Export Options ---
    def export_to_dgl(self, nx_graph):
        return dgl.from_networkx(nx_graph)

    def export_to_pyg(self, nx_graph):
        return from_networkx(nx_graph)
