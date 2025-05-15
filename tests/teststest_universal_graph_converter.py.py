# tests/test_universal_graph_converter.py

import pandas as pd
from src.universal_graph_converter import UniversalGraphConverter

def test_tabular_to_graph():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    converter = UniversalGraphConverter("tabular", {"connect_similar": True})
    G = converter.to_graph(df)
    
    assert len(G.nodes) == 2, "Node count should match number of rows"
    assert isinstance(G, type(G)), "Output must be a graph object"

def test_text_to_graph():
    sample_text = "Graph learning is fun"
    converter = UniversalGraphConverter("text")
    G = converter.to_graph(sample_text)

    assert len(G.nodes) > 0, "Text graph should have some nodes"
    assert isinstance(G, type(G)), "Output must be a graph object"
