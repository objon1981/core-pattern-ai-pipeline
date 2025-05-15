# CorePatternLearner class with attention

# src/core_pattern_learner.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import os

class CorePatternLearner(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.3):
        super(CorePatternLearner, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

    def get_attention_weights(self, x, edge_index):
        _, alpha = self.gat1(x, edge_index, return_attention_weights=True)
        return alpha
    
    def save_model(self, filepath: str):
        """
        Save the model state dictionary and any other info.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            # Add other states if needed: optimizer, epoch, etc.
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """
        Load the model state dictionary from the given filepath.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model file found at {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded from {filepath}")
