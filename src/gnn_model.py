import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class NavigationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NavigationGNN, self).__init__()

        # --- 1. GNN Encoder (GraphSAGE) ---
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # --- 2. Decoder (Link Predictor) ---
        self.predictor = nn.Sequential(
            nn.Linear(3 * hidden_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # output = score
        )

    def forward(self, x, edge_index, src_idx, cand_idx, goal_idx):
        """
        x: Node features (text embeddings)
        edge_index: Graph structure (who is connected to whom)
        src_idx, cand_idx, goal_idx: Indices of the articles for which we make the prediction
        """

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)

        h_src = h[src_idx]
        h_cand = h[cand_idx]
        h_goal = h[goal_idx]

        combined = torch.cat([h_src, h_cand, h_goal], dim=1)

        return self.predictor(combined)