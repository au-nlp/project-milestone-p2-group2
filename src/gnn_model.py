import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class NavigationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NavigationGNN, self).__init__()

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=True, dropout=0.4)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=False, dropout=0.4)

        self.predictor = nn.Sequential(
            nn.Linear(3 * hidden_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, src_idx, cand_idx, goal_idx):
        h = self.get_node_embeddings(x, edge_index)
        return self.predict_link_score(h, src_idx, cand_idx, goal_idx)

    def get_node_embeddings(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.conv2(h, edge_index)
        return h

    def predict_link_score(self, h, src_idx, cand_idx, goal_idx):
        h_src = h[src_idx]
        h_cand = h[cand_idx]
        h_goal = h[goal_idx]
        combined = torch.cat([h_src, h_cand, h_goal], dim=1)
        return self.predictor(combined)