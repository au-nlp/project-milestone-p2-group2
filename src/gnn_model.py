import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGEConv


class NavigationGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, model_type="SAGE"):
        super(NavigationGNN, self).__init__()
        self.model_type = model_type

        if model_type == "GAT":
            # GATv2: Attention (heads=2)
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=True, dropout=0.3)
            # Input to layer 2 is hidden * heads
            self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=False, dropout=0.3)
        else:
            # GraphSAGE: Aggregation
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Input dimension for predictor: 3 embeddings + 1 distance feature
        input_dim = 3 * hidden_channels + 1

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def get_node_embeddings(self, x, edge_index):
        """
        Computes node embeddings for the entire graph.
        Used once per evaluation epoch to save computation time.
        """
        if self.model_type == "GAT":
            h = self.conv1(x, edge_index)
            h = F.elu(h)
            h = self.conv2(h, edge_index)
        else:
            h = self.conv1(x, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.3, training=self.training)
            h = self.conv2(h, edge_index)
        return h

    def predict_link_score(self, h, src, cand, goal, dists):
        """
        Predicts score using pre-computed node embeddings 'h'.
        This method is called by the optimized evaluation function.
        """
        h_src = h[src]
        h_cand = h[cand]
        h_goal = h[goal]

        if dists.dim() == 1:
            dists = dists.unsqueeze(1)

        combined = torch.cat([h_src, h_cand, h_goal, dists], dim=1)
        return self.predictor(combined)

    def forward(self, x, edge_index, src, cand, goal, dists):
        """
        Standard forward pass (computes graph + prediction).
        Used during training loop.
        """
        h = self.get_node_embeddings(x, edge_index)
        return self.predict_link_score(h, src, cand, goal, dists)