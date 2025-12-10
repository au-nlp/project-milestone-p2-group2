import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class HybridGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_explicit_feats: int):
        super().__init__()
        
        # Layer 1: Preserve semantics
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        
        # Layer 2: Neighborhood context
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        
        self.feat_norm = nn.BatchNorm1d(num_explicit_feats)
        
        # Fusion Input: Source + Candidate + Goal + Explicit_Feats + CosineSim
        fusion_dim = (3 * hidden_channels) + num_explicit_feats + 1 

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Helper to compute node embeddings."""
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.normalize(h, p=2, dim=1)
        return h

    def forward(self, x, edge_index, src_idx, cand_idx, goal_idx, explicit_feats):
        # 1. GNN Pass
        h = self.forward_gnn(x, edge_index)
        
        # 2. Embedding Lookup
        h_src = h[src_idx]
        h_cand = h[cand_idx]
        h_goal = h[goal_idx]
        
        # 3. Dynamic Feature: Cosine Sim between Candidate and Goal
        cos_sim = F.cosine_similarity(h_cand, h_goal).unsqueeze(1)
        
        # 4. Normalize explicit features
        feats_norm = self.feat_norm(explicit_feats)
        
        # 5. Concatenate and Classify
        combined = torch.cat([h_src, h_cand, h_goal, feats_norm, cos_sim], dim=1)
        return self.classifier(combined)