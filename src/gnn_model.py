import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class HybridGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_explicit_feats):
        super().__init__()
        
        # Layer 1: Preserve SBERT semantics (384 -> 384)
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        
        # Layer 2: Refine neighborhood context
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        
        self.feat_norm = nn.BatchNorm1d(num_explicit_feats)
        
        # Fusion Input
        fusion_dim = (3 * hidden_channels) + num_explicit_feats + 1 

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_channels),
            nn.ReLU(),
            # Dropout 0.2 is the "sweet spot" for this dataset
            nn.Dropout(0.2), 
            # WIDER LAYER: 128 neurons (Your uploaded file had 64)
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_gnn(self, x, edge_index):
        """Helper to get node embeddings separately."""
        h = F.relu(self.conv1(x, edge_index))
        # Match training dropout
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.normalize(h, p=2, dim=1)
        return h

    def forward(self, x, edge_index, src, cand, goal, feats):
        # 1. GNN
        h = self.forward_gnn(x, edge_index)
        
        # 2. Lookup
        h_src = h[src]
        h_cand = h[cand]
        h_goal = h[goal]
        
        # 3. Explicit Features
        cos_sim = F.cosine_similarity(h_cand, h_goal).unsqueeze(1)
        feats_norm = self.feat_norm(feats)
        
        # 4. Classify
        combined = torch.cat([h_src, h_cand, h_goal, feats_norm, cos_sim], dim=1)
        return self.classifier(combined)