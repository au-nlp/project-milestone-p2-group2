"""
Model implementations for next-link prediction.

This module contains:
- BaselineModel: Semantic-only scoring based on cosine similarity to goal
- HeuristicModel: Combines semantic similarity with hub bias (outdegree)
- GNNModel: Graph Neural Network for learning complex feature interactions
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Dict, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F


class BaselineModel(BaseEstimator):
    """
    Baseline model that scores candidates purely based on semantic similarity to the goal.
    
    This tests the hypothesis that humans choose articles semantically closer to their goal.
    Score = cosine_similarity(embedding(candidate), embedding(goal))
    """
    
    def __init__(self):
        self.embedding_map = None
        self.goal_embeddings = None
        
    def fit(self, X, y=None, embedding_map=None):
        """
        Fit the model (baseline doesn't need training, but we store embeddings).
        
        Args:
            X: Not used (for sklearn compatibility)
            y: Not used (for sklearn compatibility)
            embedding_map: Dictionary mapping article IDs to embeddings
        """
        self.embedding_map = embedding_map
        return self
    
    def predict_proba(self, X, query_info=None):
        """
        Predict probabilities based on semantic similarity to goal.
        
        Args:
            X: Not used (for sklearn compatibility)
            query_info: DataFrame with columns ['query_id', 'candidate_id', 'goal_id']
        
        Returns:
            Array of shape (n_samples, 2) with [prob_negative, prob_positive]
        """
        if query_info is None:
            raise ValueError("query_info is required for BaselineModel")
        
        scores = []
        for _, row in query_info.iterrows():
            candidate_id = row['candidate_id']
            goal_id = row['goal_id']
            
            emb_candidate = self.embedding_map.get(candidate_id, np.zeros(384))
            emb_goal = self.embedding_map.get(goal_id, np.zeros(384))
            
            # Compute cosine similarity (optimized numpy version)
            def normalize(v):
                norm = np.linalg.norm(v)
                return v / (norm + 1e-8) if norm > 0 else v
            
            emb_candidate_norm = normalize(emb_candidate)
            emb_goal_norm = normalize(emb_goal)
            sim = np.dot(emb_candidate_norm, emb_goal_norm)
            
            # Convert similarity to probability-like score (normalize to [0, 1])
            # Similarity is in [-1, 1], so we shift and scale
            score = (sim + 1) / 2
            scores.append(score)
        
        scores = np.array(scores)
        # Return as [prob_negative, prob_positive]
        return np.column_stack([1 - scores, scores])


class HeuristicModel(BaseEstimator):
    """
    Heuristic model that combines semantic similarity with hub bias.
    
    Score = α * cos_sim(candidate, goal) + (1-α) * (outdegree(candidate) / max_outdegree)
    
    The parameter α can be dynamic based on path position:
    - Early in path: lower α (more weight on hubs)
    - Later in path: higher α (more weight on semantic similarity)
    """
    
    def __init__(self, alpha=0.7, dynamic_alpha=True):
        """
        Args:
            alpha: Weight for semantic similarity (1-alpha for hub bias)
            dynamic_alpha: If True, adjust alpha based on path position
        """
        self.alpha = alpha
        self.dynamic_alpha = dynamic_alpha
        self.embedding_map = None
        self.out_degree_map = None
        self.max_outdegree = None
        
    def fit(self, X, y=None, embedding_map=None, out_degree_map=None):
        """
        Fit the model.
        
        Args:
            X: Not used (for sklearn compatibility)
            y: Not used (for sklearn compatibility)
            embedding_map: Dictionary mapping article IDs to embeddings
            out_degree_map: Dictionary mapping article IDs to out-degree
        """
        self.embedding_map = embedding_map
        self.out_degree_map = out_degree_map
        if out_degree_map:
            self.max_outdegree = max(out_degree_map.values()) if out_degree_map.values() else 1.0
        return self
    
    def _get_alpha(self, path_position=None, path_length=None):
        """Get alpha value, potentially adjusted based on path position."""
        if not self.dynamic_alpha or path_position is None or path_length is None:
            return self.alpha
        
        # Early in path: lower alpha (more hub bias)
        # Later in path: higher alpha (more semantic similarity)
        position_ratio = path_position / max(path_length, 1)
        # Linear interpolation: start at 0.3, end at 0.9
        dynamic_alpha = 0.3 + 0.6 * position_ratio
        return dynamic_alpha
    
    def predict_proba(self, X, query_info=None):
        """
        Predict probabilities based on semantic similarity and hub bias.
        
        Args:
            X: Not used (for sklearn compatibility)
            query_info: DataFrame with columns ['query_id', 'candidate_id', 'goal_id', 'path_position', 'path_length']
        
        Returns:
            Array of shape (n_samples, 2) with [prob_negative, prob_positive]
        """
        if query_info is None:
            raise ValueError("query_info is required for HeuristicModel")
        
        scores = []
        for _, row in query_info.iterrows():
            candidate_id = row['candidate_id']
            goal_id = row['goal_id']
            path_position = row.get('path_position', None)
            path_length = row.get('path_length', None)
            
            # Get alpha (potentially dynamic)
            alpha = self._get_alpha(path_position, path_length)
            
            # Semantic component (optimized numpy version)
            emb_candidate = self.embedding_map.get(candidate_id, np.zeros(384))
            emb_goal = self.embedding_map.get(goal_id, np.zeros(384))
            
            def normalize(v):
                norm = np.linalg.norm(v)
                return v / (norm + 1e-8) if norm > 0 else v
            
            emb_candidate_norm = normalize(emb_candidate)
            emb_goal_norm = normalize(emb_goal)
            sim = np.dot(emb_candidate_norm, emb_goal_norm)
            semantic_score = (sim + 1) / 2  # Normalize to [0, 1]
            
            # Hub component
            outdegree = self.out_degree_map.get(candidate_id, 0)
            hub_score = outdegree / max(self.max_outdegree, 1.0)
            
            # Combined score
            score = alpha * semantic_score + (1 - alpha) * hub_score
            scores.append(score)
        
        scores = np.array(scores)
        # Return as [prob_negative, prob_positive]
        return np.column_stack([1 - scores, scores])


def build_graph_from_links(
    links_df: pd.DataFrame,
    article_id_to_index: Dict[str, int],
    embedding_map: Dict[str, np.ndarray],
    embedding_dim: int = 384
) -> Data:
    """
    Build a PyTorch Geometric Data object from the Wikipedia links.
    
    Args:
        links_df: DataFrame with 'source' and 'target' columns
        article_id_to_index: Dictionary mapping article IDs to node indices
        embedding_map: Dictionary mapping article IDs to embeddings
        embedding_dim: Dimension of embeddings (default 384 for SBERT)
    
    Returns:
        PyTorch Geometric Data object with node features and edge indices
    """
    num_nodes = len(article_id_to_index)
    
    # Create edge index (COO format: [2, num_edges])
    edge_list = []
    for _, row in links_df.iterrows():
        source_id = row['source']
        target_id = row['target']
        
        if source_id in article_id_to_index and target_id in article_id_to_index:
            source_idx = article_id_to_index[source_id]
            target_idx = article_id_to_index[target_id]
            edge_list.append([source_idx, target_idx])
    
    if len(edge_list) == 0:
        raise ValueError("No valid edges found!")
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create node features from embeddings
    node_features = []
    for article_id, idx in sorted(article_id_to_index.items(), key=lambda x: x[1]):
        embedding = embedding_map.get(article_id, np.zeros(embedding_dim))
        node_features.append(embedding)
    
    x = torch.tensor(np.array(node_features), dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    
    return data


class GNNDataset(Dataset):
    """
    Dataset class for GNN training.
    
    Stores (source_idx, candidate_idx, goal_idx, label) tuples.
    """
    
    def __init__(
        self,
        query_info_df: pd.DataFrame,
        article_id_to_index: Dict[str, int],
        y_labels: pd.Series,
        features: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ):
        """
        Args:
            query_info_df: DataFrame with source_id, candidate_id, and goal_id columns
            article_id_to_index: Dictionary mapping article IDs to indices
            y_labels: Series with binary labels
        """
        self.query_info = query_info_df.reset_index(drop=True)
        self.y_labels = y_labels.reset_index(drop=True)
        self.article_id_to_index = article_id_to_index
        if features is not None:
            features_np = np.asarray(features, dtype=np.float32)
            if len(features_np) != len(self.query_info):
                raise ValueError("features length must match query_info length")
            self._raw_features = features_np
        else:
            self._raw_features = None
        
        # Convert article IDs to indices
        self.source_indices = []
        self.candidate_indices = []
        self.goal_indices = []
        self.labels = []
        extra_features = [] if self._raw_features is not None else None
        
        for idx in range(len(self.query_info)):
            row = self.query_info.iloc[idx]
            source_id = row.get('source_id', None)
            candidate_id = row['candidate_id']
            goal_id = row['goal_id']
            
            source_idx = self.article_id_to_index.get(source_id, -1) if source_id else -1
            candidate_idx = self.article_id_to_index.get(candidate_id, -1)
            goal_idx = self.article_id_to_index.get(goal_id, -1)
            
            if candidate_idx != -1 and goal_idx != -1:
                self.source_indices.append(source_idx)
                self.candidate_indices.append(candidate_idx)
                self.goal_indices.append(goal_idx)
                self.labels.append(self.y_labels.iloc[idx])
                if extra_features is not None:
                    extra_features.append(self._raw_features[idx])
        
        self.source_indices = torch.tensor(self.source_indices, dtype=torch.long)
        self.candidate_indices = torch.tensor(self.candidate_indices, dtype=torch.long)
        self.goal_indices = torch.tensor(self.goal_indices, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float)
        self.extra_features = (
            torch.tensor(np.array(extra_features), dtype=torch.float) if extra_features is not None else None
        )
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = {
            'source_idx': self.source_indices[idx],
            'candidate_idx': self.candidate_indices[idx],
            'goal_idx': self.goal_indices[idx],
            'label': self.labels[idx]
        }
        if self.extra_features is not None:
            sample['features'] = self.extra_features[idx]
        return sample


class GNNModel(nn.Module):
    """
    Graph Neural Network model for next-link prediction.
    
    Uses GraphSAGE (SAGEConv) to learn node representations that combine graph structure
    and semantic features (SBERT embeddings).
    """
    
    def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, dropout=0.1, extra_feature_dim=0):
        """
        Args:
            input_dim: Dimension of node features (SBERT embeddings = 384)
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.extra_feature_dim = extra_feature_dim
        
        # GraphSAGE (SAGEConv) layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        
        self.dropout = dropout
        
        # Final prediction layer
        # Input: concatenated features [source_emb, candidate_emb, goal_emb, engineered_features]
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3 + self.extra_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data, source_idx, candidate_idx, goal_idx, extra_features=None):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with graph structure
            source_idx: Indices of source nodes (can be -1 if not available)
            candidate_idx: Indices of candidate nodes
            goal_idx: Indices of goal nodes
        
        Returns:
            Predicted probabilities for each (source, candidate, goal) triple
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Get representations for source, candidate, and goal nodes
        # Handle case where source_idx might be -1 (not available)
        valid_source_mask = source_idx >= 0
        if valid_source_mask.any():
            # Get valid source embeddings
            valid_source_indices = source_idx[valid_source_mask]
            source_emb_valid = x[valid_source_indices]
            
            # Create full source embedding tensor
            source_emb = torch.zeros(len(source_idx), x.size(1), device=x.device)
            source_emb[valid_source_mask] = source_emb_valid
        else:
            # All sources invalid, use zero vectors
            source_emb = torch.zeros(len(source_idx), x.size(1), device=x.device)
        
        candidate_emb = x[candidate_idx]
        goal_emb = x[goal_idx]
        
        # Concatenate features
        combined = torch.cat([source_emb, candidate_emb, goal_emb], dim=1)
        if extra_features is not None:
            combined = torch.cat([combined, extra_features], dim=1)
        
        # Predict logits
        logits = self.predictor(combined)
        return logits.squeeze()


def train_gnn_model(
    model: GNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    graph_data: Data,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 0.001,
    pos_weight: Optional[torch.Tensor] = None
) -> Tuple[GNNModel, List[float], List[float]]:
    """
    Train the GNN model.
    
    Args:
        model: GNNModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        graph_data: PyTorch Geometric Data object with graph structure
        device: torch device (cpu or cuda)
        num_epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Trained model, training losses, validation losses
    """
    model = model.to(device)
    graph_data = graph_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            source_idx = batch['source_idx'].to(device)
            candidate_idx = batch['candidate_idx'].to(device)
            goal_idx = batch['goal_idx'].to(device)
            labels = batch['label'].to(device)
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            optimizer.zero_grad()
            logits = model(graph_data, source_idx, candidate_idx, goal_idx, features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                source_idx = batch['source_idx'].to(device)
                candidate_idx = batch['candidate_idx'].to(device)
                goal_idx = batch['goal_idx'].to(device)
                labels = batch['label'].to(device)
                features = batch.get('features')
                if features is not None:
                    features = features.to(device)
                
                logits = model(graph_data, source_idx, candidate_idx, goal_idx, features)
                loss = criterion(logits, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model, train_losses, val_losses


class GNNWrapper(BaseEstimator):
    """
    Wrapper class to make GNNModel compatible with sklearn-style evaluation.
    """
    
    def __init__(
        self,
        model: GNNModel,
        graph_data: Data,
        device: torch.device,
        article_id_to_index: Dict,
        features: Optional[Union[pd.DataFrame, np.ndarray]] = None
    ):
        self.model = model
        self.graph_data = graph_data
        self.device = device
        self.article_id_to_index = article_id_to_index
        if features is not None:
            self.features_tensor = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float, device=device)
        else:
            self.features_tensor = None
    
    def predict_proba(self, X, query_info=None):
        """
        Predict probabilities for ranking.
        
        Args:
            X: Not used (for sklearn compatibility)
            query_info: DataFrame with candidate_id and goal_id columns
        
        Returns:
            Array of shape (n_samples, 2) with [prob_negative, prob_positive]
        """
        if query_info is None:
            raise ValueError("query_info is required for GNNWrapper")
        
        self.model.eval()
        scores = []
        feature_tensor = self.features_tensor
        
        with torch.no_grad():
            for idx, row in enumerate(query_info.itertuples(index=False)):
                source_id = getattr(row, 'source_id', None)
                candidate_id = row.candidate_id
                goal_id = row.goal_id
                
                source_idx = self.article_id_to_index.get(source_id, -1) if source_id else -1
                candidate_idx = self.article_id_to_index.get(candidate_id, -1)
                goal_idx = self.article_id_to_index.get(goal_id, -1)
                
                if candidate_idx == -1 or goal_idx == -1:
                    scores.append(0.0)
                    continue
                
                source_idx_tensor = torch.tensor([source_idx], device=self.device)
                candidate_idx_tensor = torch.tensor([candidate_idx], device=self.device)
                goal_idx_tensor = torch.tensor([goal_idx], device=self.device)
                extra_feat_tensor = None
                if feature_tensor is not None:
                    extra_feat_tensor = feature_tensor[idx].unsqueeze(0)
                
                logits = self.model(self.graph_data, source_idx_tensor, candidate_idx_tensor, goal_idx_tensor, extra_feat_tensor)
                prob = torch.sigmoid(logits)
                scores.append(prob.cpu().item())
        
        scores = np.array(scores)
        return np.column_stack([1 - scores, scores])

