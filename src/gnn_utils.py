import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple

# --- Dataset for GNN ---
class ListwiseWikiDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return (
            torch.tensor(item['src'], dtype=torch.long),
            torch.tensor(item['goal'], dtype=torch.long),
            torch.tensor(item['candidates'], dtype=torch.long),
            torch.tensor(item['features'], dtype=torch.float)
        )

# --- Shared Feature Computation ---
def compute_link_features(
    src_name: str, cand_name: str, src_idx: int, cand_idx: int, goal_idx: int,
    dist_matrix: np.ndarray, pagerank_map: Dict[str, float], 
    out_degree_map: Dict[str, int], link_position_map: Dict[str, Dict[str, float]], 
    max_deg: float
) -> List[float]:
    """Helper to compute the 5 manual features for a triplet."""
    # 1. Link Position
    src_pos_map = link_position_map.get(src_name, {})
    pos_feat = src_pos_map.get(cand_name, 1.0)

    # 2. PageRank
    pr_feat = pagerank_map.get(cand_name, 0.0)

    # 3. Out Degree (Normalized)
    deg_feat = out_degree_map.get(cand_name, 0.0) / max_deg

    # 4. Shortest Path Distance logic
    dist_c_g = dist_matrix[cand_idx, goal_idx]
    if np.isinf(dist_c_g): dist_c_g = 10.0
    
    dist_s_g = dist_matrix[src_idx, goal_idx]
    if np.isinf(dist_s_g): dist_s_g = 10.0
    
    # 5. Is Closer Binary
    is_closer = 1.0 if dist_c_g < dist_s_g else 0.0

    return [pos_feat, pr_feat, deg_feat, is_closer, dist_c_g]

# --- Logic for Logistic Regression Baseline ---
def create_feature_dataset(
    paths_subset_df, links_map, embedding_map, dist_matrix, 
    article_id_to_index, pagerank_map, out_degree_map, 
    link_position_map, extractor
):
    """
    Generates tabular features (X, y) for Logistic Regression.
    """
    X_features = []
    y_labels = []
    query_groups = []
    
    max_deg = max(out_degree_map.values()) if out_degree_map else 1.0

    for row in tqdm(paths_subset_df.itertuples(), total=len(paths_subset_df), desc="Processing Paths"):
        path_articles = row.path.split(';')
        goal_id = path_articles[-1]
        
        # We need the Goal Embedding for cosine similarity
        emb_goal = embedding_map.get(goal_id, np.zeros(384))
        goal_idx = article_id_to_index.get(goal_id, -1)

        for i in range(len(path_articles) - 1):
            source_id = path_articles[i]
            positive_target_id = path_articles[i+1]
            query_id = f"{row.Index}_{i}"

            all_candidates = links_map.get(source_id, [])
            if not all_candidates: continue 

            emb_source = embedding_map.get(source_id, np.zeros(384))
            source_idx = article_id_to_index.get(source_id, -1)
            
            if source_idx == -1 or goal_idx == -1: continue
                
            for candidate_id in all_candidates:
                if candidate_id not in article_id_to_index: continue
                
                is_positive = 1 if (candidate_id == positive_target_id) else 0
                candidate_idx = article_id_to_index.get(candidate_id, -1)
                
                # 1. Semantic Features (Cosine Sim)
                emb_candidate = embedding_map.get(candidate_id, np.zeros(384))
                sem_features = extractor.get_semantic_features(emb_source, emb_candidate, emb_goal)
                
                # 2. Structural Features (using helper)
                feats_list = compute_link_features(
                    source_id, candidate_id, source_idx, candidate_idx, goal_idx,
                    dist_matrix, pagerank_map, out_degree_map, link_position_map, max_deg
                )
                # Unpack list: [pos, pr, deg, closer, dist]
                
                all_features = {
                    **sem_features,
                    "link_position": feats_list[0],
                    "pagerank": feats_list[1],
                    "out_degree": feats_list[2] * max_deg, # Un-normalize for LogReg if preferred, or keep normalized
                    "is_closer": feats_list[3],
                    "dist_candidate_goal": feats_list[4],
                    "dist_source_goal": dist_matrix[source_idx, goal_idx] if not np.isinf(dist_matrix[source_idx, goal_idx]) else 10.0
                }
                
                X_features.append(all_features)
                y_labels.append(is_positive)
                query_groups.append(query_id)

    X_df = pd.DataFrame(X_features)
    y_series = pd.Series(y_labels, name="is_positive")
    query_series = pd.Series(query_groups, name="query_id")
    
    return X_df, y_series, query_series

# --- Logic for GNN Training Data ---
def prepare_hard_listwise_data(
    paths_df, article_id_to_idx, links_map, dist_matrix, 
    pagerank_map, out_degree_map, link_position_map, neg_samples=20
):
    dataset_items = []
    max_deg = max(out_degree_map.values()) if out_degree_map else 1.0
    all_names = list(article_id_to_idx.keys())
    all_feats_buffer = []

    for row in tqdm(paths_df.itertuples(), total=len(paths_df), desc="Prep GNN Data"):
        path = row.path.split(';')
        goal = path[-1]
        if goal not in article_id_to_idx: continue
        goal_idx = article_id_to_idx[goal]

        for i in range(len(path) - 1):
            src = path[i]
            true_target = path[i+1]
            if src not in article_id_to_idx or true_target not in article_id_to_idx: continue
            src_idx = article_id_to_idx[src]
            
            all_neighbors = links_map.get(src, [])
            valid_neighbors = [n for n in all_neighbors if n in article_id_to_idx]
            negatives = [n for n in valid_neighbors if n != true_target]
            
            if len(negatives) < neg_samples:
                needed = neg_samples - len(negatives)
                negatives.extend(random.choices(all_names, k=needed))
            elif len(negatives) > neg_samples:
                negatives = random.sample(negatives, neg_samples)
            
            candidates = [true_target] + negatives[:neg_samples]
            candidate_ids = [article_id_to_idx[n] for n in candidates]
            
            group_feats = []
            for c_id, c_name in zip(candidate_ids, candidates):
                fv = compute_link_features(
                    src, c_name, src_idx, c_id, goal_idx,
                    dist_matrix, pagerank_map, out_degree_map, link_position_map, max_deg
                )
                group_feats.append(fv)
                all_feats_buffer.append(fv)
            
            dataset_items.append({
                'src': src_idx, 
                'goal': goal_idx, 
                'candidates': candidate_ids, 
                'features': group_feats
            })

    if all_feats_buffer:
        scaler = StandardScaler()
        scaler.fit(all_feats_buffer)
        for item in dataset_items:
            item['features'] = scaler.transform(item['features'])

    return ListwiseWikiDataset(dataset_items)

# --- Logic for GNN Evaluation ---
def evaluate_hybrid_mrr(
    model, pyg_data, paths_df, article_id_to_idx, links_map, dist_matrix, 
    pagerank_map, out_degree_map, link_position_map, device, 
    sample_size=None
):
    if sample_size is not None and sample_size < len(paths_df):
        paths_df = paths_df.sample(n=sample_size, random_state=42)
    
    model.eval()
    reciprocal_ranks = []
    max_deg = max(out_degree_map.values()) if out_degree_map else 1.0
    
    with torch.no_grad():
        h = model.forward_gnn(pyg_data.x, pyg_data.edge_index)
        
    with torch.no_grad():
        for row in tqdm(paths_df.itertuples(), total=len(paths_df), desc="Eval MRR"):
            path = row.path.split(';')
            goal = path[-1]
            if goal not in article_id_to_idx: continue
            goal_idx = article_id_to_idx[goal]

            for i in range(len(path) - 1):
                src = path[i]
                true_target = path[i+1]
                if src not in article_id_to_idx or true_target not in article_id_to_idx: continue
                src_idx = article_id_to_idx[src]
                
                all_neighbors = links_map.get(src, [])
                candidates = [n for n in all_neighbors if n in article_id_to_idx]
                if true_target not in candidates: continue
                
                cand_indices = [article_id_to_idx[c] for c in candidates]
                
                manual_features = []
                for c_idx, c_name in zip(cand_indices, candidates):
                    fv = compute_link_features(
                        src, c_name, src_idx, c_idx, goal_idx,
                        dist_matrix, pagerank_map, out_degree_map, link_position_map, max_deg
                    )
                    manual_features.append(fv)
                
                h_src_batch = h[src_idx].unsqueeze(0).repeat(len(candidates), 1)
                h_cand_batch = h[cand_indices]
                h_goal_batch = h[goal_idx].unsqueeze(0).repeat(len(candidates), 1)
                
                cos_sim = F.cosine_similarity(h_cand_batch, h_goal_batch).unsqueeze(1)
                feat_tensor = torch.FloatTensor(manual_features).to(device)
                
                # Local scaling
                if feat_tensor.shape[0] > 1:
                    mean = feat_tensor.mean(dim=0, keepdim=True)
                    std = feat_tensor.std(dim=0, keepdim=True) + 1e-8
                    feat_tensor = (feat_tensor - mean) / std
                
                feats_norm = model.feat_norm(feat_tensor)
                combined = torch.cat([h_src_batch, h_cand_batch, h_goal_batch, feats_norm, cos_sim], dim=1)
                scores = model.classifier(combined).view(-1)
                
                true_idx = candidates.index(true_target)
                true_score = scores[true_idx].item()
                rank = (scores > true_score).sum().item() + 1
                reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0