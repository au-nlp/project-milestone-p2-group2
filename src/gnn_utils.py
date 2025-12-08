import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler

class ListwiseWikiDataset(Dataset):
    def __init__(self, items):
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

def prepare_hard_listwise_data(
    paths_df, article_id_to_idx, links_map, dist_matrix, 
    pagerank_map, out_degree_map, link_position_map, neg_samples=20
):
    dataset_items = []
    max_deg = max(out_degree_map.values()) if out_degree_map else 1.0
    all_names = list(article_id_to_idx.keys())
    all_feats_buffer = []

    for row in tqdm(paths_df.itertuples(), total=len(paths_df), desc="Prep Hard Data"):
        path = row.path.split(';')
        goal = path[-1]
        if goal not in article_id_to_idx: continue
        goal_idx = article_id_to_idx[goal]

        for i in range(len(path) - 1):
            src = path[i]
            true_target = path[i+1]
            if src not in article_id_to_idx or true_target not in article_id_to_idx: continue
            src_idx = article_id_to_idx[src]
            
            # Neighbors
            all_neighbors = links_map.get(src, [])
            valid_neighbors = [n for n in all_neighbors if n in article_id_to_idx]
            negatives = [n for n in valid_neighbors if n != true_target]
            
            # Sampling
            if len(negatives) < neg_samples:
                needed = neg_samples - len(negatives)
                negatives.extend(random.choices(all_names, k=needed))
            elif len(negatives) > neg_samples:
                negatives = random.sample(negatives, neg_samples)
            negatives = negatives[:neg_samples]

            candidates = [true_target] + negatives
            candidate_ids = [article_id_to_idx[n] for n in candidates]
            
            group_feats = []
            src_pos = link_position_map.get(src, {})
            
            for c_id, c_name in zip(candidate_ids, candidates):
                # Features
                pos = src_pos.get(c_name, 1.0)
                pr = pagerank_map.get(c_name, 0.0)
                deg = out_degree_map.get(c_name, 0.0) / max_deg
                
                dc = dist_matrix[c_id, goal_idx]
                if np.isinf(dc): dc = 10.0
                ds = dist_matrix[src_idx, goal_idx]
                if np.isinf(ds): ds = 10.0
                closer = 1.0 if dc < ds else 0.0
                
                fv = [pos, pr, deg, closer, dc]
                group_feats.append(fv)
                all_feats_buffer.append(fv)
            
            dataset_items.append({
                'src': src_idx, 'goal': goal_idx, 'candidates': candidate_ids, 'features': group_feats
            })

    # Fit scaler on this batch generation
    if all_feats_buffer:
        scaler = StandardScaler()
        scaler.fit(all_feats_buffer)
        for item in dataset_items:
            item['features'] = scaler.transform(item['features'])

    return ListwiseWikiDataset(dataset_items)

def evaluate_hybrid_mrr(
    model, pyg_data, paths_df, article_id_to_idx, links_map, dist_matrix, 
    pagerank_map, out_degree_map, link_position_map, device, 
    sample_size=None
):
    if sample_size is not None and sample_size < len(paths_df):
        paths_df = paths_df.sample(n=sample_size, random_state=42)
    
    model.eval()
    reciprocal_ranks = []
    max_out_degree = max(out_degree_map.values()) if out_degree_map else 1.0
    
    # Use model's internal GNN forward pass to ensure consistency
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
                src_positions = link_position_map.get(src, {})
                
                for c_idx, cand in zip(cand_indices, candidates):
                    pos_feat = src_positions.get(cand, 1.0)
                    pr_feat = pagerank_map.get(cand, 0.0)
                    deg_feat = out_degree_map.get(cand, 0.0) / max_out_degree
                    
                    dist_c_g = dist_matrix[c_idx, goal_idx]
                    if np.isinf(dist_c_g): dist_c_g = 10.0
                    dist_s_g = dist_matrix[src_idx, goal_idx]
                    if np.isinf(dist_s_g): dist_s_g = 10.0
                    is_closer = 1.0 if dist_c_g < dist_s_g else 0.0
                    
                    manual_features.append([pos_feat, pr_feat, deg_feat, is_closer, dist_c_g])
                
                # Batch tensors
                h_src_batch = h[src_idx].unsqueeze(0).repeat(len(candidates), 1)
                h_cand_batch = h[cand_indices]
                h_goal_batch = h[goal_idx].unsqueeze(0).repeat(len(candidates), 1)
                
                # Features
                cos_sim = F.cosine_similarity(h_cand_batch, h_goal_batch).unsqueeze(1)
                feat_tensor = torch.FloatTensor(manual_features).to(device)
                
                # Local scaling for evaluation
                if feat_tensor.shape[0] > 1:
                    mean = feat_tensor.mean(dim=0, keepdim=True)
                    std = feat_tensor.std(dim=0, keepdim=True) + 1e-8
                    feat_tensor = (feat_tensor - mean) / std
                
                # Classifier pass
                feats_norm = model.feat_norm(feat_tensor)
                combined = torch.cat([h_src_batch, h_cand_batch, h_goal_batch, feats_norm, cos_sim], dim=1)
                scores = model.classifier(combined).view(-1)
                
                true_idx = candidates.index(true_target)
                true_score = scores[true_idx].item()
                
                # Rank
                rank = (scores > true_score).sum().item() + 1
                reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks)