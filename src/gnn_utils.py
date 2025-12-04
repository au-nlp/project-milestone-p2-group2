import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def create_pyg_graph(articles_df, links_df, embedding_map):
    print("Building PyTorch Geometric Graph...")
    article_id_to_idx = {aid: i for i, aid in enumerate(articles_df['article_id'])}

    num_nodes = len(articles_df)
    emb_dim = len(next(iter(embedding_map.values())))

    x = torch.zeros((num_nodes, emb_dim), dtype=torch.float)
    for aid, idx in article_id_to_idx.items():
        emb = embedding_map.get(aid)
        if emb is not None:
            x[idx] = torch.tensor(emb, dtype=torch.float)

    sources = []
    targets = []
    for _, row in links_df.iterrows():
        src = row['source']
        dst = row['target']
        if src in article_id_to_idx and dst in article_id_to_idx:
            sources.append(article_id_to_idx[src])
            targets.append(article_id_to_idx[dst])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data, article_id_to_idx


def prepare_all_training_data(paths_df, article_id_to_idx, links_map, dist_matrix, neg_samples_ratio=5):
    print(f"Pre-computing tensors WITH DISTANCE (Neg Samples: {neg_samples_ratio})...")
    src_list, cand_list, goal_list, dist_list, label_list = [], [], [], [], []
    MAX_DIST = 8.0

    for row in tqdm(paths_df.itertuples(), total=len(paths_df), desc="Prep Data"):
        path_articles = row.path.split(';')
        goal_id = path_articles[-1]
        if goal_id not in article_id_to_idx: continue
        goal_idx = article_id_to_idx[goal_id]

        for i in range(len(path_articles) - 1):
            src_id = path_articles[i]
            true_next_id = path_articles[i + 1]
            if src_id not in article_id_to_idx or true_next_id not in article_id_to_idx: continue

            src_idx = article_id_to_idx[src_id]
            true_next_idx = article_id_to_idx[true_next_id]

            src_list.append(src_idx)
            cand_list.append(true_next_idx)
            goal_list.append(goal_idx)

            d = dist_matrix[true_next_idx, goal_idx]
            if np.isinf(d): d = MAX_DIST
            dist_list.append(d / MAX_DIST)
            label_list.append(1.0)

            candidates = links_map.get(src_id, [])
            cand_indices = [article_id_to_idx[c] for c in candidates if c in article_id_to_idx and c != true_next_id]
            if not cand_indices:
                cand_indices = [random.randint(0, len(article_id_to_idx) - 1)]

            if cand_indices:
                chosen = random.choices(cand_indices, k=neg_samples_ratio)
                for neg_idx in chosen:
                    src_list.append(src_idx)
                    cand_list.append(neg_idx)
                    goal_list.append(goal_idx)

                    d = dist_matrix[neg_idx, goal_idx]
                    if np.isinf(d): d = MAX_DIST
                    dist_list.append(d / MAX_DIST)
                    label_list.append(0.0)

    return TensorDataset(
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(cand_list, dtype=torch.long),
        torch.tensor(goal_list, dtype=torch.long),
        torch.tensor(dist_list, dtype=torch.float),
        torch.tensor(label_list, dtype=torch.float)
    )


def evaluate_gnn_mrr(model, pyg_data, val_paths_df, article_id_to_idx, links_map, dist_matrix, device, sample_size=500):
    model.eval()
    reciprocal_ranks = []
    MAX_DIST = 8.0

    sample_df = val_paths_df.sample(n=min(len(val_paths_df), sample_size), random_state=42)

    with torch.no_grad():
        all_node_embeddings = model.get_node_embeddings(pyg_data.x, pyg_data.edge_index)

        for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Eval MRR", leave=False):
            path = row.path.split(';')
            goal_id = path[-1]
            if goal_id not in article_id_to_idx: continue
            goal_idx = article_id_to_idx[goal_id]

            for i in range(len(path) - 1):
                src_id = path[i]
                true_next = path[i + 1]
                if src_id not in article_id_to_idx or true_next not in article_id_to_idx: continue

                candidates = links_map.get(src_id, [])
                valid_candidates = [c for c in candidates if c in article_id_to_idx]
                if not valid_candidates: continue

                src_idxs = torch.tensor([article_id_to_idx[src_id]] * len(valid_candidates), device=device)
                goal_idxs = torch.tensor([goal_idx] * len(valid_candidates), device=device)
                cand_idxs = torch.tensor([article_id_to_idx[c] for c in valid_candidates], device=device)

                dists = []
                for c_idx in cand_idxs.cpu().numpy():
                    d = dist_matrix[c_idx, goal_idx]
                    if np.isinf(d): d = MAX_DIST
                    dists.append(d / MAX_DIST)
                extra_feats = torch.tensor(dists, dtype=torch.float, device=device)

                scores = model.predict_link_score(all_node_embeddings, src_idxs, cand_idxs, goal_idxs,
                                                  extra_feats).squeeze()

                if scores.ndim == 0: scores = scores.unsqueeze(0)
                scores_np = scores.cpu().numpy()
                sorted_indices = np.argsort(scores_np)[::-1]
                candidates_np = np.array(valid_candidates)
                sorted_candidates = candidates_np[sorted_indices]

                try:
                    rank = np.where(sorted_candidates == true_next)[0][0] + 1
                    reciprocal_ranks.append(1.0 / rank)
                except IndexError:
                    reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)


def run_detailed_error_analysis(model, pyg_data, val_paths_df, article_id_to_idx, links_map, dist_matrix, device,
                                sample_size=1000):
    model.eval()
    results = []
    MAX_DIST = 8.0
    sample_df = val_paths_df.sample(n=min(len(val_paths_df), sample_size), random_state=42)

    with torch.no_grad():
        all_node_embeddings = model.get_node_embeddings(pyg_data.x, pyg_data.edge_index)
        for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Error Analysis"):
            path = row.path.split(';')
            goal_id = path[-1]
            if goal_id not in article_id_to_idx: continue
            goal_idx = article_id_to_idx[goal_id]
            for step_num, src_id in enumerate(path[:-1]):
                true_next = path[step_num + 1]
                if src_id not in article_id_to_idx or true_next not in article_id_to_idx: continue
                candidates = links_map.get(src_id, [])
                valid_candidates = [c for c in candidates if c in article_id_to_idx]
                if not valid_candidates: continue

                src_idxs = torch.tensor([article_id_to_idx[src_id]] * len(valid_candidates), device=device)
                goal_idxs = torch.tensor([goal_idx] * len(valid_candidates), device=device)
                cand_idxs = torch.tensor([article_id_to_idx[c] for c in valid_candidates], device=device)

                dists = []
                for c_idx in cand_idxs.cpu().numpy():
                    d = dist_matrix[c_idx, goal_idx]
                    if np.isinf(d): d = MAX_DIST
                    dists.append(d / MAX_DIST)
                extra_feats = torch.tensor(dists, dtype=torch.float, device=device)

                scores = model.predict_link_score(all_node_embeddings, src_idxs, cand_idxs, goal_idxs,
                                                  extra_feats).squeeze()
                if scores.ndim == 0: scores = scores.unsqueeze(0)

                scores_np = scores.cpu().numpy()
                sorted_indices = np.argsort(scores_np)[::-1]
                sorted_candidates = np.array(valid_candidates)[sorted_indices]

                rank = -1
                mrr = 0.0
                try:
                    rank = np.where(sorted_candidates == true_next)[0][0] + 1
                    mrr = 1.0 / rank
                except IndexError:
                    pass

                results.append({
                    'src_id': src_id,
                    'mrr': mrr,
                    'rank': rank,
                    'out_degree': len(valid_candidates),
                    'step_number': step_num,
                    'path_len_total': len(path)
                })
    return pd.DataFrame(results)


def plot_error_analysis(results_df, title_prefix="Model"):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    bins = [0, 10, 30, 60, 100, 500]
    labels = ['Very Low (1-10)', 'Low (11-30)', 'Medium (31-60)', 'High (61-100)', 'Very High (100+)']
    results_df['degree_group'] = pd.cut(results_df['out_degree'], bins=bins, labels=labels)
    sns.barplot(data=results_df, x='degree_group', y='mrr', ax=axes[0], palette="viridis")
    axes[0].set_title(f"{title_prefix}: MRR vs. Page Complexity")

    sns.lineplot(data=results_df[results_df['step_number'] < 10], x='step_number', y='mrr', ax=axes[1], marker='o',
                 color='crimson')
    axes[1].set_title(f"{title_prefix}: MRR vs. Step in Path")
    plt.tight_layout()
    plt.show()