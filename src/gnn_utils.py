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

    print(f"Graph created successfully!")
    print(f" - Nodes: {data.num_nodes}")
    print(f" - Edges: {data.num_edges}")

    return data, article_id_to_idx

def get_gnn_batch_data(paths_batch, article_id_to_idx, links_map, neg_samples_ratio=1):
    """
    Prepares tensors (src, cand, goal, label) from a batch of paths.
    For each step, creates 1 positive sample and N negative samples.
    """
    src_idxs, cand_idxs, goal_idxs, labels = [], [], [], []

    for row in paths_batch.itertuples():
        path_articles = row.path.split(';')
        goal_id = path_articles[-1]
        if goal_id not in article_id_to_idx:
            continue
        goal_idx = article_id_to_idx[goal_id]

        for i in range(len(path_articles) - 1):
            src_id = path_articles[i]
            true_next_id = path_articles[i + 1]

            if src_id not in article_id_to_idx or true_next_id not in article_id_to_idx:
                continue

            src_idx = article_id_to_idx[src_id]
            true_next_idx = article_id_to_idx[true_next_id]

            # --- Positive Sample ---
            src_idxs.append(src_idx)
            cand_idxs.append(true_next_idx)
            goal_idxs.append(goal_idx)
            labels.append(1.0)

            # --- Negative Samples (Hard Negatives) ---
            candidates = links_map.get(src_id, [])
            neg_candidates = [c for c in candidates if c != true_next_id and c in article_id_to_idx]

            if neg_candidates:
                # Randomly select N negatives
                chosen_negs = random.choices(neg_candidates, k=min(len(neg_candidates), neg_samples_ratio))
                for neg_id in chosen_negs:
                    src_idxs.append(src_idx)
                    cand_idxs.append(article_id_to_idx[neg_id])
                    goal_idxs.append(goal_idx)
                    labels.append(0.0)
            else:
                rand_node = random.randint(0, len(article_id_to_idx) - 1)
                src_idxs.append(src_idx)
                cand_idxs.append(rand_node)
                goal_idxs.append(goal_idx)
                labels.append(0.0)

    return (
        torch.tensor(src_idxs, dtype=torch.long),
        torch.tensor(cand_idxs, dtype=torch.long),
        torch.tensor(goal_idxs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float)
    )

def prepare_all_training_data(paths_df, article_id_to_idx, links_map, neg_samples_ratio=5):
    print(f"Pre-computing training data (Neg Samples: {neg_samples_ratio})...")
    src_list, cand_list, goal_list, label_list = [], [], [], []

    for row in tqdm(paths_df.itertuples(), total=len(paths_df), desc="Preparing Tensors"):
        path_articles = row.path.split(';')
        goal_id = path_articles[-1]

        if goal_id not in article_id_to_idx: continue
        goal_idx = article_id_to_idx[goal_id]

        for i in range(len(path_articles) - 1):
            src_id = path_articles[i]
            true_next_id = path_articles[i+1]

            if src_id not in article_id_to_idx or true_next_id not in article_id_to_idx: continue

            src_idx = article_id_to_idx[src_id]
            true_next_idx = article_id_to_idx[true_next_id]

            src_list.append(src_idx)
            cand_list.append(true_next_idx)
            goal_list.append(goal_idx)
            label_list.append(1.0)

            candidates = links_map.get(src_id, [])
            if not candidates:
                rand_node = random.randint(0, len(article_id_to_idx)-1)
                src_list.append(src_idx)
                cand_list.append(rand_node)
                goal_list.append(goal_idx)
                label_list.append(0.0)
                continue

            cand_indices = [article_id_to_idx[c] for c in candidates if c in article_id_to_idx and c != true_next_id]

            if cand_indices:
                chosen = random.choices(cand_indices, k=neg_samples_ratio)
                for neg_idx in chosen:
                    src_list.append(src_idx)
                    cand_list.append(neg_idx)
                    goal_list.append(goal_idx)
                    label_list.append(0.0)

    dataset = TensorDataset(
        torch.tensor(src_list, dtype=torch.long),
        torch.tensor(cand_list, dtype=torch.long),
        torch.tensor(goal_list, dtype=torch.long),
        torch.tensor(label_list, dtype=torch.float)
    )
    return dataset

def evaluate_sage_mrr(model, pyg_data, val_paths_df, article_id_to_idx, links_map, device, sample_size=500):
    model.eval()
    reciprocal_ranks = []

    sample_df = val_paths_df.sample(n=min(len(val_paths_df), sample_size), random_state=42)

    with torch.no_grad():
        for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Eval GNN MRR", leave=False):
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

                scores = model(pyg_data.x, pyg_data.edge_index, src_idxs, cand_idxs, goal_idxs).squeeze()

                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)

                scores_np = scores.cpu().numpy()
                candidates_np = np.array(valid_candidates)
                sorted_indices = np.argsort(scores_np)[::-1]
                sorted_candidates = candidates_np[sorted_indices]

                try:
                    rank = np.where(sorted_candidates == true_next)[0][0] + 1
                    reciprocal_ranks.append(1.0 / rank)
                except IndexError:
                    reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)

def evaluate_gatv2_mrr(model, pyg_data, val_paths_df, article_id_to_idx, links_map, device, sample_size=500):
    model.eval()
    reciprocal_ranks = []

    sample_df = val_paths_df.sample(n=min(len(val_paths_df), sample_size), random_state=42)

    with torch.no_grad():
        all_node_embeddings = model.get_node_embeddings(pyg_data.x, pyg_data.edge_index)

        for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Eval GNN MRR", leave=False):
            path = row.path.split(';')
            goal_id = path[-1]
            if goal_id not in article_id_to_idx: continue
            goal_idx = article_id_to_idx[goal_id]

            for i in range(len(path) - 1):
                src_id = path[i]
                true_next = path[i+1]

                if src_id not in article_id_to_idx or true_next not in article_id_to_idx: continue

                candidates = links_map.get(src_id, [])
                valid_candidates = [c for c in candidates if c in article_id_to_idx]
                if not valid_candidates: continue

                src_idxs = torch.tensor([article_id_to_idx[src_id]] * len(valid_candidates), device=device)
                goal_idxs = torch.tensor([goal_idx] * len(valid_candidates), device=device)
                cand_idxs = torch.tensor([article_id_to_idx[c] for c in valid_candidates], device=device)

                scores = model.predict_link_score(all_node_embeddings, src_idxs, cand_idxs, goal_idxs).squeeze()

                if scores.ndim == 0:
                    scores = scores.unsqueeze(0)

                scores_np = scores.cpu().numpy()
                candidates_np = np.array(valid_candidates)

                sorted_indices = np.argsort(scores_np)[::-1]
                sorted_candidates = candidates_np[sorted_indices]

                try:
                    rank = np.where(sorted_candidates == true_next)[0][0] + 1
                    reciprocal_ranks.append(1.0 / rank)
                except IndexError:
                    reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks)

def run_detailed_error_analysis(model, pyg_data, val_paths_df, article_id_to_idx, links_map, device, sample_size=1000):
    model.eval()
    results = []

    sample_df = val_paths_df.sample(n=min(len(val_paths_df), sample_size), random_state=42)

    with torch.no_grad():
        all_node_embeddings = model.get_node_embeddings(pyg_data.x, pyg_data.edge_index)

        for row in tqdm(sample_df.itertuples(), total=len(sample_df), desc="Analyzing Errors"):
            path = row.path.split(';')
            goal_id = path[-1]
            if goal_id not in article_id_to_idx: continue
            goal_idx = article_id_to_idx[goal_id]

            for step_num, src_id in enumerate(path[:-1]):
                true_next = path[step_num+1]

                if src_id not in article_id_to_idx or true_next not in article_id_to_idx: continue

                candidates = links_map.get(src_id, [])
                valid_candidates = [c for c in candidates if c in article_id_to_idx]
                if not valid_candidates: continue

                src_idxs = torch.tensor([article_id_to_idx[src_id]] * len(valid_candidates), device=device)
                goal_idxs = torch.tensor([goal_idx] * len(valid_candidates), device=device)
                cand_idxs = torch.tensor([article_id_to_idx[c] for c in valid_candidates], device=device)

                scores = model.predict_link_score(all_node_embeddings, src_idxs, cand_idxs, goal_idxs).squeeze()
                if scores.ndim == 0: scores = scores.unsqueeze(0)

                scores_np = scores.cpu().numpy()
                candidates_np = np.array(valid_candidates)
                sorted_indices = np.argsort(scores_np)[::-1]
                sorted_candidates = candidates_np[sorted_indices]

                mrr = 0.0
                rank = -1
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


def plot_training_history(history, model_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss
    ax1.plot(history['loss'], label='Train Loss', color='red', marker='o')
    ax1.set_title('Training Loss over Epochs - ' + model_name)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # MRR
    ax2.plot(history['mrr'], label='Validation MRR', color='blue', marker='o')
    ax2.set_title('Validation MRR over Epochs - ' + model_name)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MRR')
    ax2.grid(True)

    plt.show()

def plot_error_analysis(results_df, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    bins = [0, 10, 30, 60, 100, 500]
    labels = ['Very Low (1-10)', 'Low (11-30)', 'Medium (31-60)', 'High (61-100)', 'Very High (100+)']
    results_df['degree_group'] = pd.cut(results_df['out_degree'], bins=bins, labels=labels)

    sns.barplot(data=results_df, x='degree_group', y='mrr', ax=axes[0], palette="viridis")
    axes[0].set_title("Model Performance vs. Page Complexity (Out-Degree) - " + model_name)
    axes[0].set_xlabel("Number of Links on Page")
    axes[0].set_ylabel("Mean Reciprocal Rank (MRR)")
    axes[0].grid(axis='y', alpha=0.3)

    sns.lineplot(data=results_df[results_df['step_number'] < 10], x='step_number', y='mrr', ax=axes[1], marker='o', color='crimson')
    axes[1].set_title("Model Performance vs. Step in Path - " + model_name)
    axes[1].set_xlabel("Step Number (0 = Start of Game)")
    axes[1].set_ylabel("Mean Reciprocal Rank (MRR)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()