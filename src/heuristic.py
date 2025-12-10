import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.feature_extractor import EMBEDDING_DIM

def expand_paths_to_steps(paths_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts path strings into individual (current -> next) steps.
    """
    rows = []
    global_step_id = 0

    for game_id, row in paths_df.iterrows():
        path_list = row["path"].split(";")
        path_length = len(path_list)
        goal = path_list[-1]

        for step_number in range(path_length - 2):
            current_article = path_list[step_number]
            next_article = path_list[step_number + 1]

            rows.append({
                "step_id": global_step_id,
                "game_id": game_id,
                "current_article": current_article,
                "next_article": next_article,
                "goal_article": goal,
                "path_length": path_length,
                "step_number": step_number + 1
            })
            global_step_id += 1

    return pd.DataFrame(rows)

def compute_heuristic(article_id, goal_article_id, embedding_map, articles_df, links_map, alpha=0.5):
    emb_article = embedding_map.get(article_id, np.zeros(EMBEDDING_DIM))
    emb_goal = embedding_map.get(goal_article_id, np.zeros(EMBEDDING_DIM))
    
    sim = cosine_similarity(emb_article.reshape(1, -1), emb_goal.reshape(1, -1))[0][0]
    
    try:
        outdegree = articles_df.loc[articles_df['article_id'] == article_id, 'outdegree'].values[0]
    except IndexError:
        outdegree = 0
    
    linked_articles = links_map.get(article_id, [])
    if linked_articles:
        max_out = articles_df.loc[articles_df['article_id'].isin(linked_articles), 'outdegree'].max()
        max_out = max(max_out, 1)
    else:
        max_out = 1
    
    return alpha * sim + (1 - alpha) * (outdegree / max_out)

def compute_mrr_next_link(current_id, goal_id, true_next_id, links_map, embedding_map, articles_df, alpha=0.5):
    neighbors = links_map.get(current_id, [])
    
    if true_next_id not in neighbors:
        return 0.0

    scores = []
    for neighbor in neighbors:
        score = compute_heuristic(neighbor, goal_id, embedding_map, articles_df, links_map, alpha)
        scores.append((neighbor, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    
    rank = 0
    for i, (cand_id, _) in enumerate(scores):
        if cand_id == true_next_id:
            rank = i + 1
            break
            
    return 1.0 / rank if rank > 0 else 0.0