from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

from src.feature_extractor import EMBEDDING_DIM


def expand_paths_to_steps(paths_df):
    """
    Converts the paths_finished dataframe into a step-level dataframe,
    excluding the last transition before the goal and the goal itself.

    Output columns:
        - step_id
        - game_id
        - current_article
        - next_article
        - goal_article
        - path_length
        - step_number
    """

    rows = []
    global_step_id = 0

    for game_id, row in paths_df.iterrows():
        path_list = row["path"].split(";")
        path_length = len(path_list)

        goal = path_list[-1]

        # We stop at path_length - 2 to exclude:
        # - (d → e)   last real step
        # - (e → e)   impossible but excluded anyway
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
    """
    Compute the heuristic between an article and the goal article.
    
    :param article_id: The article we are evaluating.
    :param goal_article_id: The goal article we are trying to reach.
    :param embedding_map: Dictionary mapping article IDs to their embeddings.
    :param articles_df: DataFrame containing article metadata (like outdegree).
    :param links_map: Dictionary mapping article IDs to the list of linked articles.
    :param alpha: Weight parameter between 0 and 1.
    
    :return: The computed heuristic value.
    """
    # Get the embedding for the article and the goal article
    emb_article = embedding_map.get(article_id, np.zeros(EMBEDDING_DIM))
    emb_goal = embedding_map.get(goal_article_id, np.zeros(EMBEDDING_DIM))
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity([emb_article], [emb_goal])[0][0]
    
    # Get the outdegree of the article
    outdegree_article = articles_df.loc[articles_df['article_id'] == article_id, 'outdegree'].values[0]
    
    # Get the maximum outdegree from the articles linked by the current article
    linked_articles = links_map.get(article_id, [])
    max_outdegree = max(articles_df.loc[articles_df['article_id'].isin(linked_articles), 'outdegree'], default=1)
    
    # Calculate the heuristic
    heuristic_value = alpha * cosine_sim + (1 - alpha) * (outdegree_article / max_outdegree)
    
    return heuristic_value


def compute_mrr_next_link(current_article_id, goal_article_id, next_article_id, links_map, embedding_map, articles_df, alpha=0.5):
    """
    Compute the MRR (Mean Reciprocal Rank) for the current article and its linked articles,
    comparing the predicted next link (the most similar article to the goal) with the actual next link.
    
    :param current_article_id: The article we are evaluating (current article).
    :param goal_article_id: The goal article (the final article in the path).
    :param next_article_id: The actual next article in the path (the ground truth).
    :param links_map: Dictionary mapping article IDs to the list of linked articles.
    :param embedding_map: Dictionary mapping article IDs to their embeddings.
    :param articles_df: DataFrame containing article metadata (like outdegree).
    :param alpha: Weight parameter between 0 and 1 for combining cosine similarity and outdegree.
    
    :return: Reciprocal rank of the actual next article among the linked articles.
    """
    # Get the linked articles for the current article from links_map
    linked_articles = links_map.get(current_article_id, [])
    
    # Compute heuristic (cosine similarity and outdegree) for each linked article to the goal article
    heuristic_scores = []
    for linked_article in linked_articles:
        score = compute_heuristic(linked_article, goal_article_id, embedding_map, articles_df, links_map, alpha)
        heuristic_scores.append((linked_article, score))

    # Sort linked articles by heuristic (descending order)
    sorted_articles = sorted(heuristic_scores, key=lambda x: x[1], reverse=True)
    
    # Find the rank of the actual next article in the sorted list of linked articles
    rank = next((i + 1 for i, (article, _) in enumerate(sorted_articles) if article == next_article_id), len(linked_articles) + 1)
    
    # Return the reciprocal rank (1 / rank)
    return 1 / rank if rank <= len(linked_articles) else 0

