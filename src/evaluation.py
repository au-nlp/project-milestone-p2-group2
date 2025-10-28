"""
Module for model evaluation metrics.

Contains functions to calculate metrics relevant to ranking tasks,
such as Mean Reciprocal Rank (MRR).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from typing import Union


def calculate_mrr(
        model: BaseEstimator,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        query_groups: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR) for the given model.

    This function assumes the task is formulated as ranking, where
    for each query (a navigation step), the model scores all possible
    candidates, and we measure the rank of the true positive candidate.

    Args:
        model: The trained scikit-learn compatible model.
        X_val: The feature matrix for the validation set.
        y_val: The true labels (0 or 1) for the validation set.
        query_groups: An array or Series identifying which query each
                      row in X_val belongs to (e.g., a path_step_id).

    Returns:
        The Mean Reciprocal Rank (MRR) score as a float.
    """

    if not isinstance(y_val, np.ndarray):
        y_val = y_val.to_numpy()

    if not isinstance(query_groups, np.ndarray):
        query_groups = query_groups.to_numpy()

    scores = model.predict_proba(X_val)[:, 1]

    df = pd.DataFrame({
        'query_id': query_groups,
        'score': scores,
        'y_true': y_val
    })

    reciprocal_ranks = []

    for query_id, group in df.groupby('query_id'):
        group_sorted = group.sort_values('score', ascending=False).reset_index()

        try:
            rank = group_sorted[group_sorted['y_true'] == 1].index[0] + 1
            reciprocal_ranks.append(1 / rank)
        except IndexError:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)
    return float(mrr)