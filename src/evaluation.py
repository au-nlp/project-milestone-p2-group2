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
    if not isinstance(y_val, np.ndarray):
        y_val = y_val.to_numpy()

    if not isinstance(query_groups, np.ndarray):
        query_groups = query_groups.to_numpy()

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_val)[:, 1]
    else:
        scores = model.predict(X_val)

    df = pd.DataFrame({
        'query_id': query_groups,
        'score': scores,
        'y_true': y_val
    })

    reciprocal_ranks = []

    for _, group in df.groupby('query_id'):
        group_sorted = group.sort_values('score', ascending=False).reset_index(drop=True)
        
        # Find index of first positive sample
        true_indices = group_sorted.index[group_sorted['y_true'] == 1].tolist()
        
        if true_indices:
            rank = true_indices[0] + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))