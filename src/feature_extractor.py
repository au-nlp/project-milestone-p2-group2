"""
Module for feature extraction.

Contains functions to:
1. Generate text embeddings using a SentenceTransformer model.
2. Calculate semantic similarity features (e.g., cosine similarity).
3. Extract graph-based features (shortest path, topology).
"""

import numpy as np
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Optional

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384


class FeatureExtractor:
    """
    A class to handle feature extraction.
    """

    _model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        """Loads the SentenceTransformer model into cache."""
        if self._model is None:
            print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
            self._model = SentenceTransformer(MODEL_NAME)
            print("Model loaded successfully.")
        return self._model

    def _get_first_paragraph(self, full_text: str) -> str:
        """Extracts the first paragraph (text before the first double newline)."""
        return full_text.split('\n\n')[0]

    def get_text_embedding(self, text: str, strategy: str = 'first_para') -> np.ndarray:
        """
        Generates an embedding for a given text using a specified strategy.

        Args:
            text: The full text of the article.
            strategy: 'first_para', 'title', or 'full' (not recommended for PoC).

        Returns:
            A NumPy array (vector) of the text embedding.
        """
        model = self._load_model()

        if not text:
            return np.zeros(EMBEDDING_DIM)

        if strategy == 'first_para':
            processed_text = self._get_first_paragraph(text)
        elif strategy == 'title':
            processed_text = text.split('\n')[0]
        else:
            processed_text = text

        if not processed_text:
            return np.zeros(EMBEDDING_DIM)

        embedding = model.encode(processed_text, convert_to_numpy=True)
        return embedding

    def get_semantic_features(self, emb_source: np.ndarray,
                              emb_candidate: np.ndarray,
                              emb_goal: np.ndarray) -> Dict[str, float]:
        """
        Calculates cosine similarity features between embeddings.
        """
        emb_source = emb_source.reshape(1, -1)
        emb_candidate = emb_candidate.reshape(1, -1)
        emb_goal = emb_goal.reshape(1, -1)

        sim_source_cand = cosine_similarity(emb_source, emb_candidate)[0][0]
        sim_cand_goal = cosine_similarity(emb_candidate, emb_goal)[0][0]

        return {
            "sim_source_candidate": float(sim_source_cand),
            "sim_candidate_goal": float(sim_cand_goal)
        }

    def get_shortest_path_features(self, source_idx: int,
                                   candidate_idx: int,
                                   goal_idx: int,
                                   dist_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Extracts features from the shortest path distance matrix.
        """
        dist_source_goal = dist_matrix[source_idx, goal_idx]
        dist_cand_goal = dist_matrix[candidate_idx, goal_idx]

        is_closer = 0
        if dist_cand_goal < dist_source_goal:
            is_closer = 1

        return {
            "dist_source_goal": dist_source_goal,
            "dist_candidate_goal": dist_cand_goal,
            "is_closer": is_closer
        }

    @staticmethod
    def create_topology_maps(links_df: pd.DataFrame) -> (Dict[str, float], Dict[str, int]):
        """
        Creates PageRank and Out-Degree maps from the links DataFrame.
        This is a pre-computation step to be run once in main.ipynb.
        """
        print("Building graph for PageRank and Out-Degree...")
        G = nx.from_pandas_edgelist(
            links_df,
            source='source',
            target='target',
            create_using=nx.DiGraph()
        )

        print("Calculating PageRank...")
        pagerank_map = nx.pagerank(G, alpha=0.85)

        print("Calculating Out-Degree...")
        out_degree_map = dict(G.out_degree())

        print("Graph feature maps created.")
        return pagerank_map, out_degree_map

    def get_topology_features(self, candidate_id: str,
                              pagerank_map: Dict[str, float],
                              out_degree_map: Dict[str, int]) -> Dict[str, float]:
        """
        Gets pre-computed PageRank and Out-Degree for a candidate article.
        """
        pagerank = pagerank_map.get(candidate_id, 0.0)
        out_degree = out_degree_map.get(candidate_id, 0)

        return {
            "pagerank": pagerank,
            "out_degree": float(out_degree)
        }