import numpy as np
import networkx as nx
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, Optional, Tuple

MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384

class FeatureExtractor:
    _model: Optional[SentenceTransformer] = None

    @classmethod
    def _load_model(cls) -> SentenceTransformer:
        if cls._model is None:
            print(f"Loading SentenceTransformer model: {MODEL_NAME}...")
            cls._model = SentenceTransformer(MODEL_NAME)
            print("Model loaded successfully.")
        return cls._model

    def _get_first_paragraph(self, full_text: str) -> str:
        if not full_text:
            return ""
        return full_text.split('\n\n')[0]

    def get_text_embedding(self, text: str, strategy: str = 'first_para') -> np.ndarray:
        model = self._load_model()

        if not text:
            return np.zeros(EMBEDDING_DIM)

        if strategy == 'first_para':
            processed_text = self._get_first_paragraph(text)
        elif strategy == 'title':
            processed_text = text.split('\n')[0]
        else:
            processed_text = text

        if not processed_text.strip():
            return np.zeros(EMBEDDING_DIM)

        return model.encode(processed_text, convert_to_numpy=True)

    def get_semantic_features(self, emb_source: np.ndarray,
                              emb_candidate: np.ndarray,
                              emb_goal: np.ndarray) -> Dict[str, float]:
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
        # Robust check for invalid indices
        if source_idx == -1 or candidate_idx == -1 or goal_idx == -1:
             return {
                "dist_source_goal": 10.0,
                "dist_candidate_goal": 10.0,
                "is_closer": 0
            }

        dist_source_goal = dist_matrix[source_idx, goal_idx]
        dist_cand_goal = dist_matrix[candidate_idx, goal_idx]

        # Handle infinity (unreachable nodes)
        d_sg = dist_source_goal if not np.isinf(dist_source_goal) else 10.0
        d_cg = dist_cand_goal if not np.isinf(dist_cand_goal) else 10.0

        is_closer = 1 if d_cg < d_sg else 0

        return {
            "dist_source_goal": float(d_sg),
            "dist_candidate_goal": float(d_cg),
            "is_closer": is_closer
        }

    @staticmethod
    def create_topology_maps(links_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, int]]:
        print("Building graph for PageRank and Out-Degree...")
        G = nx.from_pandas_edgelist(
            links_df,
            source='source',
            target='target',
            create_using=nx.DiGraph()
        )

        print("Calculating PageRank...")
        pagerank_map = nx.pagerank(G, alpha=0.85)
        out_degree_map = dict(G.out_degree())

        print("Graph feature maps created.")
        return pagerank_map, out_degree_map

    def get_topology_features(self, candidate_id: str,
                              pagerank_map: Dict[str, float],
                              out_degree_map: Dict[str, int]) -> Dict[str, float]:
        return {
            "pagerank": pagerank_map.get(candidate_id, 0.0),
            "out_degree": float(out_degree_map.get(candidate_id, 0))
        }