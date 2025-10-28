"""
Module for loading and preprocessing data from the Wikispeedia dataset.

Contains functions to load:
- Article ID to name mappings (articles.tsv)
- Graph structure (links.tsv)
- Finished navigation paths (paths_finished.tsv)
- Shortest path distance matrix (shortest-path-distance-matrix.txt)
- Plaintext content of articles from the plaintext_articles directory

NOTE (V7 - Unholy Mix):
- articles.tsv: IS COMMA-SEPARATED (,) and has NO HEADER (1 column).
- links.tsv: IS TAB-SEPARATED (\t) and has NO HEADER (2 columns).
- paths_finished.tsv: IS TAB-SEPARATED (\t) and has NO HEADER (6 columns).
- distance-matrix.txt: IS FIXED-WIDTH (char-per-field) and has NO HEADER.
- All strings are URL-decoded upon loading.
"""

import pandas as pd
import numpy as np
import os
import urllib.parse

DATA_DIR = "./data"
GRAPH_DIR = os.path.join(DATA_DIR, "wikispeedia_paths-and-graph")
TEXT_DIR = os.path.join(DATA_DIR, "plaintext_articles")

def decode_url_string(s):
    """Helper function to decode URL-encoded strings."""
    if isinstance(s, str):
        return urllib.parse.unquote(s)
    return s

def load_articles_df(data_dir: str = GRAPH_DIR) -> pd.DataFrame:
    """
    Loads 'articles.csv', mapping hashed IDs to human-readable article names.
    This file has NO header and is COMMA-separated (1 column).
    """
    file_path = os.path.join(data_dir, "articles.tsv")
    col_names = ['article_id']
    try:
        articles_df = pd.read_csv(
            file_path,
            sep=',',
            header=None,
            names=col_names,
            comment='#'
        )
        articles_df['article_id'] = articles_df['article_id'].apply(decode_url_string)
        articles_df['article_name'] = articles_df['article_id']
        return articles_df
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading articles.tsv: {e}")
        return pd.DataFrame()


def load_links_df(data_dir: str = GRAPH_DIR) -> pd.DataFrame:
    """
    Loads 'links.tsv', containing the graph edges (links between articles).
    This file has NO header and is TAB-separated.
    """
    file_path = os.path.join(data_dir, "links.tsv")
    col_names = ['source', 'target']
    try:
        links_df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=col_names,
            comment='#'
        )
        links_df['source'] = links_df['source'].apply(decode_url_string)
        links_df['target'] = links_df['target'].apply(decode_url_string)
        return links_df
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading links.tsv: {e}")
        return pd.DataFrame()


def load_paths_df(data_dir: str = GRAPH_DIR) -> pd.DataFrame:
    """
    Loads 'paths_finished.tsv', containing completed navigation paths.
    This file has NO header and is TAB-separated.
    """
    file_path = os.path.join(data_dir, "paths_finished.tsv")
    col_names = ["session_id", "timestamp", "session_time", "path", "rating", "type"]

    try:
        paths_df = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=col_names,
            comment='#'
        )
        paths_df['path'] = paths_df['path'].apply(decode_url_string)

        return paths_df
    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR loading paths_finished.tsv: {e}")
        return pd.DataFrame()


def load_shortest_path_matrix(data_dir: str = GRAPH_DIR) -> np.ndarray:
    """
    Loads the 'shortest-path-distance-matrix.txt' into a NumPy array.
    This is a FIXED-WIDTH file, read character by character.
    """
    file_path = os.path.join(data_dir, "shortest-path-distance-matrix.txt")
    matrix_data = []

    print("Loading distance matrix (fixed-width method)...")

    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue

                line = line.strip()
                if not line:
                    continue

                row_data = []
                for char in line:
                    if char == '_':
                        row_data.append(np.inf)
                    else:
                        try:
                            row_data.append(float(char))
                        except ValueError:
                            row_data.append(np.nan)

                matrix_data.append(row_data)

        distance_matrix = np.array(matrix_data, dtype=float)
        return distance_matrix

    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return np.array([])
    except Exception as e:
        print(f"ERROR loading distance matrix: {e}")
        return np.array([])


def get_article_text(article_name: str, text_dir: str = TEXT_DIR) -> str:
    """
    Loads the text content of an article based on its name.
    """
    file_path = os.path.join(text_dir, f"{article_name}.txt")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"ERROR reading file {file_path}: {e}")
        return ""