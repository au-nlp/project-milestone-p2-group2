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
from urllib.parse import unquote, quote
from pathlib import Path
import unicodedata
import re

DATA_DIR = "./data"
GRAPH_DIR = os.path.join(DATA_DIR, "wikispeedia_paths-and-graph")
TEXT_DIR = os.path.join(DATA_DIR, "plaintext_articles")
HTML_DIR = os.path.join(DATA_DIR, "wikispeedia_articles_html", "wpcd")


def load_wikispeedia_file(filename: str, columns=None, sep='\t', decode=True):
    """
    Generic loader for Wikispeedia TSV/TXT files.
    - Skips comment lines starting with '#'
    - Assigns column names if provided
    - Optionally URL-decodes text columns
    """

    path = os.path.join(GRAPH_DIR, filename)
    print("Loading:", path)

    df = pd.read_csv(
        path,
        sep=sep,
        comment='#',
        names=columns,
        header=None,
        dtype=str,   # keep all as string to avoid numeric conversion issues
        engine='python'
    )

    if decode:
        df = df.applymap(lambda x: unquote(x) if isinstance(x, str) else x)

    return df


def load_shortest_path_matrix(data_dir: str = GRAPH_DIR) -> np.ndarray:
    """
    Loads the 'shortest-path-distance-matrix.txt' into a NumPy array.
    This is a FIXED-WIDTH file, read character by character.
    """
    file_path = os.path.join(data_dir, "shortest-path-distance-matrix.txt")
    matrix_data = []

    print("Loading distance matrix...")

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

def get_article_text(title: str, text_dir: str = TEXT_DIR) -> str | float:
    """
    load content of correct .txt-file in text_dir for given article name
    The files are fully url-encoded (utf-8), e.g.:
      'M*A*S*H_(TV_series)' -> 'M%2AA%2AS%2AH_%28TV_series%29.txt'
      'Áedán_mac_Gabráin' -> '%C3%81ed%C3%A1n_mac_Gabr%C3%A1in.txt'
    returns text or np.nan, if file is not found
    """
    if not isinstance(title, str) or not title.strip():
        return np.nan

    # url-encode fully
    encoded = quote(title, encoding="utf-8", safe="")

    path = Path(text_dir) / f"{encoded}.txt"

    if not path.exists():
        return np.nan

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ ERROR reading file {path}: {e}")
        return np.nan
    

def get_article_html(title: str, html_dir: str = HTML_DIR) -> str | float:
    """
    load html-content of a Wikipedia-file (.htm) from folder "wikispeedia_articles_html/wpcd/wp"
    The files are fully url-encoded (utf-8), e.g.:
        'Áedán_mac_Gabráin' -> '%C3%81ed%C3%A1n_mac_Gabr%C3%A1in.htm'
    all subfolder will be searched by recursion
    returns html-text or np.nan, if file is not found
    """

    html_files_dir = os.path.join(html_dir, "wp")
    if not isinstance(title, str) or not title.strip():
        return np.nan

    # url-encode fully
    encoded = quote(title, encoding="utf-8", safe="")
    target_filename = f"{encoded}.htm"

    # search for file recursively
    base_path = Path(html_files_dir)
    for path in base_path.rglob(target_filename):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"⚠️ ERROR reading file {path}: {e}")
            return np.nan

    # nothing found
    return np.nan


def find_absolute_link_position_from_df(row, articles_lookup):
    """
    Find the character index of the <a href> linking to the target article
    inside the source article’s HTML (from articles_df, not from disk).
    Handles both single and double URL encoding.
    """
    source = row["source"]
    target = row["target"]

    html = articles_lookup.get(source)
    if not isinstance(html, str) or not html.strip():
        return np.nan

    # Encode target
    encoded_target = quote(target, safe="")                 # e.g. D%C3%A1l_Riata
    double_encoded_target = quote(encoded_target, safe="")  # e.g. D%25C3%25A1l_Riata

    # Build a regex that matches <a href="...encoded_target..." OR <a href="...double_encoded_target..."
    pattern = re.compile(
        rf'<a\s+href="[^"]*({re.escape(encoded_target)}|{re.escape(double_encoded_target)})[^"]*"',
        re.IGNORECASE
    )

    match = pattern.search(html)
    if not match:
        return np.nan

    return match.start()
