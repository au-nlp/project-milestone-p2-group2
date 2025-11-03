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
    path = os.path.join(GRAPH_DIR, filename)
    print("Loading:", path)

    df = pd.read_csv(
        path,
        sep=sep,
        comment='#',
        names=columns,
        header=None,
        dtype=str,
        engine='python'
    )

    if decode:
        df = df.applymap(lambda x: unquote(x) if isinstance(x, str) else x)

    return df


def load_shortest_path_matrix(data_dir: str = GRAPH_DIR) -> np.ndarray:
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
    if not isinstance(title, str) or not title.strip():
        return np.nan

    encoded = quote(title, encoding="utf-8", safe="")
    path = Path(text_dir) / f"{encoded}.txt"

    if not path.exists():
        return np.nan

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"ERROR reading file {path}: {e}")
        return np.nan
    

def get_article_html(title: str, html_dir: str = HTML_DIR) -> str | float:
    html_files_dir = os.path.join(html_dir, "wp")
    if not isinstance(title, str) or not title.strip():
        return np.nan

    encoded = quote(title, encoding="utf-8", safe="")
    target_filename = f"{encoded}.htm"
    base_path = Path(html_files_dir)
    for path in base_path.rglob(target_filename):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"ERROR reading file {path}: {e}")
            return np.nan
    return np.nan


def find_absolute_link_position_from_df(row, articles_lookup):
    source = row["source"]
    target = row["target"]

    html = articles_lookup.get(source)
    if not isinstance(html, str) or not html.strip():
        return np.nan

    encoded_target = quote(target, safe="")
    double_encoded_target = quote(encoded_target, safe="")

    pattern = re.compile(
        rf'<a\s+href="[^"]*({re.escape(encoded_target)}|{re.escape(double_encoded_target)})[^"]*"',
        re.IGNORECASE
    )

    match = pattern.search(html)
    if not match:
        return np.nan

    return match.start()
