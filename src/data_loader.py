import os
import re
from pathlib import Path
from urllib.parse import unquote, quote
from typing import List, Optional, Union

import pandas as pd
import numpy as np

# Default paths
DATA_DIR = Path("./data")
GRAPH_DIR = DATA_DIR / "wikispeedia_paths-and-graph"
TEXT_DIR = DATA_DIR / "plaintext_articles"
HTML_DIR = DATA_DIR / "wikispeedia_articles_html" / "wpcd"

def load_wikispeedia_file(filename: str, columns: Optional[List[str]] = None, sep: str = '\t', decode: bool = True) -> pd.DataFrame:
    path = GRAPH_DIR / filename
    print(f"Loading: {path}")

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
        # decode URL-encoded strings
        df = df.map(lambda x: unquote(x) if isinstance(x, str) else x)

    return df

def load_shortest_path_matrix(data_dir: Union[str, Path] = GRAPH_DIR) -> np.ndarray:
    file_path = Path(data_dir) / "shortest-path-distance-matrix.txt"
    matrix_data = []

    print("Loading distance matrix...")
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                row_data = []
                for char in line.strip():
                    if char == '_':
                        row_data.append(np.inf)
                    else:
                        try:
                            row_data.append(float(char))
                        except ValueError:
                            row_data.append(np.nan)
                matrix_data.append(row_data)

        return np.array(matrix_data, dtype=float)

    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        return np.array([])
    except Exception as e:
        print(f"ERROR loading distance matrix: {e}")
        return np.array([])

def get_article_text(title: str, text_dir: Union[str, Path] = TEXT_DIR) -> Union[str, float]:
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

def get_article_html(title: str, html_dir: Union[str, Path] = HTML_DIR) -> Union[str, float]:
    html_files_dir = Path(html_dir) / "wp"
    if not isinstance(title, str) or not title.strip():
        return np.nan

    encoded = quote(title, encoding="utf-8", safe="")
    target_filename = f"{encoded}.htm"
    
    # Using rglob to find the file recursively
    for path in html_files_dir.rglob(target_filename):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"ERROR reading file {path}: {e}")
            return np.nan
    return np.nan

def find_absolute_link_position_from_df(row: pd.Series, articles_lookup: dict) -> float:
    source = row.get("source")
    target = row.get("target")

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
    return float(match.start()) if match else np.nan