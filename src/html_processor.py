"""
Module for processing HTML files to extract link positions.
"""

import os
import urllib.parse
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import Dict, List, Optional


def decode_url_string(s):
    """Helper to decode URL-encoded strings."""
    if isinstance(s, str):
        return urllib.parse.unquote(s)
    return s


class HtmlProcessor:
    def __init__(self, html_base_path: str):
        """
        Args:
            html_base_path: Path to the folder containing HTML files
                            (e.g., './data/wikispeedia_articles_html/wpcd/wp')
        """
        self.html_base_path = html_base_path
        self.article_to_path = self._index_files()

    def _index_files(self) -> Dict[str, str]:
        """
        Recursively maps article names to their file paths.
        """
        print(f"Indexing HTML files in {self.html_base_path}...")
        mapping = {}
        for root, _, files in os.walk(self.html_base_path):
            for file in files:
                if file.endswith('.htm') or file.endswith('.html'):
                    article_name = decode_url_string(os.path.splitext(file)[0])
                    mapping[article_name] = os.path.join(root, file)

        print(f"Found {len(mapping)} HTML files.")
        return mapping

    def get_links_positions(self, article_name: str) -> Dict[str, float]:
        """
        Parses HTML and returns normalized position of each link.
        0.0 = top of page, 1.0 = bottom of page.
        """
        file_path = self.article_to_path.get(article_name)
        if not file_path:
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'html.parser')

            all_links = soup.find_all('a', href=True)
            if not all_links:
                return {}

            valid_links_positions = {}
            total_tags = len(all_links)

            for idx, tag in enumerate(all_links):
                href = tag['href']
                if '/wp/' in href and href.endswith('.htm'):
                    raw_target = href.split('/')[-1].replace('.htm', '')
                    target_name = decode_url_string(raw_target)
                    if target_name not in valid_links_positions:
                        valid_links_positions[target_name] = idx / total_tags

            return valid_links_positions

        except Exception as e:
            # print(f"Error parsing {article_name}: {e}")
            return {}

    def precompute_positions(self, article_list: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Runs extraction for all articles.
        Returns: {source_article: {target_article: position_float}}
        """
        print("Extracting link positions from HTML...")
        result = {}
        for article in tqdm(article_list, desc="Parsing HTML"):
            result[article] = self.get_links_positions(article)
        return result