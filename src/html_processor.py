import os
from pathlib import Path
from urllib.parse import unquote
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import Dict, List

class HtmlProcessor:
    def __init__(self, html_base_path: str):
        self.html_base_path = Path(html_base_path)
        self.article_to_path = self._index_files()

    def _index_files(self) -> Dict[str, Path]:
        print(f"Indexing HTML files in {self.html_base_path}...")
        mapping = {}
        for path in self.html_base_path.rglob("*.htm*"):
            article_name = unquote(path.stem)
            mapping[article_name] = path
        return mapping

    def get_links_positions(self, article_name: str) -> Dict[str, float]:
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
                if '/wp/' in href and (href.endswith('.htm') or href.endswith('.html')):
                    raw_target = href.split('/')[-1].replace('.htm', '').replace('.html', '')
                    target_name = unquote(raw_target)
                    
                    if target_name not in valid_links_positions:
                        valid_links_positions[target_name] = idx / total_tags

            return valid_links_positions

        except Exception:
            return {}

    def precompute_positions(self, article_list: List[str]) -> Dict[str, Dict[str, float]]:
        print("Extracting link positions from HTML...")
        result = {}
        for article in tqdm(article_list, desc="Parsing HTML"):
            result[article] = self.get_links_positions(article)
        return result