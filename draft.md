# Human-Like Next-Link Prediction

**Team:** Szymon, Jendrik, Patrik

## Abstract

Humans do not navigate Wikipedia by calculating shortest paths; they rely on semantic intuition, contextual clues, and general knowledge. This project aims to predict the *next link* a human will click in the Wikispeedia "pathfinding" game, given a current article and an end goal. Our core motivation is to model this human navigation strategy as a hybrid task, blending semantic relevance with graph-structural awareness. We will build and evaluate models that combine modern NLP embeddings (to understand *what* an article is about) with graph features (to understand *where* it is in the network). Our goal is to create a model that navigates Wikipedia more like a human and less like an algorithm, testing if modern embeddings can effectively capture this complex intuition.

## Contributions

The novelty of this project lies in the explicit modeling of human navigation strategy as a hybrid task. While prior work noted that shortest-path distance is insufficient, our project combines:
1.  **Modern NLP Embeddings:** Using pre-trained `sentence-transformers` to capture the semantic content of articles.
2.  **Structural Graph Features:** Utilizing graph topology (PageRank, Out-Degree) and shortest-path distance as proxies for human awareness of "hub" articles and proximity to the goal.
3.  **Proof of Concept:** Our P2 milestone (see `main.ipynb`) already demonstrates the feasibility of this approach. A simple logistic regression model trained on these hybrid features achieved a **Mean Reciprocal Rank (MRR) of 0.4556**, proving the strong predictive power of our chosen features.
4.  **Final Model:** Our final contribution (for P3) will be a Graph Neural Network (GNN) architecture that learns the optimal, non-linear balance between these semantic and structural features.

## Proposed Additional Datasets

Our Proof of Concept (P2) successfully used the `plaintext_articles` dataset for rapid prototyping and embedding generation.

For P3, we propose to additionally incorporate the **`wikispeedia_articles_html` dataset**.

* **Goal:** To extract a crucial feature unavailable in plaintext: the **position of a link** on the page.
* **Hypothesis:** Links appearing earlier in the text (e.g., in the lead paragraph) are more likely to be noticed and clicked by a human user.
* **Plan:** We will use a library like `BeautifulSoup` to parse the HTML files. For each source article, we will find all `<a>` tags, extract their `href` (to map to a candidate article), and determine their relative position in the document's text. This will add a new feature (e.g., `link_position_normalized`) to our models, which we hypothesize will further improve predictive accuracy.

## Methods

Our methodology is split into the completed Proof of Concept (P2) and our proposed plan for the final model (P3).

### P2: Proof of Concept Model (Completed)

To validate our project's feasibility, we built a complete data pipeline in `main.ipynb`.

1.  **Task Formulation:** We treat the problem as a ranking task. For each step in a human's path (e.g., from article `A` to `B`, with goal `G`), the model must rank all available links on page `A`. The link `B` is the positive sample, and all other links are negative samples.
2.  **Feature Engineering:** We generated a feature vector for each candidate link based on:
    * **Semantic Features:** `sim_source_candidate` (cosine similarity of `A` and `B`), `sim_candidate_goal` (similarity of `B` and `G`).
    * **Shortest Path Features:** `dist_candidate_goal` (distance from `B` to `G`), `is_closer` (does `B` move closer to `G` than `A`?).
    * **Topology Features:** `pagerank` (centrality of `B`), `out_degree` (hub status of `B`).
3.  **Model:** A `LogisticRegression` model with `class_weight='balanced'` to handle the fact that only ~3% of links are positive samples.
4.  **Evaluation:** We use **Mean Reciprocal Rank (MRR)**. Our model achieved an **MRR of 0.4556**, demonstrating that, on average, the correct human-clicked link is ranked very highly by our simple model.

### P3: Proposed Final Models

We will build on our PoC by implementing and comparing several models of increasing complexity.

1.  **Baseline (Semantic-Only)**
    * We will calculate a score for every candidate link based purely on its semantic similarity to the goal:
    * $score(candidate) = cos\_sim(emb(candidate), emb(goal))$
    * We will choose the candidate with the highest score. This tests the "people choose articles semantically closer to the goal" hypothesis.
2.  **Heuristic Model (Semantic + Hubs)**
    * This model incorporates the "hub bias" (e.g., players look for broad articles like "Science" or "History" at the start).
    * $score(candidate) = \alpha \cdot cos\_sim(candidate, goal) + (1-\alpha) \cdot (outdegree(candidate) / max\_outdegree)$
    * The weighting parameter $\alpha$ can be dynamic:
        * **Early in game:** More weight on `outdegree` (to find hubs).
        * **Later in game:** More weight on `cos\_sim` (to "zoom in" on the target).
3.  **Trained Model (Graph Neural Network)**
    * This is our final proposed architecture. We will replace the Logistic Regression model with a Graph Neural Network (e.g., GraphSAGE) using PyTorch Geometric.
    * **Node Features:** The pre-computed SBERT embeddings from our PoC.
    * **Task:** The GNN will learn to combine graph structure and node semantics to predict the edge (link) a user will take, given the context of the source and goal nodes. This model will be trained end-to-end on the historical data from `paths_finished`.

## Proposed Timeline - EXAMPLE (to change)

| Week | Dates | Task(s) | Owner(s) |
| :--- | :--- | :--- | :--- |
| **P2** | Nov 03 - Nov 07 | Complete PoC notebook, finalize and submit `README.md`. | All |
| Week 1 | Nov 10 - Nov 16 | P2 Feedback. Refine data pipeline to parse HTML files and extract `link_position` feature. | All |
| Week 2 | Nov 17 - Nov 23 | Implement Baseline and Heuristic models. Establish final evaluation script. | All |
| Week 3-4 | Nov 24 - Dec 07 | Implement GNN model (e.g., GraphSAGE). Begin training and tuning. | All |
| Week 5 | Dec 08 - Dec 14 | Run final experiments, compare all models (PoC, Baseline, Heuristic, GNN). Generate plots and results. | All |
| Week 6 | Dec 15 - Dec 19 | **Final Deadline (Dec 19)**. Write final report and clean/document code for submission. | All |

## Organization Within the Team - EXAMPLE (to change)

* **?:** Lead on data pipelines, feature engineering (including new HTML features), and data analysis.
* **?:** Lead on Baseline/Heuristic model implementation and refining the evaluation framework (MRR).
* **?:** Lead on GNN architecture, implementation (PyTorch Geometric), and experimentation.
* **All:** Jointly responsible for analysis, report writing, and final code review.

---

## Appendix

### Repository Organisation

### Note on P2 Milestone Creation

We used GenAI tools to assist in the creation of helper functions for the `src` modules and to structure the `main.ipynb` file.

### Issues that we faced

The primary challenge in P2 was **data formatting**. The dataset was a mix of different formats (CSV, TSV, and fixed-width) despite all files having `.tsv` or `.txt` extensions. This made parsing in the `data_loader.py` module non-trivial and required custom logic for each file (e.g., the distance matrix was a single-character-per-field file, while others were tab- or comma-delimited).

### Questions for TAs - EXAMPLE (to change)

* Given our task (edge-level prediction in a GNN) and our plan to use pre-computed node features (embeddings), would you recommend a specific GNN framework (e.g., PyTorch Geometric vs. DGL) or model (e.g., GCN, GraphSAGE, GAT)?