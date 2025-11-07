# Human-Like Next-Link Prediction

**Team:** Szymon Zalewski, Jendrik Engels, Patrik Kolozsvári

## Abstract
Humans do not navigate Wikipedia by calculating shortest paths, they rely on semantic intuition, contextual clues, and general knowledge. This project aims to predict the next link a human will click in the Wikispeedia "pathfinding" game, given a current article and an end goal. Our core motivation is to model this human navigation strategy. We will build and evaluate models that uses modern NLP embeddings (to understand what an article is about. Our goal is to create a model that navigates Wikipedia more like a human and less like an algorithm.

## Contributions
The contribution of this project lies in developing a human-like next-link prediction model that goes beyond traditional graph or shortest-path approaches.
Unlike prior work that relied primarily on click frequency or path length (West, Pineau, and Precup 2009), this project integrates semantic and structural signals available to real human players, such as the article content, available hyperlinks, their visual order (link position, the goal article’s semantics and the path that has been used so far.
The model aims to more accurately approximate how humans navigate Wikipedia and provide new insights into cognitive navigation behaviour and improving machine models for web navigation, education and recommendation systems.

## Used dataset
For P2 Milestone we already used all kinds of data that the Wikispeedia dataset provides. Combining different types was useful to create strong model predictions since those are based on multiple sources (multiple data types that the dataset offers). We didn’t add anything extra, because we think that that data should be enough to create a strong model. The model will make use of the plaintext of the articles and the html files have been used to extract a new feature which is the position of a link in a given article. The data about the finished paths will be used to test our approach.

## Data preprocessing
For the preprocessing we first loaded the .tsv-files from the wikispeedia_paths-and-graph folder, which includes the following data:
- Articles: Dataset containing name of each article
- Links: Dataset containing the outgoing links of a certain article
- Paths finished: Dataset containing Wikispeedia games where a route between start and goal article has been found by a human. All used articles to reach the goal are listed. In addition a rating about difficulty and the required time is provided.
- Paths unfinished: Dataset containing Wikispeedia games where a route between start and goal article could not be found. All used articles until quitting the game are provided. Additionally we have information about the reason of quitting the game (restart or timeout)

We also loaded the shortest paths matrix and the categories .tsv-file. We will not be using this data because a human will not have access to the category and the shortest paths to other articles.

We proceeded by linking the plaintext and html text to the articles dataframe and added two additional columns for the length of the plaintext and the html text. We also added two columns to describe how many articles are linking to a given article (indegree) and to how many articles a given article is linking (outdegree). In the links dataframe we added a column describing the relative position of a link in an article. The value will lie between 0 and 1. A value closer to 0 indicates that link is positioned closer to the beginning, while 1 indicates the link is positioned closer to the end of the article. 

## Methods
### Achievements for Milestone 2
To validate our project's feasibility, we built a complete data pipeline in `main.ipynb`.
1.  **Task Formulation:** We treat the problem as a ranking task. For each step in a human's path (e.g., from article `A` to `B`, with goal `G`), the model must rank all available links on page `A`. The link `B` is the positive sample, and all other links are negative samples.
2.  **Feature Engineering:** We generated a feature vector for each candidate link based on:
   * **Semantic Features:** `sim_source_candidate` (cosine similarity of `A` and `B`), `sim_candidate_goal` (similarity of `B` and `G`).
   * **Shortest Path Features:** `dist_candidate_goal` (distance from `B` to `G`), `is_closer` (does `B` move closer to `G` than `A`?).
   * **Topology Features:** `pagerank` (centrality of `B`), `out_degree` (hub status of `B`).
3.  **Model:** A `LogisticRegression` model with `class_weight='balanced'` to handle the fact that only ~3% of links are positive samples.
4.  **Evaluation:** We use **Mean Reciprocal Rank (MRR)**. Our model achieved an **MRR of 0.4556**, demonstrating that, on average, the correct human-clicked link is ranked very highly by our simple model.

### Plan for Milestone 3
We will expand our PoC by implementing and comparing several models of increasing complexity.
1.  **Baseline (Semantic-Only)**
   * We will calculate a score for every candidate link based purely on its semantic similarity to the goal:
   * $score(candidate) = cos\_sim(emb(candidate), emb(goal))$
   * We will choose the candidate with the highest score. This tests the "people choose articles semantically closer to the goal" hypothesis.
2.  **Heuristic Model (Semantic + Hubs)**
   * This model incorporates the "hub bias" (e.g., players look for broad articles like "Science" or "History" at the start).
   * $score(candidate) = \alpha \cdot cos\_sim(candidate, goal) + (1-\alpha) \cdot (outdegree(candidate) / max\_outdegree)$
   * The weighting parameter $\alpha$ can be dynamic:
      * **Early in game:** More weight on `outdegree` (to find hubs).
      * **Later in game:** More weight on `cos_sim` (to "zoom in" on the target).
3.  **Trained Model (Graph Neural Network)**
   * This is our final proposed architecture. We will replace the Logistic Regression model with a Graph Neural Network (e.g., GraphSAGE) using PyTorch Geometric.
   * **Node Features:** The pre-computed SBERT embeddings from our PoC.
   * **Task:** The GNN will learn to combine graph structure and node semantics to predict the edge (link) a user will take, given the context of the source and goal nodes. This model will be trained end-to-end on the historical data from `paths_finished`.
4.  **Final Model:** Our final contribution (for P3) will be a Graph Neural Network (GNN) architecture that learns the optimal, non-linear balance between these semantic and structural features.

## Proposed Timeline

| Week       | Dates           | Task(s)                                                                                                                         |
| :-- | :-- | :-- |
| Week 1 & 2 | Nov 7 - Nov 23  | Receive and evaluate P2 Feedback and implement Baseline and Heuristic models. Establish final evaluation script.                |
| Week 3 & 4 | Nov 24 - Dec 05 | Implement the GNN model. Run final experiments, compare all models (PoC, Baseline, Heuristic, GNN). Generate plots and results. |
| Week 5 & 6 | Dec 05 - Dec 19 | Buffer for potential fixes and problems                                                                                         |


## References
West, Robert, Joelle Pineau, and Doina Precup. 2009. Wikispeedia: An Online Game for Inferring Semantic Distances between Concepts. In Proceedings of the 21st International Joint Conference on Artificial Intelligence (IJCAI), 1598–1603.

## Appendix
### Repository Organisation
Here is the overview of the repository structure:

<img width="309" height="395" alt="image" src="https://github.com/user-attachments/assets/1e643581-6f8e-4076-a65c-2e43a95fd989" />

- Directory data contains the whole provided dataset of Wikispeedia.
- The HTML are only kept locally because they are too big.
- src/ directory contains all the Python source code, organized as a module.
- main.ipynb Is the main Jupyter Notebook where the preprocessing, complete analysis, model training, and evaluation pipeline is executed and documented. This is the primary file for presenting our findings.

## Note on P2 Milestone Creation
We used GenAI tools to assist in the creation of helper functions for the `src` modules and to structure the `main.ipynb` file.
