# Human-Like Next-Link Prediction

**Team:** Szymon, Jendrik, Patrik
**Final Code avaialable on the `main` branch**

## Abstract

Humans do not navigate Wikipedia by calculating shortest paths; they rely on semantic intuition, contextual clues, visual layout, and general knowledge (West, Pineau, and Precup 2009). This project aims to predict the *next link* a human will click in the Wikispeedia "pathfinding" game, given a current article and an end goal. Our core motivation is to model this human navigation strategy using a hybrid approach. We combine modern NLP embeddings (SBERT) to understand *what* an article is about with graph-structural awareness (Graph Neural Networks) to understand *where* it is. Our final model achieves a Mean Reciprocal Rank (MRR) of **0.49**, demonstrating that blending semantic deep learning with explicit distance heuristics significantly outperforms purely rule-based baselines.

## Contributions

The contribution of this project lies in developing a human-like next-link prediction model that goes beyond traditional graph or shortest-path approaches.
Unlike prior work that relied primarily on click frequency or path length, this project integrates diverse signals:
1.  **Visual Bias:** We parse HTML to extract the vertical position of links, validating the hypothesis that users favor links appearing earlier in the text.
2.  **Hybrid Architecture:** We introduce a Hybrid GNN (Graph Attention Network) that fuses learned node embeddings with explicitly injected shortest-path distances.
3.  **Comparative Analysis:** We provide a rigorous benchmark of heuristics, linear models, and deep graph networks, showing that deep learning can effectively approximate human intuition when augmented with structural priors.

## Dataset

We use the Wikispeedia dataset (West, Pineau, and Precup 2009), available at [SNAP](https://snap.stanford.edu/data/wikispeedia.html). The dataset contains:
* **Articles:** 4,604 plaintext and HTML files.
* **Links:** Network structure (~120k links).
* **Paths:** Over 50,000 finished human navigation paths.

### Preprocessing
1.  **Text:** We embedded article content using `sentence-transformers` (all-MiniLM-L6-v2).
2.  **Graph:** We computed PageRank, Out-Degree, and All-Pairs Shortest Paths (BFS).
3.  **Visuals:** We parsed ~4,600 HTML files to extract the normalized `link_position` (0.0 = top, 1.0 = bottom) for every link in the graph.

## Methods

We formulated the problem as a ranking task: given a current article $s$ and a target $g$, rank all neighbors $c \in \mathcal{N}(s)$ by the probability of being clicked.

### 1. Logistic Regression (PoC)
A supervised linear model trained on a rich feature set:
* **Semantic:** Cosine similarity (SBERT).
* **Structural:** PageRank, Out-Degree, Shortest-Path Distance.
* **Visual:** HTML Link Position.
* **Result:** MRR $\approx 0.47$. This strong baseline highlights the predictive power of manually engineered features.

### 2. Heuristic Model
A baseline combining semantic similarity and hub-seeking behavior.
$$\text{Score}(c) = \alpha \cdot \cos(\mathbf{v}_c, \mathbf{v}_g) + (1 - \alpha) \cdot \frac{\text{deg}(c)}{\max(\text{deg})}$$
This model tests the assumption that users simply click "semantically close" or "popular" links.

### 3. Hybrid Graph Neural Networks (Final Model)
To capture non-linear dependencies, we implemented **GraphSAGE** and **GATv2** using PyTorch Geometric.
* **Hybrid Injection:** Standard GNNs struggle to learn global graph distances. We solved this by injecting the scalar `shortest_path_distance` directly into the final prediction layer, alongside the GNN-learned node embeddings.
* **Result:** The **Hybrid GraphSAGE** achieved the best performance (MRR $\approx 0.512$), effectively learning to balance visual, semantic, and structural cues.

## Results

| Model | Validation MRR |
| :--- |:---------------|
| Heuristic (Semantic + Hubs) | 0.1776         |
| Logistic Regression (PoC) | 0.4736         |
| Hybrid GATv2 | 0.4910         |
| **Hybrid GraphSAGE** | **0.5117**     |

## Member Contributions

* **Szymon:** Data processing pipeline, initial GNN approaches and improvements
* **Jendrik:** Data processing pipeline, heuristic implementation 
* **Patrik:** Improvement of GNN models

## Repository Organisation


## References
West, Robert, Joelle Pineau, and Doina Precup. 2009. Wikispeedia: An Online Game for Inferring Semantic Distances between Concepts. In Proceedings of the 21st International Joint Conference on Artificial Intelligence (IJCAI), 1598–1603.
https://snap.stanford.edu/data/wikispeedia.html

## Appendix


### Repository Organisation
Here is the overview of the repository structure:

```
project-milestone-p2-group2/
├── data/
│ ├── plaintext_articles/
│ ├── wikispeedia_articles_html/
│ └── wikispeedia_paths-and-graph/
│
├── results/ (images with plots and data inside each directory)
│ ├── gnn_p3/
│ ├── heuristic_p3/
│ └── html_p3/
│ └── logistic regression_p2/
│
├── src/
│ ├── **init**.py
│ ├── data_loader.py
│ ├── evaluation.py
│ └── feature_extractor.py
│ └── gnn_model.py
│ └── gnn_utils.py
│ └── heuristic.py
│ └── html_processor.py
│
├── .gitignore
├── main.ipynb
└── README.md (this file)
```


Directory data contains the whole provided dataset of Wikispeedia.
The HTML articles are not pushed to the repo (gitignore) since the files are too big, so that we keep them locally.


src/ directory contains all the Python source code, organized as a module.


main.ipynb Is the main Jupyter Notebook where the complete analysis, model training, and evaluation pipeline is executed and documented. This is the primary file for presenting our findings.
