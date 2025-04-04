## scRareCellDetect


This project applies Graph Neural Networks (GNNs) to identify rare cell populations in high-dimensional single-cell RNA sequencing (scRNA-seq) data. It constructs a graph from gene expression profiles where nodes represent cells and edges are based on similarity using FAISS nearest neighbor search.

Key Features:
Efficient graph construction from sparse gene expression matrices

Unsupervised GNN training with contrastive loss

Embedding-based clustering for rare cell detection

