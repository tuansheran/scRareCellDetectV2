## scRareCellDetect

This project applies Graph Neural Networks (GNNs) to identify rare cell populations in high-dimensional single-cell RNA sequencing (scRNA-seq) data. It constructs a graph from gene expression profiles where nodes represent cells and edges are based on similarity using FAISS nearest neighbor search.

### Key Features:

1.Efficient graph construction from sparse gene expression matrices

2.Unsupervised GNN training with contrastive loss

3.Embedding-based clustering for rare cell detection

### Techniques Used:

1.GraphSAGE

2.FAISS

3.CUDA GPU Optimization 

