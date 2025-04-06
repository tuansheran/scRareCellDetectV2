import faiss
import torch
import numpy as np
from .utils import get_model, get_embeddings, get_cluster, get_count, get_plot, get_graph, clean_and_split_data
from interface import get_model
from .graphSAGE import GraphSAGE

ALLOWED_MODELS = {"model_1", "model_2", "model_3"}

def cell_detection(model, data, data_threshold, cluster, count, plot):
    
    if model not in ALLOWED_MODELS:
        raise ValueError(f" Model '{model}' is not supported.\n✅ Choose from: {', '.join(sorted(ALLOWED_MODELS))}")
    
    if data is None:
        raise ValueError("Please provide input data with `.data` as a NumPy array.")

    print(f"✅ Loading model: {model}")
    loaded_model = get_model(model)

    print("🔧 Constructing graph from data...")
    pyg_data = get_graph(data, data_threshold)

    print("📐 Getting embeddings...")
    embeddings = get_embeddings(loaded_model, pyg_data.x, pyg_data.edge_index)

    if cluster == 'yes':
        print("🔍 Performing clustering...")
        labels = get_cluster(embeddings)

        if count == 'yes':
            print("🔢 Cluster counts:")
            get_count(labels)

        if plot == 'yes':
            print("📊 Plotting gene expression by cluster...")
            get_plot(labels, data.data)


data = clean_and_split_data('scRNA.mtx', 100000)
cell_detection(
    model='model_3',
    data=data,
    data_threshold=200,
    cluster='yes',
    count='yes',
    plot='yes',
)