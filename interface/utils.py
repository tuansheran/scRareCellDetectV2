import os
import torch
import faiss 
import numpy as np
from scipy.io import mmread
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from .graphSAGE import model1, model3, model4


def get_graph(data, threshold):

    gene_expression = data.data
    
    x = np.asarray(gene_expression, dtype=np.float32)
    x = x.reshape(-1, 1)

    gpu_resource_manager = faiss.StandardGpuResources() 
    similarity_object = faiss.IndexFlatL2(1)
    similarity_object_in_gpu = faiss.index_cpu_to_gpu(gpu_resource_manager, 0, similarity_object)


    print(similarity_object_in_gpu.is_trained)  
    print(f"FAISS index type: {type(similarity_object_in_gpu)}") 


    similarity_object_in_gpu.add(x)
    k=2
    distances, indices = similarity_object_in_gpu.search(x, k + 1)
    
    edge_index_list = []
    outliers = []
    
    for i in range(len(gene_expression)):
        nearest_neighbors = indices[i, 1:k+1]  
        neighbor_distances = distances[i, 1:k+1]
        
        for j, dist in zip(nearest_neighbors, neighbor_distances):
            if dist <= threshold ** 2:
                edge_index_list.append((i, j))
            else:
                outliers.append(int(j))
    

    edge_index_np = np.array(edge_index_list).T
    edge_index = torch.tensor(edge_index_np, dtype=torch.long) if edge_index_np.size > 0 else torch.empty((2, 0), dtype=torch.long)

    cleaned_outliers = list(set(outliers))
    print(cleaned_outliers)

    x_tensor = torch.tensor(x, dtype=torch.float32)
    pyg_data = Data(edge_index=edge_index, x=x_tensor)
    return pyg_data



def get_model(model_name):
    
    model_dict = {
        "model_1": model1,
        "model_2": model3,
        "model_3": model4,
    }

    # model = model_dict.get(model_name)
    model = torch.load('./entire_model1.pth', map_location=torch.device("cpu"))
    if model is None:
        raise ValueError(f"Model '{model_name}' not found.")

    model.eval()
    return model


def get_embeddings(model, x, edge_index):
    with torch.no_grad():
        embeddings = model(x, edge_index)
    return embeddings


def get_cluster(embeddings):
    embeddings_np = embeddings.numpy()

    d = embeddings_np.shape[1]  
    k = 10
    kmeans = faiss.Kmeans(d, k, niter=300, gpu=True)  

    kmeans.train(embeddings_np)
    _, labels = kmeans.index.search(embeddings_np, 1)

    labels = torch.tensor(labels.flatten(), device='cuda')
    return labels



def get_count(labels):
    cluster_counts = torch.bincount(labels)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i}: {count.item()} cells")


def get_plot(labels, gene_expression_levels):
    #plot 1
    labels_cpu = labels.cpu().numpy()
    gene_expression_levels_cpu = gene_expression_levels


    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels_cpu)

    for label in unique_labels:
        cluster_cells = gene_expression_levels_cpu[labels_cpu == label]
        num_cells = len(cluster_cells)
        plt.scatter(np.full_like(cluster_cells, label), cluster_cells, alpha=0.5, label=f'Cluster {label} ({num_cells} cells)')


    plt.xlabel('Cluster ID')
    plt.ylabel('Gene Expression Level')
    plt.title('Gene Expression Levels per Cluster')
    plt.legend()
    plt.show()

    #plot2
    plt.figure(figsize=(12, 6))

    for label in np.unique(labels_cpu):
        cluster_cells = gene_expression_levels_cpu[labels_cpu == label]
        y = np.full_like(cluster_cells, label) + np.random.uniform(-0.1, 0.1, size=cluster_cells.shape)

        plt.scatter(cluster_cells, y, alpha=0.5, s=5, label=f'Cluster {label}')

    plt.xlabel('Gene Expression Level')
    plt.ylabel('Cluster ID')
    plt.title('Gene Expression Level per Cluster (Scatter)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def clean_and_split_data(path, max_number):
    rawData = mmread(path)
    coo_matrix = rawData.tocoo()


    if max_number > 1000000:
        raise ValueError("⚠️ max_number exceeds 1 million. Cannot proceed for performance reasons.")
    
    total_nnz = coo_matrix.nnz

    if max_number >= total_nnz:
        raise ValueError(f"⚠️ max_number ({max_number}) must be less than total non-zero elements ({total_nnz})")

    rows = coo_matrix.row
    cols = coo_matrix.col
    data = coo_matrix.data

    selected_indices = np.arange(max_number)

    selected = coo_matrix.__class__(
        (data[selected_indices], (rows[selected_indices], cols[selected_indices])),
        shape=coo_matrix.shape
    )

    return selected


