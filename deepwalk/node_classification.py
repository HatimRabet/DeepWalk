import os
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import scipy.sparse as sp
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from deepwalk.visualization import visualize


def load_graph_dataset(dataset_name, data_dir="./data"):
    """
    Load graph datasets from PyG, OGB, or NetworkX.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'cora', 'citeseer', 'pubmed', 'karate', 'ogbn-arxiv').
        data_dir (str): Directory to store downloaded datasets.

    Returns:
        G (networkx.Graph): Graph object.
        labels (np.ndarray): Node labels.
    """
    
    dataset_name = dataset_name.lower()

    # Planetoid datasets (Cora, Citeseer, PubMed) from PyG
    if dataset_name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root=data_dir, name=dataset_name)
        edge_index = dataset[0].edge_index.numpy()
        labels = dataset[0].y.numpy()

        # Convert edge index to NetworkX graph
        G = nx.Graph()
        G.add_edges_from(edge_index.T)
    elif dataset_name in ["amazon-computers"]:
        dataset = Amazon(root=data_dir, name='computers')
        edge_index = dataset[0].edge_index.numpy()
        labels = dataset[0].y.numpy()

        # Convert edge index to NetworkX graph
        G = nx.Graph()
        G.add_edges_from(edge_index.T)

    # Karate Club (Simple Dataset)
    elif dataset_name == "karate":
        G = nx.karate_club_graph()
        labels = np.array([G.nodes[n]['club'] == 'Mr. Hi' for n in G.nodes])  # Binary label

    # Open Graph Benchmark (OGB) datasets (e.g., 'ogbn-arxiv')
    elif dataset_name.startswith("ogbn"):
        dataset = PygNodePropPredDataset(name=dataset_name, root=data_dir)
        edge_index = dataset[0].edge_index.numpy()
        labels = dataset[0].y.numpy()

        # Convert to NetworkX graph
        G = nx.Graph()
        G.add_edges_from(edge_index.T)

    # Load from edge list (if provided in a .txt or .edgelist file)
    elif os.path.exists(os.path.join(data_dir, f"{dataset_name}.edgelist")):
        G = nx.read_edgelist(os.path.join(data_dir, f"{dataset_name}.edgelist"), nodetype=int)
        labels = np.loadtxt(os.path.join(data_dir, f"{dataset_name}_labels.txt"), dtype=int)

    else:
        raise ValueError(f"Dataset '{dataset_name}' not found! Available: Cora, Citeseer, PubMed, Karate, OGBN.")

    print(f"Loaded dataset: {dataset_name}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Number of classes: {len(set(labels))}")

    return G, labels


def evaluate_graph_embeddings(dataset_name,
                             deepwalk_params={'n_dim': 128, 'n_walks': 10, 'walk_length': 20},
                             spectral_params={'k': 2},
                             random_state=42,
                             train_ratio=0.8,
                             not_visualize=False):
    """
    Evaluate graph embeddings using DeepWalk and spectral clustering approaches.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (without extension)
    deepwalk_params : dict
        Parameters for DeepWalk algorithm
    spectral_params : dict
        Parameters for spectral embedding
    random_state : int
        Random seed for reproducibility
    train_ratio : float
        Ratio of training data (between 0 and 1)
    visualize : bool
        Whether to visualize the graph
        
    Returns:
    --------
    dict : Dictionary containing accuracy results and embeddings
    """
    # File paths    
    G, labels = load_graph_dataset(dataset_name)
    idx_to_class_label = dict()
    for i in range(len(labels)):
        idx_to_class_label[i] = labels[i]
    
    y = list()
    for node in G.nodes():
        y.append(idx_to_class_label[node])
    
    y = np.array(y)

    # Visualize the network if requested
    if not not_visualize:
        plt.figure(figsize=(10, 8))
        nx.draw_networkx(G, with_labels=True, node_color=y)
        plt.title(f"{dataset_name} Network")
        plt.axis('off')
        plt.show()
    
    # Extract DeepWalk parameters
    n = G.number_of_nodes()
    n_dim = deepwalk_params.get('n_dim', 128)
    n_walks = deepwalk_params.get('n_walks', 100)
    walk_length = deepwalk_params.get('walk_length', 10)
    
    # Create and train DeepWalk model
    from deepwalk.model import DeepWalk
    dw = DeepWalk(num_walks=n_walks, walk_length=walk_length, n_dim=n_dim)
    dw.fit(G, epochs=100)
    
    # Get DeepWalk embeddings
    embeddings = np.zeros((n, n_dim))
    for i, node in enumerate(G.nodes()):
        embeddings[i,:] = dw.model.wv[str(node)]
    
    # Split data into train and test sets
    np.random.seed(random_state)
    idx = np.random.permutation(n)
    idx_train = idx[:int(train_ratio*n)]
    idx_test = idx[int(train_ratio*n):]
    
    X_train = embeddings[idx_train,:]
    X_test = embeddings[idx_test,:]
    
    y_train = y[idx_train]
    y_test = y[idx_test]
    
    # Train logistic regression on DeepWalk embeddings
    logisticReg = LogisticRegression(random_state=random_state)
    logisticReg.fit(X_train, y_train)
    y_pred = logisticReg.predict(X_test)
    
    accuracy_dw = accuracy_score(y_test, y_pred)
    print(f"Accuracy of Logistic Regression with DeepWalk: {100 * accuracy_dw:.2f}%")

    # visualize(dw.model, G, labels, 100, n_dim)
    visualize(X_train, y_train, n, n_dim, "amazon-computers_deepwalk")
    
    # Return results
    results = {
        'deepwalk': {
            'embeddings': embeddings,
            'accuracy': accuracy_dw,
            'model': dw
        },
        'data': {
            'graph': G,
            'labels': y,
            'train_idx': idx_train,
            'test_idx': idx_test
        }
    }
    
    return results


if __name__ == "__main__":
    dataset_name = "amazon-computers"  
    G, labels = load_graph_dataset(dataset_name)
    results = evaluate_graph_embeddings(dataset_name=dataset_name)