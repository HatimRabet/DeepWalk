import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
import urllib.request
import zipfile


def load_dataset(dataset_name="cora"):
    if dataset_name == "karate":
        G = nx.read_weighted_edgelist('data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
        print(G.number_of_nodes())
        print(G.number_of_edges())

        n = G.number_of_nodes()

        # Loads the class labels
        class_labels = np.loadtxt('data/karate_labels.txt', delimiter=',', dtype=np.int32)
        idx_to_class_label = dict()
        for i in range(class_labels.shape[0]):
            idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

        y = list()
        for node in G.nodes():
            y.append(idx_to_class_label[node])

        y = np.array(y)
        y = torch.tensor(y).long()
        n_class = 2

        adj = nx.adjacency_matrix(G) # Obtains the adjacency matrix
        adj = normalize_adjacency(adj) # Normalizes the adjacency matrix

        # Set the feature of all nodes to the same value
        # features = np.eye(n) 
        features = np.ones((n, n)) # Generates node features
    elif dataset_name == "cora":
        idx_features_labels = np.genfromtxt("data/cora.content", dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = features.todense()
        features /= features.sum(1).reshape(-1, 1)
        
        class_labels_str = idx_features_labels[:, -1]
        idx = 0
        class_to_idx = dict()

        for label in class_labels_str:
            if not label in class_to_idx:
                class_to_idx[label] = idx
                idx += 1
        
        print("Number of Classes : ", idx)
        class_labels = [class_to_idx[label] for label in class_labels_str]
        y = torch.tensor(class_labels).long()
        le = LabelEncoder()
        class_labels = le.fit_transform(class_labels)

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("data/cora.cites", dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(class_labels.size, class_labels.size), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        n = adj.shape[0] # Number of nodes
        n_class = np.unique(class_labels).size # Number of classes

        adj = normalize_adjacency(adj) # Normalize adjacency matrix
    
    elif dataset_name == "amazon":
        # Create data directory if it doesn't exist
        data_path = "../data/amazon/"
        os.makedirs(data_path, exist_ok=True)
        
        # Download the dataset if files don't exist
        amazon_url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz"
        local_file = os.path.join(data_path, "amazon_computer.npz")
        
        if not os.path.exists(local_file):
            print("Downloading Amazon Computer dataset...")
            urllib.request.urlretrieve(amazon_url, local_file)
        
        local_file = os.path.join(data_path, "amazon_computer.npz")
        data = np.load(local_file, allow_pickle=True)
        
        # Print available keys for debugging
        print(f"Available keys in the NPZ file: {data.files}")
        
        # Extract features from CSR format
        attr_data = data['attr_data']
        attr_indices = data['attr_indices']
        attr_indptr = data['attr_indptr']
        attr_shape = data['attr_shape']
        
        # Extract features, adjacency matrix, and labels
        features = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape, dtype=np.float32)
        features = features.todense()
        # Normalize features
        features = features / (features.sum(1).reshape(-1, 1) + 1e-6)  # Add small constant to avoid division by zero
        
        # Extract adjacency matrix from CSR format
        adj_data = data['adj_data']
        adj_indices = data['adj_indices']
        adj_indptr = data['adj_indptr']
        adj_shape = data['adj_shape']
        
        adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape, dtype=np.float32)
        
        # Extract class labels
        class_labels = data['labels'] if 'labels' in data else data['label']
        class_labels = np.array(class_labels)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        y = torch.tensor(class_labels).long().to(device)
        le = LabelEncoder()
        class_labels = le.fit_transform(class_labels)
        
        # Build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
        n = adj.shape[0]  # Number of nodes
        n_class = np.unique(class_labels).size  # Number of classes
        
        # Normalize adjacency matrix
        adj = normalize_adjacency(adj)
    # elif dataset_name == "citeseer":
    #     # Create data directory if it doesn't exist
    #     data_path = "../data/citeseer/"
    #     os.makedirs(data_path, exist_ok=True)
        
    #     # Download the dataset if files don't exist
    #     citeseer_url = "https://github.com/kimiyoung/planetoid/raw/master/data/citeseer"
    #     local_file = os.path.join(data_path, "citeseer.npz")
        
    #     if not os.path.exists(local_file):
    #         print("Downloading Citeseer dataset...")
    #         # Download files
    #         for ext in ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]:
    #             url = f"{citeseer_url}/ind.citeseer.{ext}"
    #             local_file_temp = os.path.join(data_path, f"ind.citeseer.{ext}")
    #             urllib.request.urlretrieve(url, local_file_temp)
            
    #         # Process files and create a single npz file
    #         print("Processing Citeseer dataset...")
    #         adj, features, labels = process_citeseer_data(data_path)
    #         np.savez(local_file, adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
    #                 adj_shape=adj.shape, attr_data=features.data, attr_indices=features.indices,
    #                 attr_indptr=features.indptr, attr_shape=features.shape, labels=labels)
        
    #     # Load processed data
    #     data = np.load(local_file, allow_pickle=True)
        
    #     # Extract features from CSR format
    #     attr_data = data['attr_data']
    #     attr_indices = data['attr_indices']
    #     attr_indptr = data['attr_indptr']
    #     attr_shape = data['attr_shape']
        
    #     # Extract features, adjacency matrix, and labels
    #     features = sp.csr_matrix((attr_data, attr_indices, attr_indptr), shape=attr_shape, dtype=np.float32)
    #     features = features.todense()
    #     # Normalize features
    #     features = features / (features.sum(1).reshape(-1, 1) + 1e-6)  # Add small constant to avoid division by zero
        
    #     # Extract adjacency matrix from CSR format
    #     adj_data = data['adj_data']
    #     adj_indices = data['adj_indices']
    #     adj_indptr = data['adj_indptr']
    #     adj_shape = data['adj_shape']
        
    #     adj = sp.csr_matrix((adj_data, adj_indices, adj_indptr), shape=adj_shape, dtype=np.float32)
        
    #     # Extract class labels
    #     class_labels = data['labels']
    #     class_labels = np.array(class_labels)
        
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     y = torch.tensor(class_labels).long().to(device)
    #     le = LabelEncoder()
    #     class_labels = le.fit_transform(class_labels)
        
    #     # Build symmetric adjacency matrix
    #     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        
    #     n = adj.shape[0]  # Number of nodes
    #     n_class = np.unique(class_labels).size  # Number of classes
        
    #     # Normalize adjacency matrix
    #     adj = normalize_adjacency(adj)

    elif dataset_name == "citeseer":
        # Create data directory if it doesn't exist
        data_path = "../data/citeseer/"
        os.makedirs(data_path, exist_ok=True)
        
        # Use a direct download for processed Citeseer dataset
        citeseer_url = "https://github.com/kimiyoung/planetoid/raw/master/data/ind.citeseer.x"
        local_file = os.path.join(data_path, "citeseer.npz")
        
        try:
            # If we can import torch_geometric, use that to get the data
            from torch_geometric.datasets import Planetoid
            import torch_geometric.transforms as T
            
            print("Loading Citeseer dataset from PyTorch Geometric...")
            dataset = Planetoid(root=data_path, name='Citeseer', transform=T.NormalizeFeatures())
            data = dataset[0]
            
            # Convert to numpy and sparse matrices
            features = data.x.numpy()
            edge_index = data.edge_index.numpy()
            labels = data.y.numpy()
            
            # Create adjacency matrix from edge index
            n = features.shape[0]
            adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                                (edge_index[0], edge_index[1])),
                                shape=(n, n), dtype=np.float32)
            
            # Make the adjacency matrix symmetric
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            
            class_labels = labels
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            y = torch.tensor(class_labels).long().to(device)
            
            # Normalize adjacency matrix
            adj = normalize_adjacency(adj)
            
        except ImportError:
            # Fallback to direct URL for a processed version
            print("PyTorch Geometric not available. Downloading pre-processed Citeseer dataset...")
            citeseer_url = "https://github.com/abdouskamel/graph-data/raw/main/citeseer.npz"
            local_file = os.path.join(data_path, "citeseer.npz")
            
            if not os.path.exists(local_file):
                print(f"Downloading from {citeseer_url}")
                urllib.request.urlretrieve(citeseer_url, local_file)
            
            data = np.load(local_file, allow_pickle=True)
            
            # Extract features
            features = data['features'] if 'features' in data else data['x']
            
            # Extract adjacency matrix
            adj = data['adj_matrix'] if 'adj_matrix' in data else data['adj'] 
            if not isinstance(adj, sp.spmatrix):
                adj = sp.csr_matrix(adj)
            
            # Extract labels
            class_labels = data['labels'] if 'labels' in data else data['y']
            class_labels = np.array(class_labels).flatten()
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            y = torch.tensor(class_labels).long().to(device)
            
            # Normalize adjacency matrix if not already normalized
            adj = normalize_adjacency(adj)
    
    return features, adj, class_labels, y


def normalize_adjacency(A):
    n = A.shape[0]
    A_hat = A + sp.identity(n)
    A_hat = sp.csr_matrix(A_hat)
    D = sp.diags(sp.csr_matrix.sum(A_hat, axis=1).A1)
    A_normalized = D.power(-0.5) @ A_hat @ D.power(-0.5)
    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def process_citeseer_data(data_path):
    """
    Process the Citeseer dataset from raw files and return adjacency matrix, features, and labels.
    """
    import pickle
    import networkx as nx
    
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    
    for name in names:
        with open(os.path.join(data_path, f"ind.citeseer.{name}"), 'rb') as f:
            objects.append(pickle.load(f, encoding='latin1'))
    
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    
    # Load test indices
    test_idx_reorder = []
    with open(os.path.join(data_path, "ind.citeseer.test.index"), 'r') as f:
        for line in f:
            test_idx_reorder.append(int(line.strip()))
    
    test_idx_range = np.sort(test_idx_reorder)
    
    # Fix incomplete data for citeseer
    if 'citeseer' in data_path:
        # Fix isolated nodes
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        
        # Add missing entries for ally
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    
    # Combine all features and labels
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    
    # Create graph from adjacency dict
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
    # Combine all labels
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    # Get the class with maximum probability
    labels = np.argmax(labels, axis=1)
    
    return adj, features, labels