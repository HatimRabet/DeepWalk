# import networkx as nx
# import numpy as np
# from deepwalk.model import DeepWalk
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# # # Loads the web graph
# # G = nx.read_weighted_edgelist('../data/web_sample.edgelist', delimiter=' ', create_using=nx.Graph())
# # print("Number of nodes:", G.number_of_nodes())
# # print("Number of edges:", G.number_of_edges())


# # ############## Task 3
# # # Extracts a set of random walks from the web graph and feeds them to the Skipgram model
# # n_dim = 128
# # n_walks = 10
# # walk_length = 20

# # model = DeepWalk(G, n_walks, walk_length, n_dim)

# ############## Task 4
# # Visualizes the representations of the 100 nodes that appear most frequently in the generated walks
# def visualize(model, n, dim):
#     model.wv.sort_by_descending_frequency()
#     nodes = model.wv.index_to_key[:n]
#     DeepWalk_embeddings = np.empty(shape=(n, dim))
    
#     for idx, node in enumerate(nodes):
#         DeepWalk_embeddings[idx] = model.wv[node]

#     my_pca = PCA(n_components=10)
#     my_tsne = TSNE(n_components=2)

#     vecs_pca = my_pca.fit_transform(DeepWalk_embeddings)
#     vecs_tsne = my_tsne.fit_transform(vecs_pca)

#     fig, ax = plt.subplots()
#     ax.scatter(vecs_tsne[:,0], vecs_tsne[:,1],s=3)
#     for x, y, node in zip(vecs_tsne[:,0] , vecs_tsne[:,1], nodes):     
#         ax.annotate(node, xy=(x, y), size=8)
#     fig.suptitle('t-SNE visualization of node embeddings',fontsize=30)
#     fig.set_size_inches(20,15)
#     plt.savefig('embeddings.pdf')  
#     plt.show()


# # visualize(model, 100, n_dim)


import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# def visualize(model, G, labels=None, n=100, dim=128, figsize=(20, 15), 
#                          save_path=None, show_plot=True, annotate=True, marker_size=50):
#     """
#     Visualize node embeddings with unified colors for nodes of the same label.
    
#     Parameters:
#     -----------
#     model : DeepWalk model
#         The trained DeepWalk model containing node embeddings
#     G : networkx.Graph
#         The input graph
#     labels : dict or None
#         Dictionary mapping nodes to their class labels. If None, no color coding is used.
#     n : int
#         Number of nodes to visualize (most frequent ones)
#     dim : int
#         Dimensionality of the embeddings
#     figsize : tuple
#         Figure size (width, height) in inches
#     save_path : str or None
#         Path to save the visualization. If None, the figure is not saved.
#     show_plot : bool
#         Whether to display the plot
#     annotate : bool
#         Whether to annotate nodes with their IDs
#     marker_size : int
#         Size of node markers
        
#     Returns:
#     --------
#     fig : matplotlib.figure.Figure
#         The figure object
#     embeddings_2d : numpy.ndarray
#         The 2D embeddings after dimensionality reduction
#     nodes_list : list
#         List of nodes that were visualized
#     """
#     # Sort nodes by frequency in the random walks
#     model.wv.sort_by_descending_frequency()
    
#     # Get the most frequent nodes
#     nodes = model.wv.index_to_key[:min(n, len(model.wv.index_to_key))]
#     nodes_list = [node for node in nodes]
    
#     # Extract embeddings for these nodes
#     embeddings = np.empty(shape=(len(nodes), dim))
#     for idx, node in enumerate(nodes):
#         embeddings[idx] = model.wv[node]
    
#     # Dimensionality reduction: PCA followed by t-SNE
#     pca = PCA(n_components=2)
#     tsne = TSNE(n_components=2, random_state=42)
    
#     # Apply dimensionality reduction
#     embeddings_pca = pca.fit_transform(embeddings)
#     embeddings_2d = tsne.fit_transform(embeddings_pca)
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=figsize)
    
#     # Prepare colors based on labels if provided
#     if labels is not None:
#         # Get unique labels
#         node_labels = [labels[i] for i, node in enumerate(nodes_list)]
#         unique_labels = sorted(set(node_labels))
        
#         # Create color map
#         cmap = plt.cm.get_cmap('tab10', len(unique_labels))
#         colors = [cmap(unique_labels.index(label)) for label in node_labels]
        
#         # Create scatter plot with label colors
#         scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
#                              c=node_labels, cmap=cmap, s=marker_size)
        
#         # Add legend
#         legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
#                            markerfacecolor=cmap(i), label=f'Class {label}', markersize=10) 
#                            for i, label in enumerate(unique_labels)]
#         ax.legend(handles=legend_elements, title="Classes", loc="best")
#     else:
#         # No labels provided, use single color
#         scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=marker_size)
    
#     # Annotate nodes if requested
#     if annotate:
#         for i, node in enumerate(nodes_list):
#             ax.annotate(node, xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]), 
#                         xytext=(5, 2), textcoords='offset points', 
#                         ha='right', va='bottom', size=8)
    
#     # Set title and layout
#     ax.set_title('t-SNE visualization of node embeddings', fontsize=24)
#     plt.tight_layout()
    
#     # Save figure if path is provided
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')
    
#     # Show plot if requested
#     if show_plot:
#         plt.show()
    
#     return fig, embeddings_pca, nodes_list

def visualize(X, Y, n, dim, pic_path):
    # X, Y = read_node_label(label_file)
    # Sort nodes by frequency in the random walks
    # dw.model.wv.sort_by_descending_frequency()
    
    # # Get the most frequent nodes
    # nodes = dw.model.wv.index_to_key[:min(n, len(dw.model.wv.index_to_key))]
    # nodes_list = [node for node in nodes]
    # print(nodes_list)
    
    # # Extract embeddings for these nodes
    # embeddings = np.empty(shape=(len(nodes), dim))
    # for idx, node in enumerate(nodes):
    #     embeddings[idx] = dw.model.wv[node]

    # emb_list = []
    # for k in X:
    #     emb_list.append(embeddings[k])
    # emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(X)

    color_idx = {}
    for i in range(len(Y)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)

    plt.legend()
    plt.savefig(pic_path)
    plt.show()