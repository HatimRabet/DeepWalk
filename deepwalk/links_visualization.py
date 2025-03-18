import networkx as nx
import numpy as np
from deepwalk.model import DeepWalk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Visualizes the representations of the 100 nodes that appear most frequently in the generated walks
def visualize_links(model, n, dim):
    model.wv.sort_by_descending_frequency()
    nodes = model.wv.index_to_key[:n]
    DeepWalk_embeddings = np.empty(shape=(n, dim))
    
    for idx, node in enumerate(nodes):
        DeepWalk_embeddings[idx] = model.wv[node]

    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2)

    vecs_pca = my_pca.fit_transform(DeepWalk_embeddings)
    vecs_tsne = my_tsne.fit_transform(vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(vecs_tsne[:,0], vecs_tsne[:,1],s=3)
    for x, y, node in zip(vecs_tsne[:,0] , vecs_tsne[:,1], nodes):     
        ax.annotate(node, xy=(x, y), size=8)
    fig.suptitle('t-SNE visualization of node embeddings',fontsize=30)
    fig.set_size_inches(20,15)
    plt.savefig('embeddings.pdf')  
    plt.show()

if __name__ == "__main__":
    # Loads the web graph
    G = nx.read_weighted_edgelist('data/web_sample.edgelist', delimiter=' ', create_using=nx.Graph())
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    n_dim = 128
    n_walks = 10
    walk_length = 20

    model = DeepWalk(n_walks, walk_length, n_dim)

    model.fit(G)

    visualize_links(model.model, 100, n_dim)