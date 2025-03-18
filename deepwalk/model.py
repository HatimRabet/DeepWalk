import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec

class DeepWalk:
    def __init__(self, num_walks=10, walk_length=80, n_dim=128):
        """
        Initialize DeepWalk with default parameters
        
        Parameters:
        -----------
        num_walks : int
            Number of random walks per node
        walk_length : int
            Length of each random walk
        n_dim : int
            Dimensionality of the output embeddings
        """
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.n_dim = n_dim
        self.model = None
    
    def random_walk(self, G, node):
        """
        Simulates a random walk starting from node
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
        node : node
            The starting node for the walk
            
        Returns:
        --------
        walk : list
            List of nodes in the walk (as strings)
        """
        walk = [node]

        while len(walk) < self.walk_length:
            curr_node = walk[-1]
            curr_neighbors = list(G.neighbors(curr_node))
            if not curr_neighbors:
                next_node = curr_node
            else:
                next_node = np.random.choice(curr_neighbors)
            walk.append(next_node)
        
        walk = [str(node) for node in walk]
        return walk

    def generate_walks(self, G):
        """
        Runs multiple random walks from each node
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
            
        Returns:
        --------
        walks : list
            List of random walks
        """
        walks = []
        all_nodes = list(G.nodes)

        for node in all_nodes:
            for _ in range(self.num_walks):
                walk = self.random_walk(G, node)
                walks.append(walk)
        
        permuted_walks = np.random.permutation(walks)
        return permuted_walks.tolist()

    def fit(self, G, window=8, min_count=0, workers=8, epochs=5):
        """
        Simulates walks and learns node representations
        
        Parameters:
        -----------
        G : networkx.Graph
            The input graph
        window : int
            Window size for Word2Vec
        min_count : int
            Minimum count for Word2Vec
        workers : int
            Number of worker threads for Word2Vec
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        self : DeepWalk
            The fitted model
        """
        print("Generating walks")
        walks = self.generate_walks(G)

        print("Training word2vec")
        self.model = Word2Vec(vector_size=self.n_dim, window=window, min_count=min_count, 
                             sg=1, workers=workers, hs=1)
        self.model.build_vocab(walks)
        self.model.train(walks, total_examples=self.model.corpus_count, epochs=epochs)

        return self
    
    def get_embeddings(self):
        """
        Returns the learned node embeddings as a dictionary
        
        Returns:
        --------
        embeddings : dict
            Dictionary mapping node IDs to their embeddings
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        embeddings = {}
        for word in self.model.wv.index_to_key:
            embeddings[word] = self.model.wv[word]
        return embeddings
    

