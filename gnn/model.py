import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        Z_0 = self.fc1(x_in)
        Z_0 = self.relu(torch.mm(adj, Z_0))
        Z_0 = self.dropout(Z_0)
        Z_1 = self.fc2(Z_0)
        Z_1 = self.relu(torch.mm(adj, Z_1))
        
        x = self.fc3(Z_1)
        return F.log_softmax(x, dim=1), Z_1
    

import torch
import torch.nn as nn
import torch.nn.functional as F

# class GATLayer(nn.Module):
#     """GAT layer with support for multi-head attention."""
#     def __init__(self, in_features, out_features, n_heads, dropout, alpha, concat=True):
#         super(GATLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.n_heads = n_heads
#         self.concat = concat
#         self.dropout = dropout
#         self.alpha = alpha

#         if self.concat:
#             self.out_per_head = out_features // n_heads
#             assert out_features % n_heads == 0, "out_features must be divisible by n_heads when concat is True"
#         else:
#             self.out_per_head = out_features

#         self.W = nn.Linear(in_features, self.out_per_head * n_heads, bias=False)
#         self.a = nn.Linear(2 * self.out_per_head, 1, bias=False)

#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#         self.dropout_layer = nn.Dropout(self.dropout)

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.W.weight)
#         nn.init.xavier_uniform_(self.a.weight)

#     def forward(self, h, adj):
#         n_nodes = h.size(0)
        
#         # Transform features: [n_nodes, in_features] -> [n_nodes, n_heads * out_per_head]
#         Wh = self.W(h)
#         Wh = Wh.view(n_nodes, self.n_heads, self.out_per_head)  # [n_nodes, n_heads, out_per_head]

#         # Compute attention scores
#         Wh_repeated = Wh.unsqueeze(1).expand(n_nodes, n_nodes, self.n_heads, self.out_per_head)
#         Wh_repeated_T = Wh.unsqueeze(0).expand(n_nodes, n_nodes, self.n_heads, self.out_per_head)
#         concatenated = torch.cat([Wh_repeated, Wh_repeated_T], dim=3)  # [n_nodes, n_nodes, n_heads, 2*out_per_head]

#         e = self.a(concatenated).squeeze(3)  # [n_nodes, n_nodes, n_heads]
#         e = self.leakyrelu(e)

#         # Mask attention scores based on adjacency
#         adj_mask = adj.unsqueeze(2).expand(n_nodes, n_nodes, self.n_heads)
#         e = e.masked_fill(adj_mask == 0, -1e18)

#         attention = F.softmax(e, dim=1)
#         attention = self.dropout_layer(attention)

#         # Apply attention to Wh
#         h_prime = torch.einsum('ijh,jhf->ihf', attention, Wh)  # [n_nodes, n_heads, out_per_head]

#         if self.concat:
#             h_prime = h_prime.view(n_nodes, self.n_heads * self.out_per_head)
#             h_prime = F.elu(h_prime)
#         else:
#             h_prime = h_prime.mean(dim=1)

#         return h_prime


# class GAT(nn.Module):
#     """GAT model implemented from scratch"""
#     def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout, alpha=0.2, n_heads=8):
#         super(GAT, self).__init__()
#         self.gat1 = GATLayer(n_feat, n_hidden_1, n_heads, dropout, alpha, concat=True)
#         self.dropout = nn.Dropout(dropout)
#         self.gat2 = GATLayer(n_hidden_1, n_hidden_2, 1, dropout, alpha, concat=False)
#         self.fc = nn.Linear(n_hidden_2, n_class)

#     def forward(self, x_in, adj):
#         x = self.gat1(x_in, adj)
#         x = self.dropout(x)
#         Z_1 = self.gat2(x, adj)
#         x_out = self.fc(Z_1)
#         return F.log_softmax(x_out, dim=1), Z_1


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import coalesce, spmm  # For sparse matrix operations

class GATLayer(nn.Module):
    """Single GAT layer with sparse adjacency handling"""
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h, adj):
        # adj should be a tuple of (edge_index, edge_weight)
        edge_index = adj.to_sparse().coalesce().indices()
        adj = (edge_index, torch.ones(edge_index.size(1)))
        edge_index = adj[0]
        num_nodes = h.size(0)
        
        # Linear transformation
        h = self.W(h)  # [N, out_features]
        
        # Compute attention scores
        row, col = edge_index
        h_cat = torch.cat([h[row], h[col]], dim=1)  # [E, 2*out_features]
        e = self.leakyrelu(self.a(h_cat)).squeeze(1)  # [E]
        
        # Apply softmax normalization
        attention = torch.sparse_coo_tensor(
            edge_index, 
            e,
            (num_nodes, num_nodes)
        ).coalesce()
        
        # Sparse softmax
        attention = self.sparse_softmax(attention)
        
        # Apply dropout
        attention = torch.sparse_coo_tensor(
            attention.indices(),
            self.dropout_layer(attention.values()),
            attention.size()
        )
        
        # Aggregate neighbors
        h_prime = spmm(attention.coalesce().indices(), attention.coalesce().values(), 
                      attention.size(0), attention.size(1),
                      h)
        
        if self.concat:
            return F.elu(h_prime)
        return h_prime

    def sparse_softmax(self, sp_mat):
        """Sparse softmax implementation"""
        values = sp_mat.values()
        indices = sp_mat.indices()
        
        # Group by row and apply softmax
        row = indices[0]
        row_unique, row_counts = torch.unique(row, return_counts=True)
        
        softmax_values = []
        start = 0
        for count in row_counts:
            end = start + count
            softmax = F.softmax(values[start:end], dim=0)
            softmax_values.append(softmax)
            start = end
            
        return torch.sparse_coo_tensor(
            indices,
            torch.cat(softmax_values),
            sp_mat.size()
        ).coalesce()


class GAT(nn.Module):
    """GAT model with sparse adjacency support"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout, alpha=0.2):
        super(GAT, self).__init__()
        self.gat1 = GATLayer(n_feat, n_hidden_1, dropout, alpha, concat=True)
        self.gat2 = GATLayer(n_hidden_1, n_hidden_2, dropout, alpha, concat=False)
        self.fc = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj):
        # adj should be (edge_index, edge_weight)
        x = F.elu(self.gat1(x_in, adj))
        x = self.dropout(x)
        Z_1 = self.gat2(x, adj)
        x_out = self.fc(Z_1)
        return F.log_softmax(x_out, dim=1), Z_1