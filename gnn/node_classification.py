import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from model import GNN, GAT
from utils import load_dataset, sparse_to_torch_sparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 


model, optimizer, features, adj, idx_train, idx_test = None, None, None, None, None, None

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)[0]
    loss_train = F.nll_loss(output[idx_train], y[idx_train])
    acc_train = accuracy_score(torch.argmax(output[idx_train], dim=1).detach().cpu().numpy(), y[idx_train].cpu().numpy())
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output, embeddings = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], y[idx_test])
    print("output :", torch.argmax(output[idx_test], dim=1).detach().cpu().numpy())
    print("prediction :", y[idx_test])
    acc_test = accuracy_score(torch.argmax(output[idx_test], dim=1).detach().cpu().numpy(), y[idx_test].cpu().numpy())
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test))
    
    return embeddings[idx_test]
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="cora", choices=["karate", "cora", "amazon", "citeseer"], help="Dataset Name")

    args = parser.parse_args()
    dataset_name = args.dataset_name

    features, adj, class_labels, y = load_dataset(dataset_name)
    n = adj.shape[0]
    n_class = np.unique(class_labels).size # Number of classes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Yields indices to split data into training, validation and test sets
    idx = np.random.permutation(n)
    idx_train = idx[:int(0.6*n)]
    idx_val = idx[int(0.6*n):int(0.8*n)]
    idx_test = idx[int(0.8*n):]

    # Transform the numpy matrices/vectors to torch tensors
    features = torch.FloatTensor(features).to(device)
    adj = sparse_to_torch_sparse(adj).to(device)
    idx_train = torch.tensor(idx_train, device=device).long()
    idx_val = torch.tensor(idx_val, device=device).long()
    idx_test = torch.tensor(idx_test, device=device).long() 
    y = y.to(device)   

    # Hyperparameters
    epochs = 100
    n_hidden_1 = 128
    n_hidden_2 = 64
    learning_rate = 0.01
    dropout_rate = 0.1

    # Creates the model and specifies the optimizer
    # model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate).to(device)
    model = GAT(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate).to(device)
    # def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout, alpha=0.2, n_heads=8):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print()

    # Testing
    embeddings_test = test()
    embeddings_test = embeddings_test.cpu().detach().numpy()

    # Projects the emerging representations to two dimensions using t-SNE
    if dataset_name == "karate":
        perplexity_value = min(30, len(embeddings_test) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity_value)
    else:
        tsne = TSNE(n_components=2)
    embeddings_test_2d = tsne.fit_transform(embeddings_test)

    labels = torch.tensor(y, device=device).long()[idx_test].cpu()
    unique_labels = np.unique(labels)

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'lime', 'brown', 'pink', 'navy', 'teal', 'coral', 'gold', 'darkgreen', 'maroon', 'olive', 'skyblue']
    fig, ax = plt.subplots()
    for i in range(unique_labels.size):
        idxs = [j for j in range((labels.size()[0])) if labels[j]==unique_labels[i]]
        ax.scatter(embeddings_test_2d[idxs,0], 
                embeddings_test_2d[idxs,1], 
                c=colors[i],
                label=i,
                alpha=0.7,
                s=10)

    ax.legend(scatterpoints=1)
    fig.suptitle('T-SNE Visualization of the nodes of the test set',fontsize=12)
    plt.show()
