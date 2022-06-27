"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    ############## Task 1
    
    ##################
    # your code here #
    n = A.shape[0]
    A = A + sp.identity(n)
    degs = A.dot(np.ones(n))
    inv_degs=np.power(degs, -1)
    D_inv = sp.diags(inv_degs)
    A_normalized = D_inv.dot(A)
    return A_normalized
    ##################


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()
    sigmoid = nn.Sigmoid()
    ############## Task 3
    
    ##################
    # your code here #
    y=list()
    y_pred=list()

    indices = adj._indices()
    m=indices.size(1)
    y.append(torch.ones(m).to(device))
    y_pred.append(sigmoid(torch.sum(torch.mul(z[indices[0],:], z[indices[1],:]),dim=1)))
    rand_indices = torch.randint(0, z.size(0),indices.size())
    y.append(torch.zeros(m).to(device))
    y_pred.append(sigmoid(torch.sum(torch.mul(z[rand_indices[0],:], z[rand_indices[1],:]),dim=1)))
    ##################
    y=torch.cat(y, dim=0)
    y_pred = torch.cat(y_pred,dim=0)

    loss = mse_loss(y_pred, y)
    return loss
