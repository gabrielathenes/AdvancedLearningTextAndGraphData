"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GAE(nn.Module):
    """GAE model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, dropout):
        super(GAE, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Task 2
    
        ##################
        # your code here #
        x = self.fc1(x_in)
        z0 = self.relu(torch.mm(adj, x))
        z0=self.dropout(z0)
        z1 = self.fc2(z0)
        z1 = torch.mm(adj, z1)
        ##################

        return z1
