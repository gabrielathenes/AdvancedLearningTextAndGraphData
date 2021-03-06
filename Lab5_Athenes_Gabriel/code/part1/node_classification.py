"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
from matplotlib import pyplot as plt

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)



############## Task 5

##################
# your code here #
##################
nx.draw(G, node_color=y)
plt.show()


############## Task 6
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length,n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
print(idx)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]



############## Task 7

##################
# your code here #
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print ("accuracy score with deepwalk embedding is : ",accuracy_score(y_pred,y_test))
##################




############## Task 8

##################
# your code here #

spectral_embeddings = SpectralEmbedding(n_components=32, affinity='precomputed')
X_spectral = spectral_embeddings.fit_transform(nx.to_numpy_array(G))

print(X_spectral.shape)
X_spectral_train=X_spectral[idx_train,:]
X_spectral_test = X_spectral[idx_test,:]

clf_spectral = LogisticRegression()
clf_spectral.fit(X_spectral_train, y_train)
y_pred_spectral = clf_spectral.predict(X_spectral_test)
print ("accuracy score with spectral embedding is : ",accuracy_score(y_pred_spectral,y_test))
##################