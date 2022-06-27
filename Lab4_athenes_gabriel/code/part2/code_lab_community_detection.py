"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
from networkx.linalg.laplacianmatrix import laplacian_matrix
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    degree_sequence = [G.degree(node) for node in G.nodes()]
    ##################
    # your code here #
    A=nx.linalg.graphmatrix.adjacency_matrix(G)
    D=np.diag(degree_sequence)
    L=np.eye(A.shape[0])-np.linalg.inv(D)@A
    eigenvalues, eigenvectors = eigs(L, which='SR',k=k)
    eigenvectors=np.real(eigenvectors)
    kmeans=KMeans(n_clusters=k).fit(eigenvectors)
    clustering={}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    ##################
    
    return clustering



############## Task 7

##################
# your code here #
##################
G=nx.read_edgelist("C:/Users/gabri/OneDrive/Bureau/Documents/MVA/Altegrad/Lab4_athenes_gabriel/code/datasets/CA-HepTh.txt",comments="#",delimiter='\t')
largest_cc=max(nx.connected_components(G),key=len)
subG=G.subgraph(largest_cc)
clustering_of_largest_cc = spectral_clustering(subG,50)
print(clustering_of_largest_cc)

############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    clusters=set(clustering.values())
    m=G.number_of_edges()
    modularity=0
    for cluster in clusters :
        
        nodes_in_cluster = [node for node in G.nodes() if clustering[node]==cluster]
        cluster_graph = G.subgraph(nodes_in_cluster)
        lc=cluster_graph.number_of_edges()
        dc = 0
        for node in nodes_in_cluster:
            dc += G.degree(node)
        modularity += lc/m - (dc/(2*m))**2
    return modularity



############## Task 9

##################
# your code here #
##################
random_clustering = {}
for node in G.nodes():
    random_clustering[node]=randint(0,49)
print("Modularity of random_clustering is : ",modularity(subG,random_clustering))
print("Modularity of the largest component is : ",modularity(subG,clustering_of_largest_cc))
