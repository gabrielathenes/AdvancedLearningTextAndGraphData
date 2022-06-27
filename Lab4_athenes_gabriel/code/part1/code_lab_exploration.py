"""
Graph Mining - ALTEGRAD - Dec 2021
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
G=nx.read_edgelist("C:/Users/gabri/OneDrive/Bureau/Documents/MVA/Altegrad/Lab4_athenes_gabriel/code/datasets/CA-HepTh.txt",comments="#",delimiter='\t')
print('Our graph has',G.number_of_nodes(),"nodes")
print('Our graph has',G.number_of_edges(),"edges")

##################



############## Task 2

##################
# your code here #
print('Our graph has',nx.number_connected_components(G), ' connected components')
largest_cc=max(nx.connected_components(G),key=len)
print('The largest connected component contains',len(largest_cc),'nodes')
subG=G.subgraph(largest_cc)
print('Our largest connected component has',subG.number_of_edges(),' edges.')
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
print("The min degree of the nodes of G is ",min(degree_sequence))
print("The max degree of the nodes of G is ",max(degree_sequence))
print("The mean degree of the nodes of G is ",np.mean(degree_sequence))
print("The median degree of the nodes of G is ",np.median(degree_sequence))
##################
# your code here #
##################



############## Task 4
plt.plot(nx.degree_histogram(G))
plt.xlabel('Degree')
plt.ylabel('frequency')
plt.title('Degree histogram')
plt.show()

plt.loglog(nx.degree_histogram(G))
plt.xlabel('Degree')
plt.ylabel('frequency')
plt.title('Degree histogram using log-log axis')
plt.show()
##################
# your code here #
##################




############## Task 5

##################
# your code here #
print('The global culstering coefficient of our network is ',nx.transitivity(G))
##################