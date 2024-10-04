import matplotlib.pyplot as plt
import networkx as nx
import closedform as cf
import numpy as np
import pickle as pkl

cf.print_graph_templates(nx.cycle_graph(6))

cf.print_pkl('adjacency_atlas.pkl', rand_p=True, rand_sig=True)

cf.print_alex_gen(30, 0.8, 0.1)

with open('adjacency_atlas.pkl', 'rb') as file:
    vector_sample = pkl.load(file)
    
with open('adjacency_atlas.pkl', 'rb') as file:
    adjacency_matrix = pkl.load(file)
    
cf.print_algorithm(adjacency_matrix, vector_sample)

# DO NOT FORGET THIS
plt.show()