import numpy as np
from nilearn import plotting
from pkl_custom import open_pkl_file
import matplotlib.pyplot as plt

class ConnectomeVisualizer:
    def __init__(self, brain_data_file, vector_file):
        self.load_data(brain_data_file, vector_file)
        
        self.n = self.adjacency_matrix.shape[0]
        self.zero_mat = np.zeros((self.n,self.n))
        self.node_degree = np.count_nonzero(self.adjacency_matrix, axis=1)
        
        norm = plt.Normalize(vmin=self.node_degree.min(), vmax=self.node_degree.max())
        self.node_colors = plt.cm.viridis_r(norm(self.node_degree))
        
    def load_data(self, brain_data_file, vector_file):    
        big_file = open_pkl_file(brain_data_file)
        
        self.adjacency_matrix = big_file['adjacency']
        self.coordinates = big_file['coords']
        self.p = open_pkl_file(vector_file)
        
    def visualize_graph(self):
        plotting.plot_connectome(self.zero_mat, self.coordinates,
            edge_threshold='100%',  # Threshold to show only the top 10% strongest connections
            edge_vmax=10,
            node_color=self.node_colors,
            node_size=(self.p) * 100,
            title="Lausanne 2018")

        plotting.show()
        
if __name__ == "__main__":
    brain_data_file = '/Users/oscra/Desktop/Data/brain_dict.pkl'
    vector_file = '/Users/oscra/Desktop/Data/vector1.pkl'
    
    visualizer = ConnectomeVisualizer(brain_data_file, vector_file)
    visualizer.visualize_graph()
        
