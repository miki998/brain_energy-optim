import matplotlib.pyplot as plt
from src.pkl_custom import load_graph_info

class ListNodes:
    def __init__(self, brain_data_file, vector_file):
        self.load_data(brain_data_file, vector_file)
        self.length_vector = self.adjacency_matrix.shape[0]
        
    def load_data(self, brain_data_file, vector_file):    
        big_file = load_graph_info(brain_data_file)
        
        self.adjacency_matrix = big_file['adjacency']
        self.node_names = big_file['gm_labels']
        self.values = load_graph_info(vector_file)
        
    def sorted_list(self, n=1):
        connections = self.adjacency_matrix.sum(axis=1)
        name_value_connections_pairs= list(zip(self.node_names, self.values, connections))
        sorted_name_value_connections_pairs = sorted(name_value_connections_pairs, key=lambda pair: pair[1], reverse=True)[:n]
        self.table_data = [[name, f"{value:.2f}", str(connection)] for name, value, connection in sorted_name_value_connections_pairs]
        
    def visualize_list(self, number_nodes):
        if number_nodes > self.length_vector:
            number_nodes = self.length_vector
            
        self.sorted_list(number_nodes)
        
        fig, ax = plt.subplots()

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

        table = ax.table(cellText=self.table_data, colLabels=["Name", "Value", "Connections"], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)

        plt.subplots_adjust(top=0.85)
        plt.show()
        
if __name__ == "__main__":
    brain_data_file = '/Users/oscra/Desktop/Data/brain_dict.pkl'
    vector_file = '/Users/oscra/Desktop/Data/vector1.pkl'
    
    visualizer = ListNodes(brain_data_file, vector_file)
    visualizer.visualize_list(30)