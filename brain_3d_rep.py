import networkx as nx
import plotly.graph_objects as go
import pickle as pkl
import numpy as np
from pkl_custom import load_graph_info

class BrainGraphVisualizer:
    def __init__(self, brain_data_file, vector_file):
        self.load_data(brain_data_file, vector_file)
        self.build_graph()
        self.calculate_node_positions()
        self.calculate_node_degrees()

    def load_data(self, brain_data_file, vector_file):    
        big_file = load_graph_info(brain_data_file)
        
        self.adjacency_matrix = big_file['adjacency']
        self.coordinates = big_file['coords']
        self.node_names = big_file['gm_labels']
        self.p = load_graph_info(vector_file)

    def build_graph(self):
        self.num_nodes = len(self.coordinates)
        self.G = nx.Graph()

        for i in range(self.num_nodes):
            self.G.add_node(i, pos=self.coordinates[i])

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adjacency_matrix[i, j] == 1:
                    self.G.add_edge(i, j)

    def calculate_node_positions(self):
        node_positions = nx.get_node_attributes(self.G, 'pos')
        self.x_nodes = [node_positions[i][0] for i in self.G.nodes()]
        self.y_nodes = [node_positions[i][1] for i in self.G.nodes()]
        self.z_nodes = [node_positions[i][2] for i in self.G.nodes()]

    def calculate_node_degrees(self):
        self.node_degrees = dict(self.G.degree())
        degree_values = list(self.node_degrees.values())

        normalized_degrees = [(degree - min(degree_values)) / (max(degree_values) - min(degree_values)) for degree in degree_values]
        self.inverted_degrees = [1 - normalized_degree for normalized_degree in normalized_degrees]

    def create_edge_traces(self):
        edge_traces = []
        for edge in self.G.edges():
            x_values = [self.x_nodes[edge[0]], self.x_nodes[edge[1]], None]
            y_values = [self.y_nodes[edge[0]], self.y_nodes[edge[1]], None]
            z_values = [self.z_nodes[edge[0]], self.z_nodes[edge[1]], None]
            edge_traces.append(go.Scatter3d(
                x=x_values, y=y_values, z=z_values,
                mode='lines',
                line=dict(color='rgba(50, 50, 50, 0.25)', width=1),  # Set to gray with lower opacity
                hoverinfo='none'
            ))
        return edge_traces

    def create_node_trace(self, size_scaling=True):
        node_sizes = None
        if size_scaling:
            min_size = 5
            max_size = 30
            p_normalized = (self.p - np.min(self.p)) / (np.max(self.p) - np.min(self.p))
            node_sizes = min_size + (max_size - min_size) * p_normalized
        else:
            node_sizes = np.full(self.num_nodes, 5)
            
        return go.Scatter3d(
            x=self.x_nodes, y=self.y_nodes, z=self.z_nodes,
            mode='markers+text',
            marker=dict(
                symbol='circle',
                size=node_sizes,
                color=self.inverted_degrees,  # Use inverted degree values for color mapping
                colorscale='Viridis',  # Use the Viridis color scale
                cmin=0,
                cmax=1,
                opacity=0.8,
                colorbar=dict(
                    title='Node Degree',
                    tickvals=np.linspace(0, 1, num=11),
                    ticktext=[str(int(max(self.node_degrees.values()) - x * (max(self.node_degrees.values()) - min(self.node_degrees.values())))) for x in np.linspace(0, 1, num=11)]
                )
            ),
            text=[f'{part:.2f}' for part in self.p],
            textposition='top center',
            textfont=dict(size=10, color='black'),
            hovertext=[f'Node: {name}<br>Degree: {self.node_degrees[i]}' for i, name in enumerate(self.node_names)],
            hoverinfo='text'
        )

    def create_layout(self):
        return go.Layout(
            title='3D Graph Visualization with Degree-based Node Coloring',
            scene=dict(
                xaxis=dict(
                    title='X-axis',
                    showbackground=False, 
                    showgrid=False,       
                    zeroline=False,       
                    showticklabels=False, 
                ),
                yaxis=dict(
                    title='Y-axis',
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                ),
                zaxis=dict(
                    title='Z-axis',
                    showbackground=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                ),
            ),
            showlegend=False
        )

    def visualize_graph(self, size_scaling=True):
        edge_traces = self.create_edge_traces()
        node_trace = self.create_node_trace(size_scaling)
        layout = self.create_layout()
        
        fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
        fig.show()

if __name__ == "__main__":
    brain_data_file = '/Users/oscra/Desktop/Data/brain_dict.pkl'
    vector_file = '/Users/oscra/Desktop/Data/vector1.pkl'
    
    visualizer = BrainGraphVisualizer(brain_data_file, vector_file)
    visualizer.visualize_graph()
