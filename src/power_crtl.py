import time
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.pkl_custom import save_pkl_file, load_graph_info

class PowerControl_updated:
    def __init__(self, input, p_init=0.1, gam_init=0.1, 
                 sig_init=0.01, epsilon=1e-6, nodes_weight=None, 
                 p_max=1, alpha=1, rand_p=False, 
                 rand_sig=False, seed_p=42):

        self.adj_m = load_graph_info(input)
        self.initialization(p_init, gam_init, sig_init, epsilon, p_max, 
                            alpha, rand_p, rand_sig, seed_p, nodes_weight)
        
    def optimize(self, logs=False):
        iterations = 0
        init_eff = float('-inf')
        curr_eff = 0
        
        scores = []
        while(np.abs(init_eff - curr_eff) >= self.epsilon and iterations < 1000000):
            self.y_update()
            self.gam_update()
            self.p_update()

            # Compute Efficiency Score
            init_eff = curr_eff
            curr_eff = self.optimizing_function()
            scores.append(curr_eff)
            iterations += 1
        
        if logs:
            return self.p, scores
        return self.p
    
    def initialization(self, p_init, gam_init, sig_init, epsilon, 
                       p_max, alpha, rand_p, rand_sig, seed_p, nodes_weight):
        
        # Parameter initialization
        self.epsilon = epsilon # Convergence Criteria
        self.p_max = p_max # Maximum Power
        
        self.n = self.adj_m.shape[0] # Number of Nodes

        # y & gam - Intermediate Variable for optimization
        self.y = np.zeros(self.n) # Initialize y
        self.gam = np.full(self.n, gam_init) # Initialize gamma
        if nodes_weight is not None:
            self.weight = nodes_weight
        else:
            self.weight = np.ones(self.n)

        self.alpha = alpha # Extra parameter for self loop
        # Define noise level per node
        rng_sig = np.random.default_rng(seed=42)
        self.sigma = rng_sig.random(self.n) if rand_sig else np.full(self.n, sig_init)

        # Define initial power per node
        rng_p = np.random.default_rng(seed=seed_p)
        self.p = rng_p.random(self.n) if rand_p else np.full(self.n, p_init)
        
        # Allowing for 1-degree self loop since Base Station on Self Client
        np.fill_diagonal(self.adj_m, np.diagonal(np.eye(self.n))  * alpha)
    
    def optimizing_function(self):       
        first_term = (2 * self.y * np.sqrt(self.weight * (1 + self.gam) * self.p)).sum()
        second_term = 0
        
        for i in range(self.n):
            # NOTE: Be careful her the adj_m is written to be outgoing edge
            # in our case since the graph is undirected it is fine but for directed diff
            inner_sum = np.sum(np.abs(self.adj_m[i, :])**2 * self.p)
            second_term += self.y[i]**2 * (inner_sum + self.sigma[i]) 

        return first_term - second_term

    def y_update(self):
        # Missing absolute value for denom
        for station_idx in range(self.n):
            num = np.sqrt(self.weight[station_idx] * (1 + self.gam[station_idx]) * self.adj_m[station_idx,station_idx] ** 2 * self.p[station_idx])
            denom = (self.adj_m[:, station_idx] ** 2 * self.p).sum() + self.sigma[station_idx] ** 2
            self.y[station_idx] = num / denom
            
    def gam_update(self):
        for station_idx in range(self.n):
            num = self.adj_m[station_idx, station_idx] ** 2 * self.p[station_idx]
            denom = (self.adj_m[:, station_idx] ** 2 * self.p).sum() - (self.adj_m[station_idx,station_idx] ** 2 * self.p[station_idx]) + self.sigma[station_idx] ** 2
            self.gam[station_idx] = num / denom
            
    def p_update(self):
        for station_idx in range(self.n):
            num = (self.y[station_idx] ** 2) * self.weight[station_idx] * (1 + self.gam[station_idx]) * self.adj_m[station_idx, station_idx] ** 2

            # AGAIN HERE IT SEEMS THAT THERE MIGHT BE A PROBLEM WHEN USING DIRECTED GRAPH
            denom = ((self.y ** 2) * (self.adj_m[:, station_idx] ** 2)).sum() ** 2
            self.p[station_idx] = min(self.p_max, num / denom)

class PowerControl:
    def __init__(self, input, p_init=0.1, gam_init=0.1, sig_init=0.01, epsilon=0.00000001, p_max=1, alpha=1, rand_p=False, rand_sig=False, seed_p=42):
        self.adj_m = load_graph_info(input)
        self.variable_initializing(p_init, gam_init, sig_init, epsilon, p_max, alpha, rand_p, rand_sig, seed_p)
        
    def run(self):
        iterations = 0
        init_eff = float('-inf')
        curr_eff = 0
        
        while(np.abs(init_eff - curr_eff) >= self.epsilon and iterations < 1000000):
            self.y_update()
            self.gam_update()
            self.p_update()

            # Efficiency Score
            init_eff = curr_eff
            curr_eff = self.optimizing_function()
            
            self.efficiency_history.append(curr_eff)
            
            iterations += 1
            if(iterations % 10000 == 0):
                print(iterations)
        
        return self.p
    
    def variable_initializing(self, p_init, gam_init, sig_init, epsilon, p_max, alpha, rand_p, rand_sig, seed_p):
        self.epsilon = epsilon
        self.p_max = p_max
        self.alpha = alpha
        self.n = self.adj_m.shape[0]
        self.y = np.zeros(self.n)
        self.gam = np.full(self.n, gam_init)

        rng_sig = np.random.default_rng(seed=42)

        # SIGMA DOES NOT NEED TO BE A VECTOR?
        self.sigma = rng_sig.random(self.n) if rand_sig else np.full(self.n, sig_init)
        rng_p = np.random.default_rng(seed=seed_p)
        self.p = rng_p.random(self.n) if rand_p else np.full(self.n, p_init)
        
        self.node_degrees = np.sum(self.adj_m, axis=1)
        
        np.fill_diagonal(self.adj_m, np.diagonal(np.eye(self.n))  * alpha)
        
        self.efficiency_history = []
        
        self.power_history = np.copy(self.p)
        self.average_power = 0
    
    def optimizing_function(self):       
        first_term = (2 * self.y * np.sqrt((1 + self.gam) * self.p)).sum()
        
        second_term = 0
        
        for i in range(self.n):
            # NOTE: Be careful her the adj_m is written to be outgoing edge
            # in our case since the graph is undirected it is fine but for directed diff

            inner_sum = np.sum(self.adj_m[i, :] * self.p) # LIKELY MISTAKE HERE
            # inner_sum = np.sum(np.abs(self.adj_m[i, :])**2 * self.p)
            second_term += self.y[i]**2 * (inner_sum + self.sigma[i]) 

        return first_term - second_term

    
    def y_update(self):
        # Bad practice why do you all of a sudden consider adj[i,i] 
        # Missing absolute value for denom
        for station_idx in range(self.n):
            num = np.sqrt((1 + self.gam[station_idx]) * self.adj_m[station_idx,station_idx] ** 2 * self.p[station_idx])
            denom = (self.adj_m[:, station_idx] ** 2 * self.p).sum() + self.sigma[station_idx] ** 2
            self.y[station_idx] = num / denom
            
    def gam_update(self):
        for station_idx in range(self.n):
            num = self.adj_m[station_idx, station_idx] ** 2 * self.p[station_idx]
            denom = (self.adj_m[:, station_idx] ** 2 * self.p).sum() - (self.adj_m[station_idx,station_idx] ** 2 * self.p[station_idx]) + self.sigma[station_idx] ** 2
            self.gam[station_idx] = num / denom
            
    def p_update(self):
        for station_idx in range(self.n):
            num = (self.y[station_idx] ** 2) * (1 + self.gam[station_idx]) * self.adj_m[station_idx, station_idx] ** 2

            # AGAIN HERE IT SEEMS THAT THERE MIGHT BE A PROBLEM WHEN USING DIRECTED GRAPH
            denom = ((self.y ** 2) * (self.adj_m[:, station_idx] ** 2)).sum() ** 2
            self.p[station_idx] = min(self.p_max, num / denom)
        
    def plot_node_v_edges(self):
        scatter = go.Scatter(x= self.node_degrees, y= self.p, mode='markers', marker=dict(color='blue', opacity=0.7, line=dict(color='black', width=1)))
        layout = go.Layout(title='Power vs. Number of Edges', xaxis=dict(title='Number of Edges (Degree)'), yaxis=dict(title='Power'), margin=dict(l=40, r=40, b=40, t=40), showlegend=False)
        fig = go.Figure(data=[scatter], layout=layout)
        fig.show()
        
    def plot_matrix_power_history(self):
        plt.figure(figsize=(12, 8))
        plt.imshow(self.power_history.T, cmap='gray_r', aspect='auto')

        plt.xlabel('Iteration Number')
        plt.ylabel('Node')

        plt.xticks(ticks=np.arange(0, self.power_history.shape[0], 5), labels=np.arange(1, self.power_history.shape[0] + 1, 5))
        plt.yticks(ticks=np.arange(0, self.power_history.shape[1], 5), labels=np.arange(1, self.power_history.shape[1] + 1, 5))

        plt.title('Matrix of node value through iterations')
        plt.colorbar(label='Value')
        plt.show()
        

    # def count_unique_edges_of_first_neighbors(self, adjacency_matrix):
    #     sum_edges = np.zeros(self.n)
        
    #     for node_index in range(self.n):
    #         first_neighbors = np.nonzero(adjacency_matrix[node_index])[0]
            
    #         unique_edges = set()

    #         for neighbor in first_neighbors:
    #             for i in range(self.n):
    #                 if adjacency_matrix[neighbor, i] == 1 and i != node_index:
    #                     edge = tuple(sorted((neighbor, i)))
    #                     unique_edges.add(edge)
                        
    #         sum_edges[node_index] = len(unique_edges)
        
    #     return sum_edges


    # def save_answer(self, file_path):
    #     save_pkl_file(file_path, self.p)
        
    # def plot_histogram(self):
    #     fig = go.Figure(data=[go.Histogram(x=self.p)])
    #     fig.update_layout(title="Histogram of one run", xaxis_title="Value", yaxis_title="Frequency", bargap=0.2)
    #     fig.show()
    
    # def plot_efficiency(self):
    #     fig = go.Figure(data=go.Scatter(x=list(range(len(self.efficiency_history))), y=self.efficiency_history, mode='lines+markers', line=dict(color='blue'), marker=dict(size=6, color='red')))
    #     fig.update_layout(title='Efficiency Over Iterations', xaxis_title='Iteration', yaxis_title='Efficiency', template='plotly_white')
    #     fig.show()

    # def iterate(self, iterations, save_file):
    #     self.power_history = np.zeros((iterations, self.n))
        
    #     for i in range(iterations):
    #         self.power_history[i, :] = algorithm.change_p_init_and_rerun(i)
            
    #     save_pkl_file(save_file, self.power_history)
    #     self.average_power = self.power_history.sum(axis=0) / iterations
        
    #     return self.average_power
        
    # def change_p_init_and_rerun(self, seed_p):
    #     rng_p = np.random.default_rng(seed=seed_p)
    #     self.p = rng_p.random(self.n)
    #     rng_gam = np.random.default_rng(seed=seed_p+5)
    #     self.gam = rng_gam.random(self.n)
    #     return self.run()
        
    
if __name__ == "__main__":
    # start_time = time.time()
    algorithm = PowerControl('/Users/oscra/Desktop/Data/brain_adj.pkl', rand_p=True)
    result = algorithm.run()

    # end_time = time.time()

    # elapsed_time = end_time - start_time
    # print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    # algorithm.plot_brain_3d('/Users/oscra/Desktop/Data/brain_dict.pkl')
    # algorithm.plot_efficiency()