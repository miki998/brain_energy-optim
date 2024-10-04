import numpy as np
import time
import cvxpy as cp
import plotly.graph_objects as go
from pkl_custom import save_pkl_file, load_graph_info
from brain_3d_rep import BrainGraphVisualizer
from brain_2d_rep import print_algorithm
from nilearn_viz import ConnectomeVisualizer
from node_list import ListNodes
import matplotlib.pyplot as plt

class EnergyEfficiency:
    def __init__(self, input, v_init=0.1, rand_v=True, p_on=0.01, sigma=0.01, p_max=1, epsilon=0.00000001):
        self.H = load_graph_info(input)
        self.variable_initializing(v_init, rand_v, p_on, sigma, p_max, epsilon)        
        
    def variable_initializing(self, v_init, rand_v, p_on, sigma, p_max, epsilon):
        self.y = 0
        self.n = self.H.shape[0]
        
        self.p_on = p_on
        self.sigma = sigma
        self.p_max = p_max
        self.epsilon = epsilon
        
        rng = np.random.default_rng(seed=42)
        self.v = rng.random(self.n) if rand_v else np.full(self.n, v_init)
        
        self.z = np.zeros((self.n, self.n))
        self.identity_matrix = sigma * np.eye(self.n)
        
        self.efficiency_history = []
        
        self.node_degrees = np.sum(self.H, axis=1)
        
        self.power_history = np.copy(self.v)
        self.average_power = 0
    
    def sig_part(self, m):
        sum_part = cp.sum([np.outer(self.v[i] * self.H[m, :], self.v[i] * self.H[m, :]) for i in range(self.n) if i != m], axis=0)
        return self.identity_matrix + sum_part
    
    def power_constraint(self, v):
        return cp.sum_squares(v) <= self.p_max
    
    def optimizing_function(self, v):
        log_sum = cp.sum([cp.log1p(2 * cp.sum(cp.multiply(self.z[m, :], (self.H[m, :] @ v))) - cp.quad_form(self.z[m, :], self.sig_part(m))) for m in range(self.n)])
        return -(2 * self.y * cp.sqrt(log_sum) - self.y ** 2 * (cp.sum_squares(v) + self.p_on))
    
    def r_m(self, m):       
        inv_part = np.linalg.solve(self.sig_part(m), self.H[m, :] * self.v[m])
        return np.log(1 + np.dot(self.H[m,:] * self.v[m], inv_part))
    
    def update_z(self):
        for m in range(self.n):                  
            self.z[m, :] = np.linalg.solve(self.sig_part(m), self.H[m, :] * self.v[m])
            
    def update_y(self):
        num = np.sum([self.r_m(m) for m in range(self.n)])
        denom = (self.v ** 2).sum() + self.p_on
        return np.sqrt(num) / denom
    
    def update_v(self):
        v_var = cp.Variable(self.n)
        objective = cp.Minimize(self.optimizing_function(v_var))
        constraints = [self.power_constraint(v_var)]
        problem = cp.Problem(objective, constraints)
        
        problem.solve(solver=cp.SCS, verbose=False, max_iters=1000000)
        
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return v_var.value
        else:
            raise ValueError("Optimization did not converge")
        
    def run(self):
        iterations = 0
        
        init_eff = float('-inf')
        curr_eff = self.optimizing_function(self.v).value
        
        while(abs(init_eff - curr_eff) >= self.epsilon and iterations < 1000):
            update_time = time.time()
            self.update_z()
            self.y = self.update_y()
            updated_time = time.time()
            print(f"Updates: {updated_time-update_time:.3f}")
            self.v = self.update_v()
            v_time = time.time()
            print(f"Optimization: {v_time-updated_time:.3f}")
            init_eff = curr_eff
            curr_eff = self.optimizing_function(self.v).value
            
            self.efficiency_history.append(curr_eff)
            
            iterations += 1
            print(iterations)
        
        print(iterations)
        return self.v
    
    def save_answer(self, file_path):
        save_pkl_file(file_path, self.v)
        
    def plot_brain_2d(self):
        print_algorithm(self.H, self.v)
        
    def plot_brain_3d(self, brain_dictionary_filepath):
        brain_plotter = BrainGraphVisualizer(brain_dictionary_filepath, self.v)
        brain_plotter.visualize_graph()
        
    def plot_brain_3_axis(self, brain_dictionary_filepath):
        brain_plotter = ConnectomeVisualizer(brain_dictionary_filepath, self.v)
        brain_plotter.visualize_graph()
        
    def plot_histogram_power(self):
        fig = go.Figure(data=[go.Histogram(x=self.v)])
        fig.update_layout(title="Histogram of one run", xaxis_title="Value", yaxis_title="Frequency", bargap=0.2)
        fig.show()
        
    def plot_efficienty(self):
        inverted_efficiency = [-value for value in self.efficiency_history]
        fig = go.Figure(data=go.Scatter(x=list(range(len(self.efficiency_history))), y=inverted_efficiency, mode='lines+markers', line=dict(color='blue'), marker=dict(size=6, color='red')))
        fig.update_layout(title='Efficiency Over Iterations', xaxis_title='Iteration', yaxis_title='Efficiency', template='plotly_white')
        fig.show()
        
    def plot_node_v_edges(self):
        scatter = go.Scatter(x= self.node_degrees, y= self.v, mode='markers', marker=dict(color='blue', opacity=0.7, line=dict(color='black', width=1)))
        layout = go.Layout(title='Power vs. Number of Edges', xaxis=dict(title='Number of Edges (Degree)'), yaxis=dict(title='Power'), margin=dict(l=40, r=40, b=40, t=40), showlegend=False)
        fig = go.Figure(data=[scatter], layout=layout)
        fig.show()
        
    def plot_list_nodes(self, brain_data_file, n=1):
        list_plotter = ListNodes(brain_data_file, self.v)
        list_plotter.visualize_list(n)
        
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
        
    def iterate(self, iterations, save_file):
        self.power_history = np.zeros((iterations, self.n))
        
        for i in range(iterations):
            self.power_history[i, :] = algorithm.change_p_init_and_rerun(i)
            
        save_pkl_file(save_file, self.power_history)
        self.average_power = self.power_history.sum(axis=0) / iterations
        
        return self.average_power
        
    def change_p_init_and_rerun(self, seed_p):
        rng_p = np.random.default_rng(seed=seed_p)
        self.v = rng_p.random(self.n)
        return self.run()
    
    
    
if __name__ == "__main__":    
    start_time = time.time()
    algorithm = EnergyEfficiency('/Users/oscra/Desktop/Data/brain_adj.pkl', rand_v=True)
    result = algorithm.run()
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    
    algorithm.iterate(2, 'tester.pkl')
    
    algorithm.plot_matrix_power_history()