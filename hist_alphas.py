import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
from power_crtl import PowerControl
from pkl_custom import load_graph_info

class AlphaHistogram3D:
    def __init__(self, input, num_alphas=5, bins=9, alpha_range=(0.01, 0.025)):
        self.adjacency_matrix = load_graph_info(input)
        self.num_alphas = num_alphas
        self.bins = bins
        self.alpha_range = alpha_range
        self.data = []
        self.alphas = np.linspace(alpha_range[0], alpha_range[1], num_alphas)

    def compute_histograms(self):
        for alpha in self.alphas:
            algorithm = PowerControl(self.adjacency_matrix, alpha=alpha)
            p = algorithm.run()
            self.data.append(p)

    def plot_histograms(self):
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, alpha in enumerate(self.alphas):
            hist, bin_edges = np.histogram(self.data[i], bins=self.bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            hist = hist / np.sum(hist)

            ax.bar(bin_centers, hist, zs=alpha, zdir='y', alpha=0.8, width=0.02)

        ax.set_xlabel('Amplitude')
        ax.set_ylabel(r'$\alpha$')
        ax.set_zlabel('Frequency')
        ax.set_title('3D Histogram Distribution as Alpha Changes')

        plt.show()

    def run(self):
        self.compute_histograms()
        self.plot_histograms()

if __name__ == "__main__":
    alpha_histogram = AlphaHistogram3D('/Users/oscra/Desktop/Data/brain_adj.pkl')
    alpha_histogram.run()
