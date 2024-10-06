import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
from power_crtl import PowerControl
from energy_eff import EnergyEfficiency
from pkl_custom import load_graph_info

class AlphaHistogram3D:
    def __init__(self, input, num_sig=10, bins=9, sig_range=(0.00001, 10)):
        self.adjacency_matrix = load_graph_info(input)
        self.num_sig = num_sig
        self.bins = bins
        self.sig_range = sig_range
        self.data = []
        self.sigs = np.linspace(sig_range[0], sig_range[1], num_sig)

    def compute_histograms(self):
        for sig in self.sigs:
            algorithm = PowerControl(self.adjacency_matrix, sig_init=sig)
            p = algorithm.run()
            self.data.append(p)

    def plot_histograms(self):
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, sig in enumerate(self.sigs):
            hist, bin_edges = np.histogram(self.data[i], bins=self.bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            hist = hist / np.sum(hist)

            ax.bar(bin_centers, hist, zs=sig, zdir='y', alpha=0.8, width=0.02)

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
