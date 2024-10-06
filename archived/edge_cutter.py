import numpy as np
import brain_2d_rep as cf
import networkx as nx

def delete_random_edges(adjacency_matrix, num_edges=1):
    """deletes the precise amount of edges

    Args:
        adjacency_matrix (np.ndarray): adjacency matrix of graph to cut edges
        num_edges (int, optional): number of edges to delete. Defaults to 1.

    Raises:
        ValueError: no edges to delete
        ValueError: asked to delete too many edges

    Returns:
        np.ndarray: the adjacency matrix with lessed edges
    """
    edges = np.argwhere(adjacency_matrix != 0)

    if edges.size == 0:
        raise ValueError("No edges to delete")

    if num_edges > len(edges):
        raise ValueError("Number of edges to delete exceeds the available edges")

    for _ in range(num_edges):
        random_edge_index = np.random.choice(len(edges))
        edge_to_delete = edges[random_edge_index]

        adjacency_matrix[edge_to_delete[0], edge_to_delete[1]] = 0
        adjacency_matrix[edge_to_delete[1], edge_to_delete[0]] = 0

        # Update the edges list
        edges = np.argwhere(adjacency_matrix != 0)

    return adjacency_matrix


if __name__ == "__main__":
    adjacency_matrix = delete_random_edges(nx.to_numpy_array(nx.complete_graph(6)), 1)


    p = cf.fp_algorithm(adjacency_matrix, alpha=adjacency_matrix.shape[0], rand_p=True)

    cf.print_algorithm(adjacency_matrix, p)