import numpy as np
import random
from typing import Optional

def random_directed_edges_id(all_ids, edge_prob, seed):
    """_summary_

    Args:
        all_ids (_type_): _description_
        edge_prob (_type_): _description_
        seed (_type_): _description_

    Returns:
        _type_: _description_
    """
    if seed is not None:
        random.seed(seed)
    num_edges = int(edge_prob * all_ids.shape[1])
    edge_indices = random.sample(range(all_ids.shape[1]), num_edges)
    return all_ids[:, edge_indices]

def random_graph(
    n_nodes: int, edge_prob=0.5, seed: Optional[int] = None
) -> np.ndarray:
    """Generate a random graph with a given number of nodes

    Parameters
    ----------
    n_nodes : int
        number of nodes
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    seed : Optional[int], optional
        seed for reproducibility, by default None

    Returns
    -------
    np.ndarray
        adjacency matrix of the generated graph
    """

    a_mat = np.zeros((n_nodes, n_nodes), dtype=int)

    all_ids = np.array(np.triu_indices_from(a_mat, k=1))

    edge_ids = random_directed_edges_id(
        all_ids, edge_prob=edge_prob, seed=seed
    )

    edge_ids = np.hstack([edge_ids, np.flip(edge_ids, axis=0)])

    a_mat[tuple(edge_ids)] = 1

    return a_mat


def toy_random(
    n_nodes: int,
    edge_prob=0.5,
    con_prob: Optional[float] = None,
) -> np.ndarray:
    """Generate a random graph with a given number of nodes. The graph will have two
    densily connected communities (density 'edge_prob') with a random number of
    connecting edges.

    Parameters
    ----------
    n_nodes : int
        number of nodes
    edge_prob : float, optional
        probability for an edge to connect two nodes, by default 0.5
    con_prob : Optional[float], optional
        density of connecting edges, by default None

    Returns
    -------
    np.ndarray
        graph adjacency matrix
    """
    half_nodes = int(n_nodes / 2)

    com_1 = random_graph(n_nodes=half_nodes, edge_prob=edge_prob)
    com_2 = random_graph(n_nodes=half_nodes, edge_prob=edge_prob)

    if con_prob is None:
        con_prob = edge_prob / 2
    n_connect = half_nodes**2
    edge_idx = np.random.choice(n_connect, int(n_connect * con_prob), replace=False)


    edge_out = edge_idx.copy()
    edge_in = edge_idx.copy()

    connect_out = np.zeros(n_connect)
    connect_in = np.zeros(n_connect)

    connect_out[edge_out] = 1
    connect_in[edge_in] = 1

    a_mat = np.block(
        [
            [com_1, connect_out.reshape((half_nodes, -1))],
            [connect_in.reshape((half_nodes, -1)).T, com_2],
        ]
    )
    return a_mat