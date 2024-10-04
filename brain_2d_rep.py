import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import alex_graph as ag
import plotly.graph_objects as go
    
def print_algorithm(adjacency_matrix, p):
    """Private function to draw the algorithm in pyplot
    Args:
        adj_m (np.ndarray): binary adjacency matrix representing the graph (h)
        p (np.ndarray): power levels of the nodes of the graph
    """
    G = nx.from_numpy_array(adjacency_matrix)
    
    pos = nx.spring_layout(G, seed=42)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x = []
    node_y = []
    annotations = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        annotations.append(dict(x=x, y=y, text=f"{p[node]:.2f}", showarrow=False, font=dict(size=10), xanchor='left', yanchor='middle'))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right')
        ))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'# of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Network graph',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=annotations,
                    xaxis=dict(showgrid=False, zeroline=False),
                    yaxis=dict(showgrid=False, zeroline=False))
                    )

    fig.show()
    
