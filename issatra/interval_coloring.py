import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import networkx as nx
from issatra.graph_coloring import color_graph

colors = ["r", "g", "b", "y", "m", "c", "k"]

def get_intervals(N=10, start=0, end=10):
    intervals = np.random.random([N, 2]) * (end - start) + start
    intervals = [list(sorted(ij)) for ij in intervals]
    return intervals


def intersection(c0, c1):
    i0, j0 = c0
    i1, j1 = c1

    i, j = max(i0, i1), min(j0, j1)
    if i < j:
        return (i, j)
    else:
        return ()    


def intervals2graph(intervals):
    G = nx.Graph()

    for u, c0 in enumerate(intervals):
        G.add_node(u)
        for du, c1 in enumerate(intervals[u+1:]):
            v = u + 1 + du
            if intersection(c0, c1):
                G.add_edge(u, v)
            else:
                pass #print("No intersection:", u, v)

    return G


def plot_intervals(intervals, interval2color=None):
    fig, ax = plt.subplots()

    margin = 0.05
    for idx, (i, j) in enumerate(intervals):
        if interval2color is not None:
            col = interval2color[idx]
            col = colors[col]
        else:
            col = "b"
        rect = Rectangle((i, idx), (j - i) - margin, 1 - margin, color=col)
        ax.add_patch(rect)

    ax.set_xlim([np.min(intervals), np.max(intervals)])
    ax.set_ylim([0, len(intervals)])
    plt.draw()
    plt.show()


def main():
    np.random.seed(1)
    intervals = get_intervals(N=10)
    G = intervals2graph(intervals)
    num_colors = 6
    interval2color = color_graph(G, num_colors)
    plot_intervals(intervals, interval2color)

if __name__ == '__main__':
    main()