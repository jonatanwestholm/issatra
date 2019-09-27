import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import networkx as nx
from sugarrush.solver import SugarRush
from issatra.graph_coloring import color_graph
from issatra.utils import flatten, optimize

colors = ["r", "g", "b", "y", "m", "c", "k"]

def get_intervals(N=10, start=0, end=10):
    intervals = np.random.random([N, 2]) * (end - start) + start
    intervals = [list(sorted(ij)) for ij in intervals]
    return intervals


def get_discrete_intervals(N=10, start=0, end=10):
    end = end - 1
    intervals = np.random.random([N, 2]) * (end - start) + start
    intervals = [list(sorted(ij)) for ij in intervals]
    intervals = [(int(i), int(j) + 1) for i, j in intervals]
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


def intervals2cliques(intervals):
    cliques = []
    current = []
    endpoints = [[(i, 1, idx), (j, 0, idx)] 
                    for idx, (i, j) in enumerate(intervals)]
    endpoints = sorted(flatten(endpoints))

    for x, is_start, idx in endpoints:
        #print(x, is_start, idx)
        #print(current)
        if is_start:
            current.append(idx)
        else:
            current.remove(idx)
        if len(current) > 1 and tuple(current) not in cliques:
            cliques.append(tuple(current))

    # remove cliques which are subsets of other cliques
    # they produce redundant constraints
    to_be_removed = []
    cliques = [set(clique) for clique in sorted(cliques, key=len)]
    for idx, c0 in enumerate(cliques):
        for c1 in cliques[idx+1:]:
            if c0.issubset(c1):
                to_be_removed.append(c0)
                break

    for clique in to_be_removed:
        cliques.remove(clique)

    #[print(clique) for clique in cliques]
    return cliques


def color_intervals(intervals, num_colors):
    N = len(intervals)

    solver = SugarRush()
    node_col2pick = [[solver.var() for _ in range(num_colors)] for _ in range(N)]

    # every node must pick at least one color
    for col2pick in node_col2pick:
        solver.add(col2pick)

    # for each conflict clique, at most one of each color
    cliques = intervals2cliques(intervals)
    for clique in cliques:
        print(clique)
        node_cols = [node_col2pick[c] for c in clique]
        for node2pick in zip(*node_cols):
            #print("clique:", node2pick)
            atmost_clauses = solver.atmost(node2pick, bound=1, encoding=1)
            #print("atmost:", atmost_clauses)
            #print()
            solver.add(atmost_clauses)

    solver.print_stats()
    t0 = time.time()
    status = solver.solve()
    print("Time: {0:.3f}".format(time.time() - t0))
    print("Satisfiable:", status)

    if not status:
        return None

    # recover solution
    node_col_solved = [solver.solution_values(col2pick) 
                            for col2pick in node_col2pick]
    node2col = [node_col.index(1) for node_col in node_col_solved]
    return node2col


def main():
    np.random.seed(1)
    intervals = get_discrete_intervals(N=100, end=100)
    #print(intervals)
    #intervals2cliques(intervals)
    #num_colors = 60
    if 1:
        print("Direct conflict constraints:")
        G = intervals2graph(intervals)
        interval2color = color_graph(G)
    else:
        print("Clique constraints")
        interval2color = color_intervals(intervals, num_colors)
    #plot_intervals(intervals, interval2color)


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


if __name__ == '__main__':
    main()