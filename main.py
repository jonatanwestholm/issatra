import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import networkx as nx
from sugarrush.solver import SugarRush

colors = ["r", "g", "b", "y", "m", "c", "k"]

def get_graph(num):
    G = nx.Graph()

    if num == 0:
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 0)
        return G, 2

    elif num == 1:
        G.add_edge(0, 1)
        G.add_edge(0, 2)
        G.add_edge(0, 3)
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(2, 3)
        return G, 4
    elif num == 2:
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 0)
        return G, 3


def draw_solution(G, node2col):
    node2col = [colors[col] for col in node2col]
    nx.draw_networkx(G, node_color=node2col)
    plt.show()

def main():
    # problem definition
    G, num_colors = get_graph(2)
    N = len(G) # number of nodes

    # decision variable creation
    solver = SugarRush()
    node_col2pick = [[solver.var() for _ in range(num_colors)] for _ in range(N)]

    # every node must pick at least one color
    for col2pick in node_col2pick:
        solver.add(col2pick)

    # adjacent nodes may not have the same color
    for u, v in G.edges:
        u_col = node_col2pick[u]
        v_col = node_col2pick[v]
        for u_pick, v_pick in zip(u_col, v_col):
            solver.add([-u_pick, -v_pick])

    status = solver.solve()
    print("Satisfiable:", status)

    if not status:
        return

    # recover solution
    node_col_solved = [solver.solution_values(col2pick) 
                            for col2pick in node_col2pick]
    node2col = [node_col.index(1) for node_col in node_col_solved]
    draw_solution(G, node2col)


if __name__ == '__main__':
    main()