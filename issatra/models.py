import time

import networkx as nx
from sugarrush.solver import SugarRush
from garageofcode.mip.solver import get_solver

def color_graph_sat(G, num_colors=None):
    N = len(G) # number of nodes
    if num_colors is None:
        optimize = True
        num_colors = N
    else:
        optimize = False

    # decision variable creation
    solver = SugarRush()
    node_col2pick = [[solver.var() for _ in range(num_colors)] for _ in range(N)]

    # every node must pick at least one color
    for col2pick in node_col2pick:
        solver.add(col2pick)

    # adjacent nodes may not have the same color
    for u, v in G.edges:
        #print("constraining edge:", u, v)
        u_col = node_col2pick[u]
        v_col = node_col2pick[v]
        for u_pick, v_pick in zip(u_col, v_col):
            solver.add([-u_pick, -v_pick])

    if optimize:
        use_cols = []
        for node2pick in zip(*node_col2pick):
            use_col, equivalence_clauses = solver.indicate_disjunction(list(node2pick))
            use_cols.append(use_col)
            solver.add(equivalence_clauses)

        # symmetry breaking - colors must be used smallest first
        for uc0, uc1 in zip(use_cols, use_cols[1:]):
            solver.add([-uc1, uc0])

        clauses, itot = solver.itotalizer(use_cols)
        solver.add(clauses)
        #print(itot)
        t0 = time.time()
        min_colors = solver.optimize(itot)
        print("Optimization time: {0:.3f}".format(time.time() - t0))
        print("min colors:", min_colors)
        assumptions = [-itot[min_colors]]
    else:
        assumptions = []
            
    solver.print_stats()
    t0 = time.time()
    status = solver.solve(assumptions=assumptions)
    print("Time: {0:.3f}".format(time.time() - t0))
    print("Satisfiable:", status)

    if not status:
        return None

    # recover solution
    node_col_solved = [solver.solution_values(col2pick) 
                            for col2pick in node_col2pick]
    node2col = [node_col.index(1) for node_col in node_col_solved]
    return node2col


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


def color_graph_mip(G, num_colors):
    N = len(G)

    solver = get_solver("CBC")
    node_col2pick = [[solver.IntVar(0, 1) for _ in range(num_colors)] for _ in range(N)]

    for col2pick in node_col2pick:
        solver.Add(solver.Sum(col2pick) == 1)

    for u, v in G.edges:
        for u_col, v_col in zip(node_col2pick[u], node_col2pick[v]):
            solver.Add(u_col + v_col <= 1)

    status = solver.Solve(time_limit=10)

    if status < 2:
        node_col_solved = [[solver.solution_value(pick) 
                                for pick in col2pick]
                                    for col2pick in node_col2pick]
        node2col = [node_col.index(1) for node_col in node_col_solved]
        print(node2col)
        return node2col
    else:
        return None


def color_intervals(intervals, num_colors=None, method="sat"):
    G = intervals2graph(intervals)
    
    if method == "sat":
        return color_graph_sat(G, num_colors)
    elif method == "mip":
        return color_graph_mip(G, num_colors)