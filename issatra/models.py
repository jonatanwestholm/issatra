import time
import numpy as np

import networkx as nx
from sugarrush.solver import SugarRush
#from garageofcode.mip.solver import get_solver
from magicwrap import get_solver
from issatra.utils import flatten, max_var

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


def color_graph_mip(intervals, num_colors, mutex):
    N = len(intervals)

    if mutex == "pairwise":
        G = intervals2graph(intervals)
        mutex_groups = G.edges
    elif mutex == "cliques":
        mutex_groups = intervals2cliques(intervals)

    if num_colors is None:
        optimize = True
        num_colors = N
    else:
        optimize = False

    solver = get_solver("CBC")
    node_col2pick = [[solver.IntVar(0, 1) for _ in range(num_colors)] for _ in range(N)]

    for col2pick in node_col2pick:
        solver.Add(solver.Sum(col2pick) >= 1)

    print("num mutex groups:", len(mutex_groups))
    for mutex_group in mutex_groups:
        for node2pick in zip(*[node_col2pick[idx] for idx in mutex_group]):
            solver.Add(solver.Sum(node2pick) <= 1)

    if optimize:
        col_cost = np.linspace(0, 1.0, num_colors)
        col2num_picked = [solver.Sum(node2pick) 
                            for node2pick in zip(*node_col2pick)]
        cost = solver.Dot(col2num_picked, col_cost)
        solver.SetObjective(cost, maximize=False)

    status = solver.Solve(time_limit=100)

    if status < 2:
        node_col_solved = [[solver.solution_value(pick) 
                                for pick in col2pick]
                                    for col2pick in node_col2pick]
        node2col = [node_col.index(1) for node_col in node_col_solved]
        print("used colors:", len(set(node2col)))
        return node2col
    else:
        return None


def color_intervals(intervals, num_colors=None, 
                    method="sat", mutex="pairwise"):
    if method == "sat":
        G = intervals2graph(intervals)
        return color_graph_sat(G, num_colors)
    elif method == "mip":
        return color_graph_mip(intervals, num_colors, mutex)


def minimize_spill(intervals, num_registers, optimize=True):
    N = len(intervals)

    solver = get_solver("CBC")
    var_reg2pick = [[solver.var(0, 1) for _ in range(num_registers)] 
                                            for _ in range(N)]

    # at most one register per variable
    var2assigned = solver.sum(var_reg2pick) # is variable assigned to a register?
    solver.add(*[assigned <= 1 for assigned in var2assigned])

    # variables that are live at the same time cannot share a register
    mutexes = intervals2cliques(intervals)
    for mutex in mutexes:
        col2num_picked = solver.sum(zip(*[var_reg2pick[idx] for idx in mutex]))
        solver.add(*[num_picked <= 1 for num_picked in col2num_picked])

    if optimize:
        var_value = [j - i for i, j in intervals] # live range length
        #var_value = [1 for c in intervals]
        value = solver.dot(var2assigned, var_value)
        solver.set_objective(value, maximize=True)

    status = solver.solve(time_limit=100)
    if status < 2:
        var_reg2pick_solved = solver.solution_value(var_reg2pick)
        picked_idx = lambda x: x.index(1) if 1 in x else None
        var2reg = list(map(picked_idx, var_reg2pick_solved))

        v = [reg for reg in var2reg if reg is not None]
        assigned_variables = len(v)
        spilled_variables = len(var2reg) - assigned_variables
        used_registers = len(set(v))
        print("Assigned variables", assigned_variables)
        print("Spilled variables:", spilled_variables)
        print("Used registers:", used_registers)
        return var2reg
    else:
        return None


def schedule_dag(G, optimize=True):
    N = len(G)
    T = int(2*N)

    solver = get_solver("CBC")
    '''
    # the Swedish adjective 'svulstig' comes to mind...

    instruction_ic2pick = solver.var_array([
                            solver.var_array([
                                solver.var(0, 1) for _ in range(T)])
                                                    for _ in range(N)])

    # each instruction gets at least one issue cycle
    solver.add(solver.sum(instruction_ic2pick) >= 1)

    # each issue cycle gets at most one instruction
    solver.add(solver.sum(zip(*instruction_ic2pick)) <= 1)
    '''

    instruction_ic2pick = [[solver.var(0, 1) for _ in range(T)] 
                                                for _ in range(N)]
    
    # each instruction gets exactly one issue cycle
    for ic2pick in instruction_ic2pick:
        solver.add(solver.sum(ic2pick) == 1)

    # each issue cycle gets at most one instruction
    ic2taken = [solver.sum(instruction2pick) 
                    for instruction2pick in zip(*instruction_ic2pick)]
    
    for taken in ic2taken:
        solver.add(taken <= 1)

    # cumulative indicator whether instruction has been issued
    instruction2issued = [[solver.sum(ic2pick[:ic+1]) 
                            for ic in range(T)]
                                for ic2pick in instruction_ic2pick]

    # ensure true dependencies
    for u, v, latency in G.edges(data="latency"):
        #solver.add(instruction2issued[v][latency:] <= instruction2issued[u])
        for u_issued, v_issued in zip(instruction2issued[u], 
                                      instruction2issued[v][latency:]):
            solver.add(v_issued <= u_issued)

        solver.add(instruction2issued[v][latency - 1] == 0)


    if optimize:
        ic2cost = np.linspace(0, 1.0, T)
        cost = solver.dot(ic2taken, ic2cost)
        solver.set_objective(cost, maximize=False)

    status = solver.solve(time_limit=10)

    if status < 2:
        instruction_ic2pick_solved = solver.solution_value(instruction_ic2pick)
        picked_idx = lambda x: x.index(1) if 1 in x else None
        instruction2ic = list(map(picked_idx, instruction_ic2pick_solved))
        print(instruction2ic)
        return instruction2ic
    else:
        return None


'''
# Linprog solution that is actually incorrect
def schedule_dag(G):
    N = len(G)

    solver = get_solver("CBC")
    issue_cycles = [solver.var(lb=0, integer=False) for _ in range(N)]

    for u, v, latency in G.edges(data="latency"):
        ui = issue_cycles[u]        
        vi = issue_cycles[v]
        solver.add(ui + latency <= vi)

    terminals = [u for u in G if not G[u]]
    makespan = solver.max_var(terminals, lb=0)
    solver.set_objective(makespan, maximize=False)

    status = solver.solve(time_limit=10)

    if status:
        return solver.solution_value(issue_cycles)
    else:
        return None
'''