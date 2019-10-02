import time
import numpy as np

import networkx as nx
from sugarrush.solver import SugarRush
#from garageofcode.mip.solver import get_solver
from magicwrap import get_solver
from issatra.utils import flatten, max_var

def allocate_registers(intervals, num_registers, optimize=True):
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


def schedule_instructions(G, optimize=True):
    N = len(G)
    T = int(2*N)

    solver = get_solver("CBC")
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