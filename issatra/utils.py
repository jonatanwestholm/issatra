from sugarrush.solver import SugarRush

def solve_cnf(cnf):
    with SugarRush() as solver:
        solver.add(cnf)
        solver.print_stats()
        print()
        return solver.solve()

def optimize(get_cnf, lb=0, ub=1, bound_check=True):
    if bound_check:
        if not solve_cnf(get_cnf(ub)):
            return None
        if solve_cnf(get_cnf(lb)):
            return lb
    mid = (ub + lb) // 2
    if mid == lb:
        return ub
    satisfiable = solve_cnf(get_cnf(mid))
    if satisfiable:
        ub = mid
    else:
        lb = mid
    return optimize(get_cnf, lb, ub, bound_check=False)


def flatten(lst):
    return [elem for sub in lst for elem in sub]