import numpy as np
import random


def get_2sat_formula():
    out = np.loadtxt("../../instances/zzzntbwlkgogbyddwlkpjifqxnjqef.m2s")
    return out.astype(int)


def solver(formula, C):
    m = len(formula[:,0])                           # number of clauses
    max_lit_1 = np.amax(formula[:,3]) + 1
    max_lit_2 = np.amax(formula[:,3]) + 1
    n = max(max_lit_1, max_lit_2)                   # number of literals

    assignment = guess_truth_assignment(n)
    best_assignment = assignment
    max_sat = 0
    for step in range(C*m**2):
        unsat_clause, num_sat = check_satisfiability(formula, assignment, m)
        print(num_sat)
        if num_sat > max_sat:
            max_sat = num_sat
            best_assignment = np.array(assignment)
        if unsat_clause == -1:
            return m, m
        assignment = random_flip(assignment, unsat_clause, formula)
        # print(assignment)
    return max_sat, m, best_assignment


def check_satisfiability(formula, assignment, num_clauses):
    """Inputs: formula, assignment
     output 1: index of the last unsatisfied clause (-1 if all clauses satisfied)
     output 2: number of satisfied clauses"""
    unsatisfied_clause = -1
    num_satisfied = num_clauses
    for clause_index in range(num_clauses):
        clause = formula[clause_index]
        bool_1 = assignment[abs(clause[1])]
        bool_2 = assignment[abs(clause[3])]
        if clause[0] < 0:
            bool_1 = not bool_1
        if clause[2] < 0:
            bool_2 = not bool_2
        if not (bool_1 or bool_2):
            unsatisfied_clause = clause_index
            num_satisfied -= 1
    return unsatisfied_clause, num_satisfied


def guess_truth_assignment(num_literals):
    """Returns a truth assignment guessed uniformly at random"""
    out = np.zeros(num_literals, dtype=bool)
    for i in range(num_literals):
        if bool(random.getrandbits(1)):
            out[i] = True
    return out


def random_flip(assignment, unsatisfied_clause, formula):
    """Flips one of the two truth values at random for the unsatisfied clause"""
    literal_1 = formula[unsatisfied_clause, 1]
    literal_2 = formula[unsatisfied_clause, 3]
    if bool(random.getrandbits(1)):
        assignment[literal_1] = not assignment[literal_1]
    else:
        assignment[literal_2] = not assignment[literal_2]
    return assignment


if __name__ == "__main__":
    C = 10                                          # multiplicative constant to scale solution probability
    formula = get_2sat_formula()
    max_sat_solution, m, best_assign = solver(formula, C)
    print(max_sat_solution, "out of", m, "clauses are satisfiable. The optimum assignment is", best_assign)
