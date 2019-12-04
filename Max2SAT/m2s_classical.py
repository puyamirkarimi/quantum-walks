import numpy as np
import random


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return data[:, 0], data[:, 1].astype(int)


def solver(formula, C, n):
    m = len(formula[:,0])                           # number of clauses

    assignment = guess_truth_assignment(n)
    best_assignment = assignment
    max_sat = 0
    for step in range(int(np.ceil(C*m**2))):
        unsat_clauses, num_sat = check_satisfiability(formula, assignment, m)
        print(num_sat)
        if num_sat > max_sat:
            max_sat = num_sat
            best_assignment = np.array(assignment)
        if len(unsat_clauses) == 0:
            return m, m, best_assignment
        unsat_clause = random.choice(unsat_clauses)
        assignment = random_flip(assignment, unsat_clause, formula)
    return max_sat, m, best_assignment


def check_satisfiability(formula, assignment, num_clauses):
    """Inputs: formula, assignment
     output 1: indices of the unsatisfied clauses in a list
     output 2: number of satisfied clauses"""
    unsatisfied_clauses = list()
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
            unsatisfied_clauses.append(clause_index)
            num_satisfied -= 1
    return unsatisfied_clauses, num_satisfied


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
    C = 1                                          # multiplicative constant to scale solution probability
    instance_names, instance_n_bits = get_instances()
    formula = get_2sat_formula(instance_names[-54])
    n = instance_n_bits[-54]          # number of literals
    max_sat_solution, m, best_assign = solver(formula, C, n)
    print(max_sat_solution, "out of", m, "clauses are satisfiable. The optimum assignment is", best_assign)
