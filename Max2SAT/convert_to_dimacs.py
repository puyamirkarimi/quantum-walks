import numpy as np


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return data[:, 0], data[:, 1].astype(int)


def make_file(name, formula, num_literal, num_clause):
    with open(name+'.txt', 'w') as file:
        file.write('p cnf'++'\n')


if __name__ == "__main__":
    instances_num = 1000
    instance_names, instance_n_bits = get_instances()
    instance_name = instance_names[instances_num]
    formula = get_2sat_formula(instance_name)
    n = instance_n_bits[instances_num]                          # number of variables
    m = len(formula[:,0])                                       # number of clauses
    make_file(instance_name, formula, n, m)

