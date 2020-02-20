import numpy as np


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances_crosson/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return data[:, 0], data[:, 1].astype(int)


def make_file(name, formula, num_var, num_cls):
    with open("../../instances_crosson_dimacs/"+name+'.txt', 'w') as file:
        file.write('p cnf '+str(num_var)+' '+str(num_cls)+'\n')
        for clause in range(num_cls):
            sign_1 = formula[clause, 0]
            sign_2 = formula[clause, 2]
            v_1 = formula[clause, 1] + 1
            v_2 = formula[clause, 3] + 1
            file.write(str(int(sign_1*v_1))+" "+str(int(sign_2*v_2))+" 0\n")


if __name__ == "__main__":
    instance_names = np.array(range(137))
    for instance_name_int in instance_names:
        instance_name = str(instance_name_int)
        sat_formula = get_2sat_formula(instance_name)
        n = 20
        m = len(sat_formula[:, 0])  # number of clauses

        make_file(instance_name, sat_formula, n, m)