import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return instance_data[:, 0], instance_data[:, 1]


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../instances_original/" + instance_name + ".m2s")  # path of instance files in adam's format
    return out.astype(int)


if __name__ == '__main__':
    instance_names, instance_n_bits_str = get_instances()

    n = 5
    n_shifted = n-5                     # n_shifted runs from 0 to 15 instead of 5 to 20

    out = np.zeros((len(instance_names), 2), dtype=(np.str_, 30))

    for i, instance_name in enumerate(instance_names):
        out[i, 0] = instance_name
        sat_formula = get_2sat_formula(instance_name)
        unsatisfied = 0
        for clause in sat_formula:
            if clause[0] == -1 and clause [2] == -1:
                unsatisfied +=1
        if unsatisfied == 0:
            out[i, 1] = 1
        else:
            out[i, 1] = 0
        if i % 1000 == 0:
            print(i, 'done')

    np.savetxt('m2s_satisfiable.csv', out, delimiter=',', fmt='%s', header='id,satisfiable')
