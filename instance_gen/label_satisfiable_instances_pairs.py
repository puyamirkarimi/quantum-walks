import subprocess
import time
import numpy as np


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../instances_pairs_adam_format_5/" + instance_name + ".txt")  # path of instance files in adam's format
    return out.astype(int)


if __name__ == '__main__':
    n = 5
    num_instances = 10000
    out = np.zeros((num_instances, 2), dtype=(np.str_, 25))

    for i in range(num_instances):
        instance_name = str(n) + "_" + str(i)
        instance_name_transformed = str(n) + "_" + str(i) + "_transformed"
        out[i, 0] = instance_name
        sat_formula = get_2sat_formula(instance_name_transformed)
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

    np.savetxt('m2s_pairs_satisfiable.csv', out, delimiter=',', fmt='%s', header='id,satisfiable')
