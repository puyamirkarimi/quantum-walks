import subprocess
import time
import numpy as np


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../instances_crosson/" + instance_name + ".m2s")  # path of instance files in adam's format
    return out.astype(int)


if __name__ == '__main__':

    num_satisfiable = 0
    n = 20

    for i in range(137):
        instance_name = str(i)
        sat_formula = get_2sat_formula(instance_name)
        unsatisfied = 0
        for clause in sat_formula:
            if clause[0] == -1 and clause [2] == -1:
                unsatisfied +=1
        if unsatisfied == 0:
            num_satisfiable += 1
        
    print('The number of satisfiable Crosson instances for is {}.'.format(num_satisfiable))
        