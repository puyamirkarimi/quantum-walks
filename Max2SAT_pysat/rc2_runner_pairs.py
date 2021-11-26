import subprocess
import time
import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return instance_data[:, 0], instance_data[:, 1]


if __name__ == '__main__':
    runtimes = np.zeros(10000)
    runtimes_transformed = np.zeros(10000)

    n = 5

    num_instances = 10000

    for i in range(num_instances):
        instance_name = str(n) + "_" + str(i)
        instance_name_transformed = str(n) + "_" + str(i) + "_transformed"

        wcnf = WCNF(from_file='./../../instances_pairs_wcnf_5/'+instance_name+'.txt')
        with RC2(wcnf) as rc2:
            rc2.compute()
            runtime = rc2.oracle_time()
        runtimes[i] = runtime

        wcnf = WCNF(from_file='./../../instances_pairs_wcnf_5/'+instance_name_transformed+'.txt')
        with RC2(wcnf) as rc2:
            rc2.compute()
            runtime = rc2.oracle_time()
        runtimes_transformed[i] = runtime

    with open("pairs_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes to .txt file
        f.write(b"\n")
        np.savetxt(f, runtimes)
    
    with open("pairs_transformed_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes to .txt file
        f.write(b"\n")
        np.savetxt(f, runtimes_transformed)

