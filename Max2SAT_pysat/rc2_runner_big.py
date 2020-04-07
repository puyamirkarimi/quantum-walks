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

    n = 20 #
    num_instances = 1000

    for i in range(num_instances):
        instance_name = str(n) + "_" + str(i)
        wcnf = WCNF(from_file='./../../instances_big_wcnf/'+instance_name+'.txt')
        with RC2(wcnf) as rc2:
            rc2.compute()
            runtime = rc2.oracle_time()
        runtimes[i] = runtime

    with open("big_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes to .txt file
        f.write(b"\n")
        np.savetxt(f, runtimes)

