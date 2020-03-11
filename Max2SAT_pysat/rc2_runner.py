import subprocess
import time
import numpy as np
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1]


if __name__ == '__main__':
    instance_names, instance_n_bits_str = get_instances()
    # qubits_array = np.array(range(5, 21))                   # Adam's instances range from n=5 to n=20
    runtimes = np.zeros(10000)
    states_count = np.zeros(10000)

    n = 20
    n_shifted = n-5                     # n_shifted runs from 0 to 15 instead of 5 to 20

    for loop, i in enumerate(range(n_shifted*10000, (n_shifted+1)*10000)):                   # 10000 instances per value of n
        instance_name = instance_names[i]
        wcnf = WCNF(from_file='./../../instances_wcnf/'+instance_name+'.txt')
        with RC2(wcnf) as rc2:
            rc2.compute()
            runtime = rc2.oracle_time()
        runtimes[loop] = runtime

    with open("adam_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes using time.time()
        f.write(b"\n")
        np.savetxt(f, runtimes)

