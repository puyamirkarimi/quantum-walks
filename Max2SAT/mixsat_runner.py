import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)       #  can add _noGT_nondg on end
    return instance_data[:, 0], instance_data[:, 1]


if __name__ == '__main__':
    instance_names, instance_n_bits_str = get_instances()
    # qubits_array = np.array(range(5, 21))                   # Adam's instances range from n=5 to n=20
    runtimes = np.zeros(10000)
    states_count = np.zeros(10000)

    n = 20
    n_shifted = n-5                     # n_shifted runs from 0 to 15 instead of 5 to 20

    for loop, i in enumerate(range(n_shifted*10000, (n_shifted+1)*10000)):                               # 10000 instances per value of n
        instance_name = instance_names[i]
        time_start_inst = time.time()
        result = subprocess.run(['./../../mixsat/complete', './../../instances_dimacs/'+instance_name+'.txt'], stdout=subprocess.PIPE)        #  can add _noGT_nondg on end
        time_end_inst = time.time()
        runtime = time_end_inst - time_start_inst
        runtimes[loop] = runtime
        output = str(result.stdout)

        string_start_index = output.find('state_visited ') + 14
        string_end_index = output.find(' pruned')
        states_visited = int(output[string_start_index: string_end_index])
        states_count[loop] = states_visited

    with open("adam_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes using time.time()        #  can add _noGT_nondg in middle
        f.write(b"\n")
        np.savetxt(f, runtimes)

    with open("adam_counts_"+str(n)+".txt", "ab") as f:         # saves counts       #  can add _noGT_nondg in middle
        f.write(b"\n")
        np.savetxt(f, states_count)
