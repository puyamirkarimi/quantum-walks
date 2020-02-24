import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


if __name__ == '__main__':
    instance_names, instance_n_bits = get_instances()
    # qubits_array = np.array(range(5, 21))                   # Adam's instances range from n=5 to n=20
    time_array = np.zeros(16)

    for n in range(1, 17):
        time_start = time.time()
        for i in range((n-1)*10000, n*10000):                               # 10000 instances per value of n
            instance_name = instance_names[i]
            result = subprocess.run(['./../../mixsat/complete', './../../instances_dimacs/'+instance_name+'.txt'], stdout=subprocess.PIPE)
            # output = result.stdout
            # print(output)
            # failed = b'-' in output
        # print(failed)
        time_end = time.time()
        average_runtime = (time_end - time_start) / 10000
        print("average runtime for n =", n+4, ":", average_runtime)
        time_array[n-1] = average_runtime

    with open("mixsat_runtimes_averaged.txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f, time_array)
