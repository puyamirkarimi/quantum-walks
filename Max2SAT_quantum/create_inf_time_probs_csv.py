import numpy as np
import csv


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


if __name__ == "__main__":
    instance_names, instance_n_bits = get_instances()

    n = 5
    n_shifted = n - 5

    probs = np.loadtxt("inf_time_probs_n_" + str(n) + ".txt")
    csv_array = np.empty([10000, 2], dtype="S30")

    for loop, i in enumerate(range(n_shifted*10000, (n_shifted+1)*10000)):
        csv_array[loop, 0] = instance_names[i]
        csv_array[loop, 1] = str(probs[loop])

    filename = "inf_time_probs_n_" + str(n) + ".csv"
    with open(filename, 'wb') as f:
        csv.writer(f).writerows(csv_array)
