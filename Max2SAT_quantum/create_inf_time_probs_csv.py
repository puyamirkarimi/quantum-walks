import numpy as np
import csv


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


if __name__ == "__main__":
    instance_names, instance_n_bits = get_instances()

    n = 11
    n_shifted = n - 5

    probs = np.loadtxt("inf_time_probs_n_" + str(n) + ".txt")
    names = list()

    for loop, i in enumerate(range(n_shifted*10000, (n_shifted+1)*10000)):
        names.append(str(instance_names[i]))

    filename = "inf_time_probs_n_" + str(n) + ".csv"
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        f.write('id,P_infty\n')
        for row in zip(names, probs):
            writer.writerow(row)