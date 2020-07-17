import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


if __name__ == "__main__":
    instance_names, instance_n_bits = get_instances()

    n = 5
    n_shifted = n - 5

    probs = np.loadtxt("inf_time_probs_n_" + str(n) + ".txt")

    indices = np.argsort(probs)[:100]

    with open('hardest_instances_n_'+str(n)+'.txt', 'a') as file:
        for index in indices:
            file.write(instance_names[n_shifted*10000+index]+'\n')