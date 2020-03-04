import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1]


if __name__ == '__main__':
    instance_names, instance_n_bits_str = get_instances()
    time_array = np.zeros(16)


