import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits_noGT_nondg.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1]


def func():
    y = 0
    i = 0
    for i in range(5000):
        y += i
        y -= 10
        y *= 2


if __name__ == '__main__':
    for i in range(10):
        start = time.time()
        func()
        end = time.time()
        print("time.time()", end-start)
    for i in range(10):
        start = time.process_time()
        func()
        end = time.process_time()
        print("time.process_time()", end-start)
