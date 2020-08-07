from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


def quantum_walk_data(n):
    probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return np.reciprocal(probs)


def adiabatic_data(n):
    if n <= 8:
        times = np.loadtxt("./../Max2SAT_quantum/adiabatic/new_adiabatic_time_n_" + str(n) + ".txt")
    else:
        times = np.genfromtxt('./../Max2SAT_quantum/adiabatic/adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)[:, 1].astype(int)
    return times


def bnb_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/bnb/mixbnb.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 4].astype(int)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    n = 11

    z = quantum_walk_data(n)
    y = adiabatic_data(n)
    x = bnb_data(n)

    ax.scatter(x, y, z, marker='.', s=4)

    ax.set_xlabel('bnb')
    ax.set_ylabel('adiabatic')
    ax.set_zlabel('qw')

    plt.show()