from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.ticker as mticker

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


def adams_quantum_walk_data(n):
    return np.reciprocal(np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 2].astype(float))


def adams_adiabatic_data(n):
    a = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', missing_values='', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 10]
    b = []
    skipped = 0
    for i, element in enumerate(a):
        if element != '':
            b.append(float(element))
        else:
            b.append(float('nan'))
            skipped += 1
    print("n:", n, " skipped:", skipped)
    return np.array(b)


def log_tick_formatter(val, pos=None):
    return "{:.0f}".format(10**val)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n = 15

    z = adams_quantum_walk_data(n)
    y = adams_adiabatic_data(n)
    x = bnb_data(n)

    ax.scatter(np.log10(x), np.log10(y), np.log10(z), marker='.', s=3, alpha=0.17)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))


    ax.set_xlabel(r'$N_{calls}$ (Classical)')
    ax.set_ylabel(r'$\langle T_{0.99} \rangle$ (AQC)')
    ax.set_zlabel(r'$1/P_\infty$ (QW)')

    plt.show()