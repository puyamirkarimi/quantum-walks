from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.ticker as mticker
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def quantum_walk_data(n):
    probs = np.loadtxt(Path("./data/inf_time_probs_n_" + str(n) + ".txt"))
    return np.reciprocal(probs)


def adiabatic_data(n):
    if n <= 8:
        times = np.loadtxt(Path("./data/new_adiabatic_time_n_" + str(n) + ".txt"))
    else:
        times = np.genfromtxt(Path('./data/adiabatic_time_n_' + str(n) + '.csv'), delimiter=',', skip_header=1, dtype=str)[:, 1].astype(int)
    return times


def bnb_data(n):
    return np.genfromtxt(Path('./data/mixbnb.csv'), delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 4].astype(int)


def log_tick_formatter(val, pos=None):
    return "{:.0f}".format(10**val)


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n = 11

    z = quantum_walk_data(n)
    y = adiabatic_data(n)
    x = bnb_data(n)

    ax.scatter(np.log10(x), np.log10(y), np.log10(z), marker='.', s=3, alpha=0.17)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    ax.set_xlabel(r'$N_{calls}$ (Classical)')
    ax.set_ylabel(r'$\langle T_{0.99} \rangle$ (AQC)')
    ax.set_zlabel(r'$1/P_\infty$ (QW)')

    plt.show()