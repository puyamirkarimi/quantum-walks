import matplotlib.pyplot as plt
import numpy as np


def average_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error[x] = np.std(data[:, x], ddof=1) / np.sqrt(num_repeats)

    return y_av, y_std_error


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


def quantum_walk_data(n):
    probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return np.reciprocal(probs)


def adiabatic_data(n):
    if n <= 8:
        times = np.loadtxt("./../Max2SAT_quantum/adiabatic/new_adiabatic_time_n_" + str(n) + ".txt")
    else:
        times = np.genfromtxt('./../Max2SAT_quantum/adiabatic/adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)[:, 1].astype(int)
    return times


def mask_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    out = np.zeros((num_repeats-2, num_x_vals))
    for x in range(num_x_vals):
        vals = data[:, x]
        vals1 = np.delete(vals, vals.argmin())
        vals2 = np.delete(vals1, vals1.argmax())
        out[:, x] = vals2
    return out


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (6, 6)

    marker_size = 4

    n = 11
    fig, ax = plt.subplots()

    x = quantum_walk_data(n)
    y = adiabatic_data(n)

    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = 100

    ax.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0)
    ax.set_xlim([8, 150])
    ax.set_ylim([19, 34000])
    ax.set_xlabel("$1/P_{\infty}$")
    ax.set_ylabel(r"$\langle T_{0.99} \rangle$")
    ax.loglog()

    # plt.tight_layout()
    # plt.show()
    plt.savefig('n_'+str(n)+'_adiabatic_vs_QW.png', dpi=200)

    # # COUNTS SCATTER
    # if classical_solver.lower() != "pysat":
    #     for n in n_list:
    #         y = quantum_data(n)
    #         x = counts_data(n, classical_solver)[0][:len(y)]
    #
    #         min_x = np.min(x)
    #         min_y = np.min(y)
    #         max_x = np.max(x)
    #         max_y = np.max(y)
    #
    #         fig, ax = plt.subplots()
    #         plt.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0)
    #         plt.xlim([min_x, max_x])
    #         plt.ylim([min_y, max_y])
    #         plt.xlabel("Average instance runtime for " + classical_solver)
    #         plt.ylabel("Inverse infinite time probability")
    #         plt.legend()
    #         plt.loglog()
    #         plt.tight_layout()
    #         plt.show()