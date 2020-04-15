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


def runtimes_data(n, name):
    if name.lower() == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name.lower() == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name.lower() == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(runtimes)


def runtimes_data_masked(n, name):
    if name.lower() == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name.lower() == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name.lower() == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(mask_data(runtimes))


def quantum_data(n):
    probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return np.reciprocal(probs)


def counts_data(n, name):
    if name == "mixsat":
        counts = np.loadtxt("./../Max2SAT/adam_counts_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        counts = np.loadtxt("./../Max2SAT_bnb/adam_counts_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(counts)


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
    plt.rcParams["figure.figsize"] = (9.6, 4.8)

    marker_size = 4

    n = 10
    classical_solvers = ["pysat", "mixsat"]
    x_labels = ["RC2", "MIXSAT"]
    colors = ["slateblue", "slateblue"]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax = [ax1, ax2]

    for i, classical_solver in enumerate(classical_solvers):
        # RUNTIMES SCATTER
        y = quantum_data(n)
        x = runtimes_data_masked(n, classical_solver)[0][:len(y)]

        min_x = np.min(x)
        min_y = np.min(y)
        max_x = np.max(x)
        max_y = 100

        ax[i].scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0, color=colors[i])
        ax[i].set_xlim([min_x, max_x])
        ax[i].set_ylim([min_y, max_y])
        if i == 0:
            ax[i].set_ylabel("$1/P_{\infty}$")
        ax[i].set_xlabel(r"$\langle T_{instance} \rangle$~/~$s$~~" + "(" + x_labels[i] + ")")
        ax[i].loglog()
    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)
    plt.tight_layout()
    plt.show()
    # plt.savefig('quantum_classical_scatters.png', dpi=200)

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