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
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(runtimes)


def counts_data(n, name):
    if name == "mixsat":
        counts = np.loadtxt("./../Max2SAT/adam_counts_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        counts = np.loadtxt("./../Max2SAT_bnb/adam_counts_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(counts)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=13)

    marker_size = 4

    n_list = [5, 9]
    x_solver = "pysat"
    y_solver = "mixsat"

    # RUNTIMES SCATTER
    for n in n_list:
        x = runtimes_data(n, x_solver)[0]
        y = runtimes_data(n, y_solver)[0]
        min_x = np.min(x)
        min_y = np.min(y)
        max_x = np.max(x)
        max_y = np.max(y)

        fig, ax = plt.subplots()
        plt.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0)
        plt.xlim([min_x, max_x])
        plt.ylim([min_y, max_y])
        plt.xlabel("Average instance runtime for " + x_solver)
        plt.ylabel("Average instance runtime for " + y_solver)
        plt.legend()
        plt.loglog()
        plt.tight_layout()
        plt.show()

    # COUNTS SCATTER
    if "pysat" not in x_solver + y_solver:
        for n in n_list:
            x = counts_data(n, x_solver)[0]
            y = counts_data(n, y_solver)[0]
            min_x = np.min(x)
            min_y = np.min(y)
            max_x = np.max(x)
            max_y = np.max(y)

            fig, ax = plt.subplots()
            plt.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0)
            plt.xlim([min_x, max_x])
            plt.ylim([min_y, max_y])
            plt.xlabel("Instance count for " + x_solver)
            plt.ylabel("Instance count for " + y_solver)
            plt.legend()
            plt.loglog()
            plt.tight_layout()
            plt.show()
