import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr


def average_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error[x] = np.std(data[:, x], ddof=1) / np.sqrt(num_repeats)

    return y_av, y_std_error


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


def runtimes_data_unaveraged(n, name):
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return runtimes


def counts_data(n, name):
    if name == "mixsat":
        counts = np.loadtxt("./../Max2SAT/adam_counts_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        counts = np.loadtxt("./../Max2SAT_bnb/adam_counts_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(counts)


def get_satisfiable_list(n):
    data = np.genfromtxt('./../instance_gen/m2s_satisfiable.csv', delimiter=',', skip_header=1, dtype=str)
    satisfiable_data = data[:, 1]
    m = n - 5
    return satisfiable_data[m*10000:(m+1)*10000]


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (10, 4.8)

    marker_size = 4

    n_list = [5, 20]
    x_solver = "pysat"
    y_solver = "mixsat"
    x_label = "RC2"
    y_label = "MIXSAT"

    # RUNTIMES SCATTER
    fig, (ax1, ax2) = plt.subplots(1, 2)

    for n in n_list:
        x_raw_unmasked = runtimes_data_unaveraged(n, x_solver)
        y_raw_unmasked = runtimes_data_unaveraged(n, y_solver)
        x_raw = mask_data(x_raw_unmasked)
        y_raw = mask_data(y_raw_unmasked)

        limit = 1000
        r = np.corrcoef(np.swapaxes(x_raw, 0, 1)[:limit,:], np.swapaxes(y_raw, 0, 1)[:limit,:])[1, 0]
        print(np.shape(np.swapaxes(x_raw, 0, 1)))
        print("n=", n)
        print("R:", r)

        x = average_data(x_raw)[0]
        y = average_data(y_raw)[0]

        print(pearsonr(x, y))

        if n == 5:
            min_x = np.min(x)
        min_y = np.min(y)
        max_x = np.max(x)
        max_y = np.max(y)

        x_satisfiable = []
        y_satisfiable = []
        x_unsatisfiable = []
        y_unsatisfiable = []

        satisfiable = get_satisfiable_list(n)

        for i in range(len(x)):
            if int(satisfiable[i]) == 1:
                x_satisfiable.append(x[i])
                y_satisfiable.append(y[i])
            elif int(satisfiable[i]) == 0:
                x_unsatisfiable.append(x[i])
                y_unsatisfiable.append(y[i])


        if n == 5:
            ax1.scatter(x_satisfiable, y_satisfiable, label="n={} (satisfiable instances)".format(n), marker='.', s=marker_size, linewidths=0, color='red')
            ax1.scatter(x_unsatisfiable, y_unsatisfiable, label="n={} (unsatisfiable instances)".format(n), marker='.', s=marker_size, linewidths=0, color='green')
            # ax1.set_xlim([min_x, max_x])
            # ax1.set_ylim([min_y, max_y])
            ax1.set_xlabel(r"$\overline{T}_{inst}$~/~$s$~~" + "(for " + x_label + ")")
            ax1.set_ylabel(r"$\overline{T}_{inst}$~/~$s$~~" + "(for " + y_label + ")")
            # ax1.set_yticks([5e-3, 10e-2, 20e-2])
            # plt.legend()
            ax1.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)
            ax1.loglog()
        if n == 20:
            max_y = 0.02
            min_x = 3e-5
            min_y = 4.5e-3
            ax2.scatter(x_satisfiable, y_satisfiable, label="n={} (satisfiable instances)".format(n), marker='.', s=marker_size, linewidths=0, color='red')
            ax2.scatter(x_unsatisfiable, y_unsatisfiable, label="n={} (unsatisfiable instances)".format(n), marker='.', s=marker_size, linewidths=0, color='green')
            # ax1.set_xlim([min_x, max_x])
            # ax1.set_ylim([min_y, max_y])
            ax2.set_xlim([min_x, max_x])
            ax2.set_ylim([min_y, max_y])
            ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)
            ax2.set_xlabel(r"$\overline{T}_{inst}$~/~$s$~~" + "(for " + x_label + ")")
            # ax2.set_ylabel(r"$\langle T_{inst} \rangle$~/~$s$~~" + "(" + y_label + ")")
            # plt.legend()
            ax2.loglog()

    plt.tight_layout()
    plt.show()
    # plt.savefig('classical_runtimes_scatters.png', dpi=200)



    # # COUNTS SCATTER
    # if "pysat" not in x_solver + y_solver:
    #     for n in n_list:
    #         x = counts_data(n, x_solver)[0]
    #         y = counts_data(n, y_solver)[0]
    #         min_x = np.min(x)
    #         min_y = np.min(y)
    #         max_x = np.max(x)
    #         max_y = np.max(y)
    #
    #         fig, ax = plt.subplots()
    #         plt.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0)
    #         plt.xlim([min_x, max_x])
    #         plt.ylim([min_y, max_y])
    #         plt.xlabel("Instance count for " + x_solver)
    #         plt.ylabel("Instance count for " + y_solver)
    #         plt.legend()
    #         plt.loglog()
    #         plt.tight_layout()
    #         plt.show()
