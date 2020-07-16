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


def probs_data(n):
    data = np.loadtxt("./../Max2SAT_quantum/opt_inf_time_probs_n_" + str(n) + ".txt")
    return np.reciprocal(data)


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

    n_list = np.arange(5, 10)
    r1 = np.zeros(len(n_list))
    r2 = np.zeros(len(n_list))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)

    solvers = ['pysat', 'mixsat']
    ax = [ax1, ax2]

    for solver_i, solver in enumerate(solvers):
        for i, n in enumerate(n_list):
            x_raw = mask_data(runtimes_data_unaveraged(n, solver))
            y_raw = probs_data(n)

            limit = 10000

            x = average_data(x_raw)[0]
            y = y_raw
            r1[i] = np.corrcoef(x[:limit], y[:limit])[1, 0]

            r2[i] = pearsonr(x, y)[0]
            print(i)
        ax[solver_i].scatter(n_list, r2, color='slateblue', s=23)
        print(solver, ":", r2)
        ax[solver_i].set_xlabel("$n$")

    ax1.set_ylabel("$r$")
    ax1.set_yticks(np.arange(-0.2, 0.3, 0.1))
    min_y = -0.2
    max_y = 0.2
    ax1.set_xlim([4.7, 11.3])
    ax1.set_xticks([5, 7, 9, 11])
    ax1.set_ylim([min_y, max_y])
    ax2.set_xlim([4.7, 11.3])
    ax2.set_xticks([5, 7, 9, 11])
    ax2.set_ylim([min_y, max_y])

    # plt.tight_layout()
    plt.show()
    # plt.savefig('pearson_r_quantum_classical.png', dpi=200)

