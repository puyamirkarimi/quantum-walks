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
    return average_data(mask_data(runtimes))


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
    plt.rc('font', size=14)

    marker_size = 4

    n_list = np.arange(5, 21)
    r1 = np.zeros(len(n_list))
    r2 = np.zeros(len(n_list))
    x_solver = "pysat"
    y_solver = "mixsat"

    # RUNTIMES SCATTER
    for i, n in enumerate(n_list):
        x_raw = mask_data(runtimes_data_unaveraged(n, x_solver))
        y_raw = mask_data(runtimes_data_unaveraged(n, y_solver))

        limit = 1000
        r1[i] = np.corrcoef(np.swapaxes(x_raw, 0, 1)[:limit,:], np.swapaxes(y_raw, 0, 1)[:limit,:])[1, 0]

        x = average_data(x_raw)[0]
        y = average_data(y_raw)[0]

        r2[i] = pearsonr(x, y)[0]
        print(i)

    fig, ax = plt.subplots()
    # plt.scatter(n_list, r1, label="r1")
    plt.scatter(n_list, r2, s=18, color='slateblue')
    plt.xlabel("$n$")
    plt.xlim([4.7, 20.3])
    plt.xticks([5, 10, 15, 20])
    plt.ylim([0.07, 0.5])
    plt.yticks(np.arange(0.1, 0.6, 0.1))
    plt.ylabel("$r$")
    plt.tick_params(direction='in', top=True, right=True)
    # plt.tight_layout()
    # plt.show()
    plt.savefig('pearson_r.png', dpi=200)
