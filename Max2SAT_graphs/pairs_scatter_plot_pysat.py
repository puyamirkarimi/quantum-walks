import matplotlib.pyplot as plt
import numpy as np


def bnb_data(n):
    return np.loadtxt('./../Max2SAT_bnb/paired_counts_{}.txt'.format(n))


def bnb_data_transformed(n):
    return np.loadtxt('./../Max2SAT_bnb/paired_transformed_counts_{}.txt'.format(n))


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


def average_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error[x] = np.std(data[:, x], ddof=1) / np.sqrt(num_repeats)

    return y_av, y_std_error


def runtimes_data_unaveraged(n, name):
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/pairs_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/pairs_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/pairs_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return runtimes


def runtimes_data_unaveraged_transformed(n, name):
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/pairs_transformed_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/pairs_transformed_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/pairs_transformed_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return runtimes


def get_satisfiable_list(n):
    data = np.genfromtxt('./../instance_gen/m2s_pairs_satisfiable_{}.csv'.format(n), delimiter=',', skip_header=1, dtype=str)
    satisfiable_data = data[:, 1]
    return satisfiable_data


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=18)
    plt.rcParams["figure.figsize"] = (6, 5)

    marker_size = 4

    n = 10
    fig, ax = plt.subplots()

    x_raw_unmasked = runtimes_data_unaveraged(n, 'pysat')
    y_raw_unmasked = runtimes_data_unaveraged_transformed(n, 'pysat')
    x = average_data(mask_data(x_raw_unmasked))[0]
    y = average_data(mask_data(y_raw_unmasked))[0]

    satisfiable = get_satisfiable_list(n).astype(int)

    x_av = np.mean(x)
    y_av = np.mean(y)
    x_err = np.std(x, ddof=1)/np.sqrt(len(x))
    y_err = np.std(y, ddof=1)/np.sqrt(len(y))
    print('average counts (untransformed): {} +- {}'.format(x_av, x_err))
    print('average counts (transformed): {} +- {}'.format(y_av, y_err))

    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)

    colors = ['green', 'red']
    from matplotlib.colors import ListedColormap

    ax.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0, c=satisfiable, cmap=ListedColormap(colors))
    # ax.set_xlim([8, 150])
    # ax.set_ylim([19, 34000])
    ax.set_xlabel('RC2 runtime (untransformed)')
    ax.set_ylabel('RC2 runtime (transformed)')
    # ax.set_xscale('log', basex=2)
    # ax.set_yscale('log', basey=2)

    plt.tight_layout()
    plt.savefig(f'pysat_instance_pairs_scatter_plot_n_{n}.png', dpi=200)
    plt.show()
    