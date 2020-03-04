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


def before_plot():
    return plt.subplots()


def plot_graph(x, y, fit, label):
    if label == "MIXSAT runtimes":
        plt.scatter(x[:4], y[:4], color='gray')
        plt.scatter(x[4:], y[4:], label=label)
    else:
        plt.scatter(x, y, label=label)
    if fit is not None:
        plt.plot(x, fit, '--')


def after_plot(fig, ax):
    # plt.errorbar(x, y, y_std_error)
    ax.set_xlabel("Number of variables, $n$")
    ax.set_ylabel("Average runtime ($s$)")
    ax.set_xlim([5, 20])
    ax.set_xticks(range(5, 21, 3))
    #ax.set_ylim([0.004, 0.014])
    ax.set_yscale('log')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def fit_and_plot(runtimes, label):
    n_array = np.array(range(5, len(runtimes)+5))
    if label == "MIXSAT counts":
        plot_graph(n_array, runtimes, None, label)
        return
    elif "MIXSAT" in label:
        m_log, c_log = np.polyfit(n_array[4:], np.log2(runtimes[4:]), 1, w=np.sqrt(runtimes[4:]))
    else:
        m_log, c_log = np.polyfit(n_array[0:], np.log2(runtimes[0:]), 1, w=np.sqrt(runtimes[0:]))
    print(label+":", str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    exp_fit = np.exp2(m_log * n_array + c_log)
    plot_graph(n_array, runtimes, exp_fit, label)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=12)

    max_n_bnb = 10
    max_n_others = 20

    runtimes_bnb = np.zeros(max_n_bnb - 4)
    counts_bnb = np.zeros(max_n_bnb - 4)
    runtimes_mixsat = np.zeros(max_n_others - 4)
    counts_mixsat = np.zeros(max_n_others - 4)
    runtimes_pysat = np.zeros(max_n_others - 4)

    for i, n in enumerate(range(5, max_n_bnb+1)):
        runtimes_bnb[i] = np.mean(runtimes_data(n, "branch and bound")[0].flatten())
        counts_bnb[i] = np.mean(counts_data(n, "branch and bound")[0].flatten())

    for i, n in enumerate(range(5, max_n_others+1)):
        runtimes_mixsat[i] = np.mean(runtimes_data(n, "mixsat")[0].flatten())
        counts_mixsat[i] = np.mean(counts_data(n, "mixsat")[0].flatten())
        runtimes_pysat[i] = np.mean(runtimes_data(n, "pysat")[0].flatten())

    counts_bnb *= runtimes_bnb[0]/counts_bnb[0]     # normalisation
    counts_mixsat *= runtimes_mixsat[0] / counts_mixsat[0]  # normalisation

    fig, ax = before_plot()
    fit_and_plot(runtimes_bnb, "B\&B runtimes")
    fit_and_plot(counts_bnb, "B\&B counts")
    fit_and_plot(runtimes_pysat, "PySAT runtimes")
    fit_and_plot(runtimes_mixsat, "MIXSAT runtimes")
    fit_and_plot(counts_mixsat, "MIXSAT counts")
    after_plot(fig, ax)
