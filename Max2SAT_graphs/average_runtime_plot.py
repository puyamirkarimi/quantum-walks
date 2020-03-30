import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize


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
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$T_{classical}$ ($s$)")
    ax.set_xlim([5, 20])
    ax.set_xticks(range(5, 21, 3))
    return fig, ax


def plot_graph(x, y, fit, label):
    # if label == "MIXSAT runtimes":
    #     plt.scatter(x[:4], y[:4], color='gray')
    #     plt.scatter(x[4:], y[4:], label=label)
    # else:
    #     plt.scatter(x, y, label=label)

    # plt.scatter(x, y, label=label)
    color = "gray"
    if label == "PySAT runtimes":
        color = "blue"
    elif label == "MIXSAT runtimes":
        color = "forestgreen"
    elif label == "MIXSAT runtimes":
        color = "forestgreen"
    elif label == "Quantum":
        color = "red"
    elif label == "Guess":
        color = "black"
    plt.scatter(x, y, color=color)
    if fit is not None:
        plt.plot(x, fit, '--', color=color)


def after_plot(fig, ax):
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([0.00001, 0.02])
    ax.set_yscale('log')
    # plt.legend(loc="upper right")
    return 0.02 / 0.00001


def after_plot2(fig, ax, scale):
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([4, scale*4])
    ax.set_yscale('log')
    # plt.legend(loc="upper right")


def fit_and_plot(runtimes, label):
    n_array = np.array(range(5, len(runtimes)+5))
    if label == "MIXSAT counts":
        plot_graph(n_array, runtimes, None, label)
        return
    # m_log, c_log = np.polyfit(n_array[0:], np.log2(runtimes[0:]), 1, w=np.sqrt(runtimes[0:]))
    # print(label+":", str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    # exp_fit = np.exp2(m_log * n_array + c_log)
    opt, cov = optimize.curve_fit(lambda x, a, b: a * np.exp2(b * x), n_array, runtimes, p0=(0.0001, 0.087))
    a = opt[0]
    b = opt[1]
    a_error = np.sqrt(cov[0, 0])
    b_error = np.sqrt(cov[1, 1])
    exp_fit = a * np.exp2(b * n_array)
    print(label + ": " + str(a) + " * 2^(" + str(b) + " * n)")
    print("a error:", a_error, "b error:", b_error)
    plot_graph(n_array, runtimes, exp_fit, label)


def fit_and_plot2(x_array, y_array, label):
    # m_log, c_log = np.polyfit(x_array[0:], np.log2(y_array), 1, w=np.sqrt(y_array))
    # exp_fit = np.exp2(m_log * x_array + c_log)
    # print("Quantum:" + str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    opt, cov = optimize.curve_fit(lambda x, a, b: a * np.exp2(b * x), x_array, y_array, p0=(1, 0.5))
    a = opt[0]
    b = opt[1]
    a_error = np.sqrt(cov[0, 0])
    b_error = np.sqrt(cov[1, 1])
    exp_fit = a * np.exp2(b * x_array)
    print(label + ": " + str(a) + " * 2^(" + str(b) + " * n)")
    print("a error:", a_error, "b error:", b_error)
    plot_graph(x_array, y_array, exp_fit, label)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)
    colors = ['forestgreen', 'blue', 'red', 'black']

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

    fig, ax1 = before_plot()
    # fit_and_plot(runtimes_bnb, "B\&B runtimes")
    # fit_and_plot(counts_bnb, "B\&B counts")
    fit_and_plot(runtimes_pysat, "PySAT runtimes")
    fit_and_plot(runtimes_mixsat, "MIXSAT runtimes")
    # fit_and_plot(counts_mixsat, "MIXSAT counts")
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$ 1/ \langle P_{\infty} \rangle $")

    n_array = np.array([5, 6, 7, 8, 9, 10])
    av_probs = np.zeros(len(n_array))
    n_array2 = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    av_probs2 = np.zeros(len(n_array2))
    av_probs3 = np.zeros(len(n_array2))

    for i, n in enumerate(n_array):
        probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
        av_probs[i] = 1/np.mean(probs)

    for i, n in enumerate(n_array2):
        probs2 = np.loadtxt("./../Max2SAT_quantum/zero_time_probs_n_" + str(n) + ".txt")
        av_probs2[i] = 1 / np.mean(probs2)
        av_probs3[i] = np.sqrt(av_probs2[i])

    fit_and_plot2(n_array, av_probs, "Quantum")
    fit_and_plot2(n_array2, av_probs2, "Guess")
    fit_and_plot2(n_array2, av_probs3, "Guess Sqrt")

    scale = after_plot(fig, ax1)
    after_plot2(fig, ax2, scale)
    plt.tight_layout()
    plt.show()
