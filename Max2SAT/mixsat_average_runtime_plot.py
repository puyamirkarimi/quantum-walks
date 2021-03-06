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


def before_plot():
    return plt.subplots()


def plot_graph(x, y, y_std_error, fit, label):
    plt.scatter(x[:4], y[:4], color='gray')
    plt.scatter(x[4:], y[4:], label=label)
    plt.plot(x, fit, '--')


def after_plot(fig, ax):
    # plt.errorbar(x, y, y_std_error)
    ax.set_xlabel("Number of variables, $n$")
    ax.set_ylabel("Average runtime ($s$)")
    ax.set_xlim([5, 20])
    ax.set_xticks(range(5, 21, 3))
    ax.set_ylim([0.004, 0.014])
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


def fit_and_plot(runtimes, label):
    average, standard_error = average_data(runtimes)
    n_array = np.array(range(5, 21))
    m_log, c_log = np.polyfit(n_array[4:], np.log2(average[4:]), 1, w=np.sqrt(average[4:]))
    print(label+":", str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    exp_fit = np.exp2(m_log * n_array + c_log)
    plot_graph(n_array, average, standard_error, exp_fit, label)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)

    runtimes = np.loadtxt("mixsat_runtimes_averaged.txt").reshape((-1, 16))
    runtimes_noGT = np.loadtxt("mixsat_runtimes_averaged_noGT.txt").reshape((-1, 16))
    runtimes_noGT_nondg = np.loadtxt("mixsat_runtimes_averaged_noGT_nondg.txt").reshape((-1, 16))

    fig, ax = before_plot()
    fit_and_plot(runtimes, "MIXSAT")
    #fit_and_plot(runtimes_noGT, "untransformed, non-degenerate")
    #fit_and_plot(runtimes_noGT_nondg, "untransformed, degenerate")
    after_plot(fig, ax)
