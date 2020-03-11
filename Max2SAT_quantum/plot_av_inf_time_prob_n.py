import numpy as np
import matplotlib.pyplot as plt


def plot_graph(x, y, fit):
    plt.figure()
    plt.scatter(x, y)
    if fit is not None:
        plt.plot(x, fit, '--')
    plt.tick_params(direction='in', top=True, right=True)
    plt.xlim(5, 9)
    plt.ylim(2, 16)
    plt.xticks(range(5, 10, 1))
    # plt.yticks(range(start, end+1, step))
    plt.xlabel("$n$")
    plt.ylabel(r"$\langle P_{\infty} \rangle ^{-1}$")
    plt.yscale('log', basey=2)
    plt.show()


def fit_and_plot(x_array, y_array):
    m_log, c_log = np.polyfit(x_array[0:], np.log2(y_array), 1, w=np.sqrt(y_array))
    exp_fit = np.exp2(m_log * x_array + c_log)
    print(str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    plot_graph(x_array, y_array, exp_fit)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n_array = np.array([5, 6, 7, 8, 9])
    av_probs = np.zeros(len(n_array))

    for i, n in enumerate(n_array):
        probs = np.loadtxt("inf_time_probs_n_" + str(n) + ".txt")
        av_probs[i] = 1/np.mean(probs)

    fit_and_plot(n_array, av_probs)
