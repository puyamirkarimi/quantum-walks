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


def plot_graph(x, y, y_std_error, fit_1, fit_2):
    fig, ax = plt.subplots()
    plt.scatter(x[4:], y[4:])
    plt.scatter(x[:4], y[:4], color="gray")
    plt.plot(x, fit_1, '--', label="$y=0.0005x + 0.0012$", color="red")
    plt.plot(x, fit_2, label=r"$y=0.0036 \times 2^{0.0871x}$", color="green")
    #plt.errorbar(x, y, y_std_error)
    ax.set_xlabel("Number of variables, $n$")
    ax.set_ylabel("Average runtime ($s$)")
    ax.set_xlim([5, 20])
    ax.set_xticks(range(5, 21, 3))
    ax.set_ylim([0.004, 0.012])
    ax.set_yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)

    runtimes = np.loadtxt("mixsat_runtimes_averaged.txt").reshape((-1, 16))
    average, standard_error = average_data(runtimes)
    n_array = np.array(range(5, 21))
    m_linear, c_linear = np.polyfit(n_array[4:], average[4:], 1)
    linear_fit = m_linear * n_array + c_linear
    m_log, c_log = np.polyfit(n_array[4:], np.log2(average[4:]), 1, w=np.sqrt(average[4:]))
    print(m_log, c_log, np.exp2(c_log))
    print(m_linear, c_linear)
    exp_fit = np.exp2(m_log * n_array + c_log)
    plot_graph(n_array, average, standard_error, linear_fit, exp_fit)
