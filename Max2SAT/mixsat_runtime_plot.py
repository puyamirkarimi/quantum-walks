import matplotlib.pyplot as plt
import numpy as np


def average_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error = np.std(data[:, x]) / np.sqrt(num_repeats)

    return y_av, y_std_error


def plot_graph(x, y, y_std_error=None):
    fig, ax = plt.subplots()
    plt.plot(x, y)
    plt.scatter(x, y)
    #plt.errorbar(x, y, y_std_error)
    ax.set_xlabel("Number of variables, $n$")
    ax.set_ylabel("Average runtime ($s$)")
    ax.set_xlim([5, 20])
    ax.set_xticks(range(5, 21, 3))
    plt.show()


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    runtimes = np.loadtxt("mixsat_runtimes.txt").reshape((-1, 16))
    average, standard_error = average_data(runtimes)
    n_array = np.array(range(5, 21))
    plot_graph(n_array, average, y_std_error=standard_error)
