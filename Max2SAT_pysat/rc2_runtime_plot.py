import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic


def average_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error[x] = np.std(data[:, x], ddof=1) / np.sqrt(num_repeats)

    return y_av, y_std_error


# def plot_graph(x, y, y_std_error, fit_1, fit_2):
#     fig, ax = plt.subplots()
#     plt.scatter(x[4:], y[4:])
#     plt.scatter(x[:4], y[:4], color="gray")
#     plt.plot(x, fit_1, '--', label="$y=0.0005x + 0.0012$", color="red")
#     plt.plot(x, fit_2, label=r"$y=0.0036 \times 2^{0.0871x}$", color="green")
#     #plt.errorbar(x, y, y_std_error)
#     ax.set_xlabel("Number of variables, $n$")
#     ax.set_ylabel("Average runtime ($s$)")
#     ax.set_xlim([5, 20])
#     ax.set_xticks(range(5, 21, 3))
#     ax.set_ylim([0.004, 0.012])
#     ax.set_yscale('log')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


def counts_data_crosson():
    counts_crosson = np.loadtxt("crosson_counts.txt").reshape((-1, 137))
    return average_data(counts_crosson)


def runtimes_data_crosson():
    runtimes_crosson = np.loadtxt("crosson_runtimes.txt").reshape((-1, 137))
    return average_data(runtimes_crosson)


def counts_data_adam(n):
    counts_adam = np.loadtxt("adam_counts_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(counts_adam)


def runtimes_data_adam(n):
    runtimes_adam = np.loadtxt("adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(runtimes_adam)


def counts_data_adam_noGT(n):
    counts_adam = np.loadtxt("adam_noGT_counts_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(counts_adam)


def runtimes_data_adam_noGT(n):
    runtimes_adam = np.loadtxt("adam_noGT_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(runtimes_adam)


def counts_data_adam_noGT_nondg(n):
    counts_adam = np.loadtxt("adam_noGT_nondg_counts_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(counts_adam)


def runtimes_data_adam_noGT_nondg(n):
    runtimes_adam = np.loadtxt("adam_noGT_nondg_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(runtimes_adam)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=13)

    n_list = [5, 12, 20]
    counts_list_adam = []
    runtimes_list_adam = []


    ################## RUNTIMES ##################
    num_bins = 100

    for n in n_list:
        runtimes_adam_average, runtimes_adam_standard_error = runtimes_data_adam(n)
        print(np.argmax(runtimes_adam_average))
        runtimes_list_adam.append(runtimes_adam_average)

    min_runtime = np.min(np.array(runtimes_list_adam).flatten())
    max_runtime = np.max(np.array(runtimes_list_adam).flatten())
    print(min_runtime, max_runtime)

    x = np.linspace(min_runtime, max_runtime, num=num_bins)
    y_adam = np.zeros((len(n_list), len(x)))

    for i_adam in range(len(n_list)):
        y_adam[i_adam] = np.histogram(runtimes_list_adam[i_adam], bins=num_bins, density=True, range=(min_runtime, max_runtime))[0]

    for i_adam in range(len(n_list)):
        y_adam[i_adam] = zero_to_nan(y_adam[i_adam])          # replace zero elements in list with NaN so they aren't plotted

    fig2, ax2 = plt.subplots()
    for i_adam, n in enumerate(n_list):
        plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # plt.xlim([0, 0.021])
    # plt.ylim([9e-5, 0.013])
    plt.yscale('log')
    plt.xlabel("Rounded runtime of MIXSAT algorithm")
    plt.ylabel("Number of instances (normalised)")
    plt.legend()
    plt.show()

