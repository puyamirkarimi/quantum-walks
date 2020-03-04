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
    plt.rc('font', size=16)

    n_list = [5, 9, 20]
    counts_list_adam = []
    runtimes_list_adam = []

    ################## COUNTS ##################
    counts_crosson_average, counts_crosson_standard_error = counts_data_crosson()
    min_count = np.min(counts_crosson_average)
    max_count = np.max(counts_crosson_average)

    for n in n_list:
        counts_adam_average, counts_adam_standard_error = counts_data_adam(n)
        counts_list_adam.append(counts_adam_average)
        min_count_temp = np.min(counts_adam_average)
        max_count_temp = np.max(counts_adam_average)
        if min_count_temp < min_count:
            min_count = min_count_temp
        if max_count_temp > max_count:
            max_count = max_count_temp

    x = np.arange(np.floor(min_count), np.ceil(max_count)+1)
    y_adam = np.zeros((len(n_list), len(x)))
    y_crosson = np.zeros(len(x))

    for i, count in enumerate(x):
        for i_adam in range(len(n_list)):
            y_adam[i_adam, i] = np.count_nonzero(counts_list_adam[i_adam] == count) / 10000           # division by 10000 is to normalise
        y_crosson[i] = np.count_nonzero(counts_crosson_average == count) / 137

    for i_adam in range(len(n_list)):
        y_adam[i_adam] = zero_to_nan(y_adam[i_adam])          # replace zero elements in list with NaN so they aren't plotted

    y_crosson = zero_to_nan(y_crosson)

    fig1, ax1 = plt.subplots()
    for i_adam, n in enumerate(n_list):
        plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    #plt.scatter(x, y_crosson, label="n=20 (Crosson)")
    #plt.errorbar(x, counts_average, counts_standard_error)
    plt.xlim([0, 375])
    plt.ylim([9e-5, 1])
    plt.yscale('log')
    plt.legend()
    plt.xlabel("Number of states visited by MIXSAT algorithm")
    plt.ylabel("Number of instances (normalised)")
    plt.tight_layout()
    plt.show()

    ################## RUNTIMES ##################
    runtimes_crosson_average, runtimes_crosson_standard_error = runtimes_data_crosson()
    runtimes_crosson_average = np.around(runtimes_crosson_average, 4)           # binning
    min_runtime = np.min(runtimes_crosson_average)
    max_runtime = np.max(runtimes_crosson_average)

    for n in n_list:
        runtimes_adam_average, runtimes_adam_standard_error = runtimes_data_adam(n)
        print(np.argmax(runtimes_adam_average))
        runtimes_adam_average = np.around(runtimes_adam_average, 5)         # binning
        runtimes_list_adam.append(runtimes_adam_average)
        min_runtime_temp = np.min(runtimes_adam_average)
        max_runtime_temp = np.max(runtimes_adam_average)
        if min_runtime_temp < min_runtime:
            min_runtime = min_runtime_temp
        if max_runtime_temp > max_runtime:
            max_runtime = max_runtime_temp

    x = np.arange(np.floor(min_runtime), np.ceil(max_runtime) + 1, step=0.00001)
    x_crosson = np.arange(np.floor(min_runtime), np.ceil(max_runtime) + 1, step=0.0001)
    y_adam = np.zeros((len(n_list), len(x)))
    y_crosson = np.zeros(len(x_crosson))

    for i, runtime in enumerate(x):
        for i_adam in range(len(n_list)):
            y_adam[i_adam, i] = np.count_nonzero(runtimes_list_adam[i_adam] == runtime) / 10000           # division by 10000 is to normalise
    for i, runtime in enumerate(x_crosson):
        y_crosson[i] = np.count_nonzero(runtimes_crosson_average == runtime) / 1370

    for i_adam in range(len(n_list)):
        y_adam[i_adam] = zero_to_nan(y_adam[i_adam])          # replace zero elements in list with NaN so they aren't plotted

    y_crosson = zero_to_nan(y_crosson)

    fig2, ax2 = plt.subplots()
    for i_adam, n in enumerate(n_list):
        plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    #plt.scatter(x_crosson, y_crosson, label="n=20 (Crosson)")
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    plt.xlim([0, 0.021])
    plt.ylim([9e-5, 0.013])
    plt.yscale('log')
    plt.xlabel("Binned runtime of MIXSAT algorithm ($s$)")
    plt.ylabel("Normalised number of instances")
    plt.legend()
    plt.tight_layout()
    plt.show()

