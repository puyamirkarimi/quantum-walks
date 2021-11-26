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


def runtimes_data_adam(n):
    runtimes_adam = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(mask_data(runtimes_adam))


def runtimes_data_adam_pysat(n):
    runtimes_adam = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    return average_data(mask_data(runtimes_adam))


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (9.6, 4.8)

    n_list = [10, 20]
    counts_list_adam = []
    runtimes_list_adam = []

    num_bins = 40

    for n in n_list:
        runtimes_adam_average, runtimes_adam_standard_error = runtimes_data_adam(n)
        runtimes_list_adam.append(runtimes_adam_average)

    min_runtime = np.min(np.array(runtimes_list_adam).flatten())
    max_runtime = np.max(np.array(runtimes_list_adam).flatten())
    print(min_runtime, max_runtime)

    x = np.linspace(min_runtime, max_runtime, num=num_bins)

    fig2, (ax2, ax1) = plt.subplots(1, 2, sharey=True)
    # for i_adam, n in enumerate(n_list):
    #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # plt.xlim([0, 0.021])
    # plt.ylim([9e-5, 0.013])
    print(np.swapaxes(np.array(runtimes_list_adam), 0, 1))
    ax1.hist(np.swapaxes(np.array(runtimes_list_adam), 0, 1), x, color=('deeppink', 'mediumblue'))
    ax1.set_yscale('log')

    ax1.set_xlim([0, 0.017])

    # ax1.yaxis.tick_left()
    # ax1.tick_params(labelright='off')
    ax1.set_ylim([0.6, 4000])
    ax2.set_ylim([0.6, 5000])

    ax1.set_xlabel(r"$\overline{T}_{inst}$~/~$s$")
    # ax2.set_xlabel(r"$\overline{T}_{inst}$~/~$s$")
    ax2.set_ylabel(r"Frequency")
    # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")
    # plt.tight_layout()
    # plt.savefig('mixsat.png', dpi=300)
    # plt.show()


    ################## pysat ##################
    num_bins = 70
    runtimes_list_adam = []

    for n in n_list:
        runtimes_adam_average, runtimes_adam_standard_error = runtimes_data_adam_pysat(n)
        runtimes_list_adam.append(runtimes_adam_average)

    min_runtime = np.min(np.array(runtimes_list_adam).flatten())
    max_runtime = np.max(np.array(runtimes_list_adam).flatten())
    print(min_runtime, max_runtime)

    x = np.linspace(min_runtime, max_runtime, num=num_bins)
    # y_adam = np.zeros((len(n_list), len(x)))
    #
    # for i_adam in range(len(n_list)):
    #     y_adam[i_adam] = np.histogram(runtimes_list_adam[i_adam], bins=num_bins, density=True, range=(min_runtime, max_runtime))[0]
    #
    # for i_adam in range(len(n_list)):
    #     y_adam[i_adam] = zero_to_nan(y_adam[i_adam])          # replace zero elements in list with NaN so they aren't plotted

    # for i_adam, n in enumerate(n_list):
    #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # plt.xlim([0, 0.021])
    # plt.ylim([9e-5, 0.013])
    ax2.hist(np.swapaxes(np.array(runtimes_list_adam), 0, 1), x, color=('deeppink', 'mediumblue'))
    # ax1.set_aspect('equal', 'box')
    ax2.set_yscale('log')
    ax2.set_xlim([0, 0.001])
    ax2.set_xticks([0, 0.0004, 0.0008])
    ax2.set_ylim([0.6, 4000])
    ax2.tick_params(direction='in', top=True, right=True, which='both')
    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.set_xlabel(r"$\overline{T}_{inst}$~/~$s$")
    # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")

    # plt.savefig('runtimes_histograms.png', dpi=200)
    plt.tight_layout()
    plt.show()