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


def probs_data(n):
    data = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return np.reciprocal(data)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n_list = [5, 10]
    probs_list = []

    num_bins = 100

    for n in n_list:
        probs = probs_data(n)
        probs_list.append(probs)

    min_prob = np.min(np.array(probs_list).flatten())
    max_prob = np.max(np.array(probs_list).flatten())
    print(min_prob, max_prob)

    x = np.linspace(min_prob, max_prob, num=num_bins)

    fig, ax1 = plt.subplots()
    # for i_adam, n in enumerate(n_list):
    #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # plt.xlim([0, 0.021])
    # plt.ylim([9e-5, 0.013])
    print(np.shape(np.array(probs_list)))
    ax1.hist(np.swapaxes(np.array(probs_list), 0, 1), x, color=('deeppink', 'seagreen'))
    ax1.set_yscale('log')

    ax1.set_xlim([0, 100])

    # ax1.yaxis.tick_left()
    # ax1.tick_params(labelright='off')
    ax1.set_ylim([0, 10e4])

    ax1.set_xlabel(r"$P_\infty$")
    # ax2.set_xlabel(r"$\overline{T}_{inst}$~/~$s$")
    ax1.set_ylabel(r"$p(P_\infty)$")
    # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")
    # plt.tight_layout()
    # plt.savefig('mixsat.png', dpi=300)
    # plt.show()

    ax1.tick_params(direction='in', top=True, right=True, which='both')

    # plt.savefig('runtimes_histograms.png', dpi=200)
    plt.tight_layout()
    plt.show()