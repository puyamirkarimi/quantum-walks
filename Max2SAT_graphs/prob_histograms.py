import matplotlib.pyplot as plt
import numpy as np


def quantum_data(n):
    probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return np.reciprocal(probs)


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


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n_list = [7, 10]
    prob_list = []

    num_bins = 60

    for n in n_list:
        prob_list.append(quantum_data(n))

    min_runtime = np.min(np.array(prob_list).flatten())
    max_runtime = np.max(np.array(prob_list).flatten())
    print(min_runtime, max_runtime)

    x = np.linspace(min_runtime, max_runtime, num=num_bins)

    plt.figure()
    # for i_adam, n in enumerate(n_list):
    #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    plt.xlim([0, 100])
    plt.ylim([0.6, 1e4])
    print(np.swapaxes(np.array(prob_list), 0, 1))
    plt.hist(np.swapaxes(np.array(prob_list), 0, 1), x, color=('royalblue', 'deeppink'))
    plt.yscale('log')

    plt.xlabel(r"$1/P_\infty$")

    plt.ylabel("Frequency")
    # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")
    # plt.tight_layout()
    plt.savefig('probability_histogram.png', dpi=200)
    # plt.show()
