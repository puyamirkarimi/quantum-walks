import matplotlib.pyplot as plt
import numpy as np


def counts_data(n):
    return np.loadtxt("./../Max2SAT/adam_counts_" + str(n) + ".txt")[:10000]


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


def get_satisfiable_list(n):
    data = np.genfromtxt('./../Max2SAT/m2s_satisfiable.csv', delimiter=',', skip_header=1, dtype=str)
    satisfiable_data = data[:, 1]
    m = n - 5
    return satisfiable_data[m*10000:(m+1)*10000]


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)


    n = 20
    counts_list = []

    num_bins = 50

    counts = counts_data(n)
    satisfiable_list = get_satisfiable_list(n)
    
    satisfiable_counts = []
    unsatisfiable_counts = []
    for i in range(len(counts)):
        if satisfiable_list[i] == '1':
            satisfiable_counts.append(counts[i])
        elif satisfiable_list[i] == '0':
            unsatisfiable_counts.append(counts[i])

    counts_list.append(np.array(satisfiable_counts))
    counts_list.append(np.array(unsatisfiable_counts))

    min_runtime = np.min((np.min(counts_list[0]), np.min(counts_list[1])))
    max_runtime = np.max((np.max(counts_list[0]), np.max(counts_list[1])))

    x = np.linspace(min_runtime, max_runtime, num=num_bins+1)

    plt.figure()
    # for i_adam, n in enumerate(n_list):
    #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # plt.xlim([0, 100])
    # plt.ylim([0.6, 1e4])
    plt.hist(counts_list[0], x, color='red', align='mid', rwidth=0.5, density=True, label='satisfiable')
    plt.hist(counts_list[1], x, color='green', align='left', rwidth=0.5, density=True, label='unsatisfiable')
    # plt.yscale('log')

    plt.xlabel("Number of calls (bnb)")
    plt.ylabel("Normalised probability")

    plt.legend()

    plt.tick_params(direction='in', top=True, right=True, which='both')
    # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")
    # plt.tight_layout()
    # plt.savefig('probability_histogram.png', dpi=200)
    plt.show()
