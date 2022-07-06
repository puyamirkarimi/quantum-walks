import matplotlib.pyplot as plt
import numpy as np


def adams_quantum_walk_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 2].astype(float)


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

    blue = '#0072B2'
    orange = '#EF6900'
    green = '#009E73'

    n = 9
    counts_list = []

    num_bins = 50

    counts = adams_quantum_walk_data(n)
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
    # max_runtime = 0.002

    x = np.linspace(min_runtime, max_runtime, num=num_bins+1)

    x_logarithmic = np.ones(num_bins+1) * min_runtime
    multiply_factor = 2**((np.log(max_runtime)-np.log(min_runtime))/num_bins)
    x_logarithmic = [x_logarithmic[i] * multiply_factor**i for i in range(len(x_logarithmic))]

    sat_average = np.mean(counts_list[0])
    unsat_average = np.mean(counts_list[1])

    plt.figure()
    # for i_adam, n in enumerate(n_list):
    #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # plt.xlim([0, 100])
    # plt.ylim([0.6, 1e4])
    plt.hist(counts_list[0], x, color=orange, align='mid', rwidth=0.5, density=True, label='satisfiable')
    plt.hist(counts_list[1], x, color=green, align='left', rwidth=0.5, density=True, label='unsatisfiable')
    plt.vlines(sat_average, 0, 35, color='black')
    plt.vlines(unsat_average, 0, 35, color='black', linestyle='--')
    # plt.yscale('log')

    plt.xlabel(r"$\overline{P}(0, 100)$")
    plt.ylabel("Normalised probability")

    plt.legend()

    plt.tick_params(direction='in', top=True, right=True, which='both')
    # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")
    # plt.tight_layout()
    # plt.savefig('probability_histogram.png', dpi=200)
    plt.show()


    # ############################ LOGARITHMIC PLOT ###############################
    # plt.figure()
    # # for i_adam, n in enumerate(n_list):
    # #     plt.scatter(x, y_adam[i_adam], label="n="+str(n), marker='+')
    # # plt.errorbar(x, runtimes_average, runtimes_standard_error)
    # # plt.xlim([0, 100])
    # # plt.ylim([0.6, 1e4])
    # plt.hist(counts_list[0], x_logarithmic, color=orange, align='mid', rwidth=0.5, density=True, label='satisfiable')
    # plt.hist(counts_list[1], x_logarithmic, color=green, align='left', rwidth=0.5, density=True, label='unsatisfiable')
    # plt.xscale('log')

    # plt.xlabel(r"$\overline{P}(0, 100)$")
    # plt.ylabel("Normalised probability")

    # plt.legend()

    # plt.tick_params(direction='in', top=True, right=True, which='both')
    # # ax2.set_ylabel(r"$\overline{T}_{inst}$~/~$s$")
    # # plt.tight_layout()
    # # plt.savefig('probability_histogram.png', dpi=200)
    # plt.show()
