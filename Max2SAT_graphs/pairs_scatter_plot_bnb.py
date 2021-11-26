import matplotlib.pyplot as plt
import numpy as np


def bnb_data(n):
    return np.loadtxt('./../Max2SAT_bnb/paired_counts_{}.txt'.format(n))


def bnb_data_transformed(n):
    return np.loadtxt('./../Max2SAT_bnb/paired_transformed_counts_{}.txt'.format(n))


def quantum_walk_data(n):
    probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return probs


def adiabatic_data(n):
    if n <= 8:
        times = np.loadtxt("./../Max2SAT_quantum/adiabatic/new_adiabatic_time_n_" + str(n) + ".txt")
    else:
        times = np.genfromtxt('./../Max2SAT_quantum/adiabatic/adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)[:, 1].astype(int)
    return times


def adams_quantum_walk_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 2].astype(float)


def adams_adiabatic_data(n):
    a = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', missing_values='', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 10]
    b = []
    skipped = 0
    for i, element in enumerate(a):
        if element != '':
            b.append(float(element))
        else:
            b.append(float('nan'))
            skipped += 1
    print("n:", n, " skipped:", skipped)
    return np.array(b)


def get_satisfiable_list(n):
    data = np.genfromtxt('./../instance_gen/m2s_pairs_satisfiable.csv', delimiter=',', skip_header=1, dtype=str)
    satisfiable_data = data[:, 1]
    m = n - 5
    return satisfiable_data[m*10000:(m+1)*10000]


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=26)
    plt.rcParams["figure.figsize"] = (6, 6)

    marker_size = 4

    n = 5
    fig, ax = plt.subplots()

    x = bnb_data(n)
    y = bnb_data_transformed(n)

    satisfiable = get_satisfiable_list(n).astype(int)

    x_av = np.mean(x)
    y_av = np.mean(y)
    x_err = np.std(x, ddof=1)/np.sqrt(len(x))
    y_err = np.std(y, ddof=1)/np.sqrt(len(y))
    print('average counts (untransformed): {} +- {}'.format(x_av, x_err))
    print('average counts (transformed): {} +- {}'.format(y_av, y_err))

    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)

    colors = ['green', 'red']
    from matplotlib.colors import ListedColormap

    ax.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0, c=satisfiable, cmap=ListedColormap(colors))
    # ax.set_xlim([8, 150])
    # ax.set_ylim([19, 34000])
    ax.set_xlabel('Counts (untransformed instance)')
    ax.set_ylabel('Counts (transformed instance)')
    # ax.set_xscale('log', basex=2)
    # ax.set_yscale('log', basey=2)

    plt.tight_layout()
    plt.show()