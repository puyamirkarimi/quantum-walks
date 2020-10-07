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


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


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


def bnb_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/bnb/mixbnb.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 4].astype(int)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=24)
    plt.rcParams["figure.figsize"] = (6, 6)

    marker_size = 4

    n = 15
    fig, ax = plt.subplots()

    x = adams_quantum_walk_data(n)
    y = bnb_data(n)

    min_x = np.min(x)
    min_y = np.min(y)
    max_x = np.max(x)
    max_y = np.max(y)

    ax.scatter(x, y, label="n=" + str(n), marker='.', s=marker_size, linewidths=0)
    # ax.set_xlim([8, 150])
    # ax.set_ylim([19, 34000])
    ax.set_xlabel("$P_{\infty}$")
    ax.set_ylabel(r"$N_{calls}$")
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)

    # plt.tight_layout()
    # plt.show()
    plt.savefig('n_'+str(n)+'_bnb_vs_QW.png', dpi=200)
