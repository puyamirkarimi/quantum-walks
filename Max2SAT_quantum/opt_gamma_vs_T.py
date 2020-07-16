import numpy as np
import matplotlib.pyplot as plt


def heuristic_gamma(n):
    out = "haven't defined heuristic gamma for given n"
    if n == 5:
        out = 0.56503
    if n == 6:
        out = 0.587375
    if n == 7:
        out = 0.5984357142857143
    if n == 8:
        out = 0.60751875
    if n == 9:
        out = 0.6139833333333333
    if n == 10:
        out = 0.619345
    if n == 11:
        out = 0.6220136363636364
    print("heuristic gamma: ", out)
    return out


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


def runtimes_data_masked(n, name):
    if name.lower() == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name.lower() == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name.lower() == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(mask_data(runtimes))


def quantum_data_unopt(n):
    probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
    return probs


def quantum_data_opt(n):
    probs = np.loadtxt("./../Max2SAT_quantum/opt_inf_time_probs_n_" + str(n) + ".txt")
    return probs


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (9.6, 4.8)

    n_array = np.array([9])
    gamma_limit = 1.5
    gamma_step = 0.01

    fig, (ax1, ax2) = plt.subplots(1, 2)
    axes = (ax1, ax2)
    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)

    ax1.set_xlabel("$T_{inst}$")
    ax2.set_xlabel("$T_{inst}$")
    ax1.set_ylabel("$\gamma_{opt}$")

    for i, n in enumerate(n_array):
        opt_gammas = np.loadtxt("new_opt_gammas_"+str(n)+".txt")
        heur_gam = heuristic_gamma(n)
        probs = runtimes_data_masked(n, "pysat")[0]
        axes[i].scatter(probs, opt_gammas, linewidths=0, marker='.', s=4, c='deeppink')
        axes[i] .hlines(heur_gam, 0, 0.0004, colors='yellow')

    opt_gammas = np.loadtxt("new_opt_gammas_" + str(n_array[0]) + ".txt")
    heur_gam = heuristic_gamma(n_array[0])
    probs = runtimes_data_masked(n_array[0], "mixsat")[0]
    axes[1].scatter(probs, opt_gammas, s=0.25)
    axes[1].hlines(heur_gam, 0, 0.007, colors='yellow')

    # plt.savefig('opt_gamma_vs_p_infty.png', dpi=200)
    plt.show()