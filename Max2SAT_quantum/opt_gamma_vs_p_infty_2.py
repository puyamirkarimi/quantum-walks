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

    n_array = np.array([5, 9])
    gamma_limit = 1.5
    gamma_step = 0.01

    fig, (ax1, ax2) = plt.subplots(1, 2)
    axes = (ax1, ax2)
    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)

    ax1.set_xlabel(r"$P_\infty$")
    ax2.set_xlabel(r"$P_\infty$")
    ax1.set_ylabel(r"$\gamma_{opt}$")

    colors = ['forestgreen', 'crimson']
    ax1.set_xlim([0, 0.35])
    ax2.set_xlim([0, 0.2])
    ax1.set_ylim([0.2, 1.3])
    ax2.set_ylim([0.2, 1.3])
    ax1.set_yticks(np.arange(0.4, 1.4, 0.2))

    for i, n in enumerate(n_array):
        opt_gammas = np.loadtxt("new_opt_gammas_"+str(n)+".txt")
        heur_gam = heuristic_gamma(n)
        probs1 = quantum_data_unopt(n)
        axes[i].scatter(probs1, opt_gammas, linewidths=0, marker='.', s=4, color=colors[i])
        axes[i].hlines(heur_gam, 0, 0.4, colors='yellow')

    plt.savefig('opt_gamma_vs_p_infty.png', dpi=200)
    # plt.show()