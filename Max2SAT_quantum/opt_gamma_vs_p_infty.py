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

    n = 9
    gamma_limit = 1.5
    gamma_step = 0.01

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.25]})
    axes = (ax1, ax2)
    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)

    ax1.set_xlabel(r"$P_\infty$")
    ax2.set_xlabel(r"$P_\infty$")
    ax1.set_ylabel(r"$\gamma_{opt}$")
    for ax in axes:
        ax.set_xlim([0, 0.2])
        ax.set_ylim([0.15, 1.4])

    probs1 = quantum_data_unopt(n)
    probs2 = quantum_data_opt(n)
    delta_probs = probs2 - probs1
    cm = plt.get_cmap("seismic")

    opt_gammas = np.loadtxt("new_opt_gammas_"+str(n)+".txt")
    heur_gam = heuristic_gamma(n)
    axes[0].scatter(probs1, opt_gammas, c=delta_probs, cmap=cm, vmin = -0.1, vmax=0.1, linewidths=0.075, marker='.', s=4, edgecolors='black')
    axes[0].hlines(heur_gam, 0, 0.2, colors='yellow')

    im = axes[1].scatter(probs2, opt_gammas, c=delta_probs, cmap=cm, vmin = -0.1, vmax=0.1, linewidths=0.075, marker='.', s=4, edgecolors='black')
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.ax.set_ylabel('$\Delta P_\infty$')
    axes[1].hlines(heur_gam, 0, 0.2, colors='yellow')

    # plt.savefig('opt_gamma_vs_p_infty_n_'+ str(n) +'.png', dpi=200)
    plt.show()