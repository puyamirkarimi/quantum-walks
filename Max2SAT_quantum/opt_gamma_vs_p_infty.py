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


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n_array = np.array([8])
    gamma_limit = 1.5
    gamma_step = 0.01

    plt.figure()

    #plt.plot(gammas_array, frequency)
    plt.xlim([0, 1.5])
    plt.xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
    plt.ylim([0, 300])
    plt.xlabel("$\gamma_{opt}$")
    plt.ylabel("Frequency")

    for i, n in enumerate(n_array):
        unbinned_gammas = np.loadtxt("opt_gammas_3_"+str(n)+".txt")
        heur_gam = heuristic_gamma(n)

        num_less = 0
        num_more = 0
        for gamma in unbinned_gammas:
            if gamma > heur_gam:
                num_more += 1
            elif gamma <= heur_gam:
                num_less += 1
        mean_gam = np.mean(unbinned_gammas)
        median_gam = np.median(unbinned_gammas)
        print("number of opt gammas below heur gamma:", num_less)
        print("number of opt gammas above heur gamma:", num_more)
        print("mean opt gamma", mean_gam)

        plt.hist(unbinned_gammas, np.arange(0, gamma_limit, gamma_step), color='mediumblue')
        plt.vlines(heur_gam, 0, 300, colors='yellow')
        plt.vlines(mean_gam, 0, 300, colors='lime', linestyles='dotted')
        plt.vlines(median_gam, 0, 300, colors='orange', linestyles='dotted')

    # plt.savefig('opt_gamma_n_8.png', dpi=200)
    plt.show()