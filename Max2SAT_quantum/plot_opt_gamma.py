import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    n_array = np.array([5, 8])
    gamma_limit = 1.5
    gamma_step = 0.01

    plt.figure()

    #plt.plot(gammas_array, frequency)
    # plt.xlim([np.min(gammas), np.max(gammas)])
    # plt.ylim([0, 0.2])
    plt.xlabel("$\gamma$")
    plt.ylabel("$p(\gamma)$")

    for i, n in enumerate(n_array):
        unbinned_gammas = np.loadtxt("opt_gammas_"+str(n)+".txt")
        plt.hist(unbinned_gammas, np.arange(0, gamma_limit, gamma_step))

    plt.show()