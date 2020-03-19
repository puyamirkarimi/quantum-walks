import numpy as np

if __name__ == '__main__':
    n_list = [5, 6, 7, 8, 9, 10]
    for n in n_list:
        energy_spreads = np.loadtxt("energy_spread_n_" + str(n) + ".txt")
        av_energy_spread = np.mean(energy_spreads)
        print("n=" + str(n), "average energy spread:", av_energy_spread)
        heur_gamma = (1/(2*n))*av_energy_spread
        print("n=" + str(n), "heuristic gamma:", heur_gamma)