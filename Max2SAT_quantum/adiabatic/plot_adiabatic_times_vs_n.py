import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)
    n_list = np.array([5,6,7,8,9,10,11,12])
    mean_times = np.zeros(len(n_list))
    median_times = np.zeros(len(n_list))
    for j, n in enumerate(n_list):
        print("-----", n, "------")
        if n > 8:
            data = np.genfromtxt('adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)
            times = data[:, 1].astype(int)
            success = data[:, 2]
        else:
            times = np.loadtxt("new_adiabatic_time_n_" + str(n) + ".txt")
            success = np.ones(10000)

        new_times = []
        num_discarded = 0
        for i, time in enumerate(times):
            failed = success[i] == 'False'
            if time < 0 or failed:
                num_discarded += 1
                times[i] = 10000
            else:
                new_times.append(time)

        print("Discarded", num_discarded, "instances for n =", n)
        mean_times[j] = np.mean(new_times)
        median_times[j] = np.median(times)

    plt.figure()
    plt.tick_params(direction='in', top=True, right=True, which='both')
    plt.scatter(n_list, mean_times)
    plt.scatter(n_list, median_times)
    # plt.xlim(start, end)
    # plt.ylim(start, end)
    # plt.xticks(range(start, end + 1, step))
    # plt.yticks(range(start, end + 1, step))
    plt.xlabel("$n$")
    plt.yscale('log', basey=2)
    plt.ylabel(r"$\langle T_{0.99} \rangle$")
    plt.show()



