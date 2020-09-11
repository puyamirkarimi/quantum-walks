import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as optimize


def plot_graph(x, y, y_err=None, fit=None, label=None, clr='black'):
    if y_err is not None:
        plt.errorbar(x, y, y_err, fmt='o', ms=4.2, capsize=1.5, color=clr)
    else:
        plt.scatter(x, y, s=18, color=clr)
    if fit is not None:
        plt.plot(x, fit, '--', color=clr)


def fit_and_plot(x_array, y_array, label, y_err, clr='black'):
    opt, cov = optimize.curve_fit(lambda x, a, b: a * np.exp2(b * x), x_array, y_array, p0=(1, 0.5))
    a = opt[0]
    b = opt[1]
    a_error = np.sqrt(cov[0, 0])
    b_error = np.sqrt(cov[1, 1])
    exp_fit = a * np.exp2(b * x_array)
    print(label + ": " + str(a) + " * 2^(" + str(b) + " * n)")
    print("a error:", a_error, "b error:", b_error)
    plot_graph(x_array, y_array, y_err, exp_fit, label, clr=clr)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)
    n_list = np.array([5,6,7,8,9,10,11,12,13])
    mean_times = np.zeros(len(n_list))
    median_times = np.zeros(len(n_list))
    errors = np.zeros(len(n_list))
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
                times[i] = 1000000000000
            else:
                new_times.append(time)

        print("Discarded", num_discarded, "instances for n =", n)
        mean_times[j] = np.mean(new_times)
        median_times[j] = np.median(times)
        errors[j] = np.std(new_times, ddof=1) / np.sqrt(len(times))

    plt.figure()
    plt.tick_params(direction='in', top=True, right=True, which='both')
    fit_and_plot(n_list, mean_times, "mean", errors, clr='blue')
    fit_and_plot(n_list, median_times, "median", errors, clr='red')
    # plt.xlim(start, end)
    # plt.ylim(start, end)
    # plt.xticks(range(start, end + 1, step))
    # plt.yticks(range(start, end + 1, step))
    plt.xlabel("$n$")
    plt.yscale('log', basey=2)
    plt.ylim([2**5, 2**8])
    plt.yticks([2**5, 2**5.5, 2**6, 2**6.5, 2**7, 2**7.5, 2**8])
    plt.xticks(range(5,14))
    plt.ylabel(r"$\langle T_{0.99} \rangle$")
    # plt.show()
    plt.savefig('times_scaling.png', dpi=200)

