import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize


def average_data(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error[x] = np.std(data[:, x], ddof=1) / np.sqrt(num_repeats)

    return y_av, y_std_error


def bnb_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/bnb/mixbnb.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 4].astype(int)


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
            skipped += 1
    print("n:", n, " skipped:", skipped)
    return np.array(b)


def total_error(std_errors):
    """ From back page of Hughes and Hase errors book. Calculating error of averaging each instance runtime. """
    error = 0
    for std_err in std_errors:
        error += std_err**2
    return np.sqrt(error)/len(std_errors)


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


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


def runtimes_data(n, name):
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(runtimes)


def runtimes_data_masked(n, name):
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(mask_data(runtimes))


def counts_data(n, name):
    if name == "mixsat":
        counts = np.loadtxt("./../Max2SAT/adam_counts_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        counts = np.loadtxt("./../Max2SAT_bnb/adam_counts_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(counts)


def before_plot():
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$ 1/ \langle P \rangle $")
    ax.set_xlim([5, 20])
    ax.set_xticks(range(5, 21, 3))
    return fig, ax


def plot_graph(x, y, y_err=None, fit=None, label=None):
    # if label == "MIXSAT runtimes":
    #     plt.scatter(x[:4], y[:4], color='gray')
    #     plt.scatter(x[4:], y[4:], label=label)
    # else:
    #     plt.scatter(x, y, label=label)

    # plt.scatter(x, y, label=label)
    color = "yellow"
    if label == "PySAT runtimes":
        color = "blue"
    if label == "Classical B&B Counts":
        color = "blue"
    elif label == "MIXSAT runtimes":
        color = "forestgreen"
    elif label == "MIXSAT runtimes":
        color = "forestgreen"
    elif label == "QW":
        color = "red"
    elif label == "AQC":
        color = "blue"
    elif label == "Guess":
        color = "black"
        plt.plot(x, y, '--', color=color)
        return
    elif label == "Guess Sqrt":
        color = "gray"
        plt.plot(x, y, ':', color=color)
        return
    if y_err is not None:
        plt.errorbar(x, y, y_err, color=color, fmt='o', ms=4.2, capsize=1.5)
    else:
        plt.scatter(x, y, color=color, s=18)
    if fit is not None:
        plt.plot(x, fit, '--', color=color)


def after_plot(fig, ax):
    ratio = 0.02 / 0.00001
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([2**16/ratio, 2**16])
    ax.set_yscale('log', basey=2)
    # plt.legend(loc="upper right")
    return ratio


def after_plot2(fig, ax, scale):
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([2**0.5, scale*2**0.5])
    ax.set_yscale('log', basey=2)
    # plt.legend(loc="upper right")


def fit_and_plot(runtimes, label, y_err):
    n_array = np.array(range(5, len(runtimes)+5))
    if label == "MIXSAT counts":
        plot_graph(n_array, runtimes, None, label)
        return
    # m_log, c_log = np.polyfit(n_array[0:], np.log2(runtimes[0:]), 1, w=np.sqrt(runtimes[0:]))
    # print(label+":", str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    # exp_fit = np.exp2(m_log * n_array + c_log)
    opt, cov = optimize.curve_fit(lambda x, a, b: a * np.exp2(b * x), n_array, runtimes, p0=(0.0001, 0.087))
    a = opt[0]
    b = opt[1]
    a_error = np.sqrt(cov[0, 0])
    b_error = np.sqrt(cov[1, 1])
    exp_fit = a * np.exp2(b * n_array)
    print(label + ": " + str(a) + " * 2^(" + str(b) + " * n)")
    print("a error:", a_error, "b error:", b_error)
    plot_graph(n_array, runtimes, y_err, exp_fit, label)


def fit_and_plot2(x_array, y_array, label, y_err):
    # m_log, c_log = np.polyfit(x_array[0:], np.log2(y_array), 1, w=np.sqrt(y_array))
    # exp_fit = np.exp2(m_log * x_array + c_log)
    # print("Quantum:" + str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    opt, cov = optimize.curve_fit(lambda x, a, b: a * np.exp2(b * x), x_array, y_array, p0=(1, 0.5))
    a = opt[0]
    b = opt[1]
    a_error = np.sqrt(cov[0, 0])
    b_error = np.sqrt(cov[1, 1])
    exp_fit = a * np.exp2(b * x_array)
    print(label + ": " + str(a) + " * 2^(" + str(b) + " * n)")
    print("a error:", a_error, "b error:", b_error)
    plot_graph(x_array, y_array, y_err, exp_fit, label)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=26)
    colors = ['forestgreen', 'blue', 'red', 'black']

    max_n_others = 20

    # runtimes_mixsat = np.zeros(max_n_others - 4)
    # runtimes_pysat = np.zeros(max_n_others - 4)
    # errors_mixsat = np.zeros(max_n_others - 4)
    # errors_pysat = np.zeros(max_n_others - 4)

    # runtimes_bnb = np.zeros(max_n_others - 4)
    # errors_bnb = np.zeros(max_n_others - 4)

    # for i, n in enumerate(range(5, max_n_others+1)):
        # run_data_mixsat, error_data_mixsat = runtimes_data_masked(n, "mixsat")
        # runtimes_mixsat[i] = np.mean(run_data_mixsat.flatten())  # .flatten() needed?
        # errors_mixsat[i] = total_error(error_data_mixsat)
        #
        # run_data_pysat, error_data_pysat = runtimes_data_masked(n, "pysat")
        # runtimes_pysat[i] = np.mean(run_data_pysat.flatten())  # .flatten() needed?
        # errors_pysat[i] = total_error(error_data_pysat)

        # bnbdata = bnb_data(n)
        # runtimes_bnb[i] = np.mean(bnbdata)
        # errors_bnb[i] = np.std(bnbdata, ddof=1) / np.sqrt(len(bnbdata))

    # counts_bnb *= runtimes_bnb[0]/counts_bnb[0]     # normalisation
    # counts_mixsat *= runtimes_mixsat[0] / counts_mixsat[0]  # normalisation

    fig, ax2 = before_plot()
    plt.tick_params(direction='in', top=True, right=True)

    n_array = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    av_probs = np.zeros(len(n_array))
    quantum_errors = np.zeros(len(n_array))
    n_array2 = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    av_probs2 = np.zeros(len(n_array2))
    av_probs3 = np.zeros(len(n_array2))

    n_array_adiabatic = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    av_probs_adiabatic = np.zeros(len(n_array_adiabatic))
    quantum_errors_adiabatic = np.zeros(len(n_array_adiabatic))

    for i, n in enumerate(n_array):
        probs = adams_quantum_walk_data(n)
        av_probs[i] = 1 / np.mean(probs)
        quantum_errors[i] = np.std(1/probs, ddof=1) / np.sqrt(len(probs))
    for i, n in enumerate(n_array_adiabatic):
        probs = adams_adiabatic_data(n)
        av_probs_adiabatic[i] = np.mean(probs)
        quantum_errors_adiabatic[i] = np.std(probs, ddof=1) / np.sqrt(len(probs))

    # for i, n in enumerate(n_array2):
    #     probs2 = np.loadtxt("./../Max2SAT_quantum/zero_time_probs_n_" + str(n) + ".txt")
    #     av_probs2[i] = 1 / np.mean(probs2)
    #     av_probs3[i] = np.sqrt(av_probs2[i])

    av_probs2 = np.exp2(n_array2)
    av_probs3 = np.sqrt(n_array2)

    fit_and_plot2(n_array, av_probs, "QW", quantum_errors)

    fit_and_plot2(n_array2, av_probs2, "Guess", None)
    fit_and_plot2(n_array2, av_probs3, "Guess Sqrt", None)

    ax1 = ax2.twinx()
    ax1.set_ylabel(r"$\langle T_{0.99} \rangle$")
    fit_and_plot2(n_array_adiabatic, av_probs_adiabatic, "AQC", quantum_errors_adiabatic)
    # fit_and_plot(runtimes_bnb, "B\&B runtimes")
    # fit_and_plot(counts_bnb, "B\&B counts")
    # fit_and_plot(runtimes_pysat, "PySAT runtimes", errors_pysat)
    # fit_and_plot(runtimes_mixsat, "MIXSAT runtimes", errors_mixsat)
    # fit_and_plot(counts_mixsat, "MIXSAT counts")
    # fit_and_plot(runtimes_bnb, "Classical B&B Counts", errors_bnb)

    scale = after_plot(fig, ax1)
    after_plot2(fig, ax2, scale)
    ax1.tick_params(direction='in', right=True, which='both')
    ax2.tick_params(direction='in', top=True, which='both')
    plt.tight_layout()
    # plt.show()
    plt.savefig('scalings.png', dpi=200)
