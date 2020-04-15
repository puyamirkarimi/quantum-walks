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


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


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
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(mask_data(runtimes))


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


def runtimes_data_unaveraged(n, name):
    if name == "mixsat":
        runtimes = np.loadtxt("./../Max2SAT/adam_runtimes_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "pysat":
        runtimes = np.loadtxt("./../Max2SAT_pysat/adam_runtimes_" + str(n) + ".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        runtimes = np.loadtxt("./../Max2SAT_bnb/adam_runtimes_processtime_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return runtimes


def counts_data(n, name):
    if name == "mixsat":
        counts = np.loadtxt("./../Max2SAT/adam_counts_"+str(n)+".txt").reshape((-1, 10000))
    elif name == "branch and bound":
        counts = np.loadtxt("./../Max2SAT_bnb/adam_counts_" + str(n) + ".txt").reshape((-1, 10000))
    else:
        raise Exception
    return average_data(counts)


def plot_graph(x, y, fit, label):
    # if label == "MIXSAT runtimes":
    #     plt.scatter(x[:4], y[:4], color='gray')
    #     plt.scatter(x[4:], y[4:], label=label)
    # else:
    #     plt.scatter(x, y, label=label)

    # plt.scatter(x, y, label=label)
    color = "yellow"
    if label == "PySAT runtimes":
        color = "blue"
    elif label == "MIXSAT runtimes":
        color = "forestgreen"
    elif label == "MIXSAT runtimes":
        color = "forestgreen"
    elif label == "Quantum":
        color = "red"
    elif label == "Guess":
        color = "black"
        plt.plot(x, y, '--', color=color)
        return
    elif label == "Guess Sqrt":
        color = "gray"
        plt.plot(x, y, ':', color=color)
        return
    plt.scatter(x, y, color=color, s=18)
    if fit is not None:
        plt.plot(x, fit, '--', color=color)


def after_plot(fig, ax):
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([0.00001, 0.02])
    ax.set_yscale('log')
    # plt.legend(loc="upper right")
    return 0.02 / 0.00001


def after_plot2(fig, ax, scale):
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([4, scale*4])
    ax.set_yscale('log')
    # plt.legend(loc="upper right")


def fit_and_plot(runtimes, label):
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
    plot_graph(n_array, runtimes, exp_fit, label)


def fit_and_plot2(x_array, y_array, label):
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
    plot_graph(x_array, y_array, exp_fit, label)


def fit_and_error(x_array, y_array, label):
    # m_log, c_log = np.polyfit(x_array[0:], np.log2(y_array), 1, w=np.sqrt(y_array))
    # exp_fit = np.exp2(m_log * x_array + c_log)
    # print("Quantum:" + str(np.exp2(c_log))+" * 2^(" + str(m_log) + " * n)")
    opt, cov = optimize.curve_fit(lambda x, a, b: a * np.exp2(b * x), x_array, y_array, p0=(1, 0.5))
    a = opt[0]
    b = opt[1]
    a_error = np.sqrt(cov[0, 0])
    b_error = np.sqrt(cov[1, 1])
    print(label + ": " + str(a) + " * 2^(" + str(b) + " * n)")
    print("a error:", a_error, "b error:", b_error)
    return (a, a_error), (b, b_error)


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)
    colors = ['forestgreen', 'blue', 'red', 'black']

    max_n_others = 20
    n_array = np.array([5, 6, 7, 8, 9, 10])
    n_array2 = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    hard_percentage_array = np.array([.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1])
    # hard_percentage_array = np.array([0.1])

    scalings_mixsat = np.zeros(len(hard_percentage_array))
    scalings_pysat = np.zeros(len(hard_percentage_array))
    errors_mixsat = np.zeros(len(hard_percentage_array))
    errors_pysat = np.zeros(len(hard_percentage_array))
    scalings_quantum_pysat = np.zeros(len(hard_percentage_array))
    errors_quantum_mixsat = np.zeros(len(hard_percentage_array))
    scalings_quantum_mixsat = np.zeros(len(hard_percentage_array))
    errors_quantum_pysat = np.zeros(len(hard_percentage_array))

    for j, hard_percentage in enumerate(hard_percentage_array):
        runtimes_mixsat = np.zeros(max_n_others - 4)
        runtimes_pysat = np.zeros(max_n_others - 4)

        num_hard_instances = int(hard_percentage * 10000)
        # num_hard_instances_total = int(2*num_hard_instances)
        hard_instances_mixsat = np.zeros((16, num_hard_instances), dtype=int)
        hard_instances_pysat = np.zeros((16, num_hard_instances), dtype=int)

        for i, n in enumerate(range(5, max_n_others+1)):
            masked_data_mixsat = runtimes_data_masked(n, "mixsat")[0].flatten()
            masked_data_pysat = runtimes_data_masked(n, "pysat")[0].flatten()
            runtimes_data_mixsat = runtimes_data(n, "mixsat")[0].flatten()
            runtimes_data_pysat = runtimes_data(n, "pysat")[0].flatten()
            hard_instances_mixsat[i, :] = np.argpartition(masked_data_mixsat, -1*num_hard_instances)[-1*num_hard_instances:]
            hard_instances_pysat[i, :] = np.argpartition(masked_data_pysat, -1 * num_hard_instances)[-1 * num_hard_instances:]
            for instance in hard_instances_mixsat[i]:
                runtimes_mixsat[i] += runtimes_data_mixsat[instance]/num_hard_instances
                # runtimes_pysat[i] += runtimes_data_pysat[instance]/num_hard_instances_total
            for instance in hard_instances_pysat[i]:
                # runtimes_mixsat[i] += runtimes_data_mixsat[instance]/num_hard_instances_total
                runtimes_pysat[i] += runtimes_data_pysat[instance]/num_hard_instances

        print("---------- fraction:", hard_percentage)
        scaling_pysat, error_pysat = fit_and_error(n_array2, runtimes_pysat, "PySAT runtimes")[1]
        scaling_mixsat, error_mixsat = fit_and_error(n_array2, runtimes_mixsat, "MIXSAT runtimes")[1]

        av_probs_mixsat = np.zeros(len(n_array))
        av_probs_pysat = np.zeros(len(n_array))

        for i, n in enumerate(n_array):
            probs = np.loadtxt("./../Max2SAT_quantum/inf_time_probs_n_" + str(n) + ".txt")
            av_prob_mixsat = 0
            av_prob_pysat = 0
            for instance in hard_instances_pysat[i]:
                av_prob_pysat += probs[instance]/num_hard_instances
            for instance in hard_instances_mixsat[i]:
                av_prob_mixsat += probs[instance]/num_hard_instances
            av_probs_mixsat[i] = 1/av_prob_mixsat
            av_probs_pysat[i] = 1 / av_prob_pysat

        scaling_quantum_mixsat, error_quantum_mixsat = fit_and_error(n_array, av_probs_mixsat, "Quantum probabilities")[1]
        scaling_quantum_pysat, error_quantum_pysat = fit_and_error(n_array, av_probs_pysat, "Quantum probabilities")[1]

        scalings_mixsat[j] = scaling_mixsat
        errors_mixsat[j] = error_mixsat
        scalings_pysat[j] = scaling_pysat
        errors_pysat[j] = error_pysat
        scalings_quantum_mixsat[j] = scaling_quantum_mixsat
        errors_quantum_mixsat[j] = error_quantum_mixsat
        scalings_quantum_pysat[j] = scaling_quantum_pysat
        errors_quantum_pysat[j] = error_quantum_pysat

    # for i, n in enumerate(n_array2):
    #     probs2 = np.loadtxt("./../Max2SAT_quantum/zero_time_probs_n_" + str(n) + ".txt")
    #     av_probs2[i] = 1 / np.mean(probs2)
    #     av_probs3[i] = np.sqrt(av_probs2[i])

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    axes = (ax1, ax2)
    ax2.set_xlabel("$1-f$")
    ax2.set_ylabel("$b$")
    ax1.set_xlim([0, 1])
    ax2.set_xlim([0, 1])
    ax1.set_ylim([0.38, 0.55])
    ax1.set_yticks([0.4, 0.45, 0.5, 0.55])
    ax2.set_ylim([0, 0.17])
    ax2.set_yticks([0, 0.05, 0.1, 0.15])
    for ax in axes:
        ax.errorbar(1-hard_percentage_array, scalings_quantum_mixsat, errors_quantum_mixsat, color='red', marker='o', ms=4.2, capsize=1.5)
        ax.errorbar(1-hard_percentage_array, scalings_quantum_pysat, errors_quantum_pysat, color='purple', marker='o', ms=4.2, capsize=1.5)
        ax.errorbar(1-hard_percentage_array, scalings_pysat, errors_pysat, color='blue', marker='o', ms=4.2, capsize=1.5)
        ax.errorbar(1-hard_percentage_array, scalings_mixsat, errors_pysat, color='forestgreen', marker='o', ms=4.2, capsize=1.5)
    # plt.tight_layout()

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # This looks pretty good, and was fairly painless, but you can get that
    # cut-out diagonal lines look with just a bit more work. The important
    # thing to know here is that in axes coordinates, which are always
    # between 0-1, spine endpoints are at these locations (0,0), (0,1),
    # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
    # appropriate corners of each of our axes, and so long as we use the
    # right transform and disable clipping.

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.savefig("scaling_hard.png", dpi=200)
    # plt.show()
