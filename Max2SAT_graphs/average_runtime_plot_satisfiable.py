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


def adams_quantum_walk_data(n):
    '''average success prob from T=0 to T=100'''
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 2].astype(float)


def adams_quantum_walk_data_crosson():
    '''average success prob from T=0 to T=100 for the crosson instances'''
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug_crosson.csv', delimiter=',', skip_header=1, dtype=str)[:,2].astype(float)


def adams_adiabatic_data(n):
    a = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', missing_values='', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 10]
    b = []
    skipped = 0
    for i, element in enumerate(a):
        if element != '':
            b.append(float(element))
        else:
            b.append(float('nan'))
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


def before_plot():
    fig, ax = plt.subplots()
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$ 1/ \langle \overline{P}(0, 100) \rangle $")
    ax.set_xlim([4.8, 20.2])
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
    if label == "QW satisfiable":
        color = "red"
    elif label == "QW unsatisfiable":
        color = "orange"
    elif label == "AQC satisfiable":
        color = "blue"
    elif label == "AQC unsatisfiable":
        color = "forestgreen"
    elif label == "QW Crosson":
        color = "Purple"
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
    scale = 0.02 / 0.00001
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([2**15/scale, 2**15])
    ax.set_yscale('log', base=2)
    # plt.legend(loc="upper right")
    return scale


def after_plot2(fig, ax, scale):
    # plt.errorbar(x, y, y_std_error)
    ax.set_ylim([2**0, scale*2**0])
    ax.set_yscale('log', base=2)
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


def get_satisfiable_list(n):
    data = np.genfromtxt('./../Max2SAT/m2s_satisfiable.csv', delimiter=',', skip_header=1, dtype=str)
    satisfiable_data = data[:, 1]
    m = n - 5
    return satisfiable_data[m*10000:(m+1)*10000]


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)
    colors = ['forestgreen', 'blue', 'red', 'black']

    fig, ax2 = before_plot()
    plt.tick_params(direction='in', top=True, right=True)

    n_array_qw = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    av_qw_probs_satisfiable = np.zeros(len(n_array_qw))
    av_qw_probs_unsatisfiable = np.zeros(len(n_array_qw))
    errors_qw_satisfiable = np.zeros(len(n_array_qw))
    errors_qw_unsatisfiable = np.zeros(len(n_array_qw))

    n_array_aqc = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    av_aqc_times_satisfiable = np.zeros(len(n_array_aqc))
    av_aqc_times_unsatisfiable = np.zeros(len(n_array_aqc))
    errors_aqc_satisfiable = np.zeros(len(n_array_aqc))
    errors_aqc_unsatisfiable = np.zeros(len(n_array_aqc))

    crosson_probs = adams_quantum_walk_data_crosson()
    av_crosson_prob = 1 / np.mean(crosson_probs)
    error_crosson = np.std(1/crosson_probs, ddof=1) / np.sqrt(len(crosson_probs))

    for i, n in enumerate(n_array_qw):
        probs = adams_quantum_walk_data(n)
        satisfiable_list = get_satisfiable_list(n).astype(int)

        probs_satisfiable = np.delete(probs, np.where(satisfiable_list == 0))
        probs_unsatisfiable = np.delete(probs, np.where(satisfiable_list == 1))

        num_satisfiable = len(probs_satisfiable)
        num_unsatisfiable = len(probs_unsatisfiable)

        print(f'Number of satisfiable instances for n={n} is {num_satisfiable} (out of 10000)')

        av_qw_probs_satisfiable[i] = 1 / np.mean(probs_satisfiable)
        av_qw_probs_unsatisfiable[i] = 1 / np.mean(probs_unsatisfiable)

        errors_qw_satisfiable[i] = np.std(1/probs_satisfiable, ddof=1) / np.sqrt(len(probs_satisfiable))
        errors_qw_unsatisfiable[i] = np.std(1/probs_unsatisfiable, ddof=1) / np.sqrt(len(probs_unsatisfiable))

    for i, n in enumerate(n_array_aqc):
        times = adams_adiabatic_data(n)
        satisfiable_list = get_satisfiable_list(n).astype(int)

        times_satisfiable = np.delete(times, np.where(satisfiable_list == 0))
        times_unsatisfiable = np.delete(times, np.where(satisfiable_list == 1))
        times_satisfiable = np.delete(times_satisfiable, np.where(np.isnan(times_satisfiable)))
        times_unsatisfiable = np.delete(times_unsatisfiable, np.where(np.isnan(times_unsatisfiable)))

        num_satisfiable = len(times_satisfiable)
        num_unsatisfiable = len(times_unsatisfiable)
        
        av_aqc_times_satisfiable[i] = np.mean(times_satisfiable)
        av_aqc_times_unsatisfiable[i] = np.mean(times_unsatisfiable)

        errors_aqc_satisfiable[i] = np.std(times_satisfiable, ddof=1) / np.sqrt(len(times_satisfiable))
        errors_qw_unsatisfiable[i] = np.std(times_unsatisfiable, ddof=1) / np.sqrt(len(times_unsatisfiable))


    fit_and_plot2(n_array_qw, av_qw_probs_satisfiable, "QW satisfiable", errors_qw_satisfiable)
    fit_and_plot2(n_array_qw, av_qw_probs_unsatisfiable, "QW unsatisfiable", errors_qw_unsatisfiable)

    plot_graph(20, av_crosson_prob, y_err=error_crosson, label='QW Crosson')

    # fit_and_plot2(n_array2, av_probs2, "Guess", None)
    # fit_and_plot2(n_array2, av_probs3, "Guess Sqrt", None)

    ax1 = ax2.twinx()
    ax1.set_ylabel(r"$\langle T_{0.99} \rangle$")
    fit_and_plot2(n_array_aqc, av_aqc_times_satisfiable, "AQC satisfiable", errors_aqc_satisfiable)
    fit_and_plot2(n_array_aqc, av_aqc_times_unsatisfiable, "AQC unsatisfiable", errors_aqc_unsatisfiable)

    scale = after_plot(fig, ax1)
    after_plot2(fig, ax2, scale)
    ax1.tick_params(direction='in', right=True, which='both')
    ax2.tick_params(direction='in', top=True, which='both')
    plt.tight_layout()
    plt.show()
    # plt.savefig('scalings_satisfiable_vs_unsatisfiable.png', dpi=200)
