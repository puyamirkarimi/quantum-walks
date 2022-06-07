# %%
# imports

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import matplotlib.gridspec as gridspec

# %%
# initialisations

plt.rc('text', usetex=True)
plt.rc('font', size=14)

n_array_qw = np.arange(5, 21)
n_array_aqc = np.arange(5, 16)

line = lambda x,m,c: (m*x)+c

# %%
# function definitions

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


def nan_to_largest(x, addition=1):
    max = np.max(x[np.logical_not(np.isnan(x))])
    return np.nan_to_num(x, nan=max+addition)


def nan_to_pos_inf(x):
    return np.nan_to_num(x, nan=float('inf'))


def remove_nan(x):
    return x[np.logical_not(np.isnan(x))]


# %%
# get average QW probabilities and AQC durations

mean_qw_probs_satisfiable = np.zeros(len(n_array_qw))
mean_qw_probs_unsatisfiable = np.zeros(len(n_array_qw))
errors_qw_satisfiable = np.zeros(len(n_array_qw))
errors_qw_unsatisfiable = np.zeros(len(n_array_qw))
median_qw_probs_satisfiable = np.zeros(len(n_array_qw))
median_qw_probs_unsatisfiable = np.zeros(len(n_array_qw))

mean_aqc_durations_satisfiable = np.zeros(len(n_array_aqc))
mean_aqc_durations_unsatisfiable = np.zeros(len(n_array_aqc))
errors_aqc_satisfiable = np.zeros(len(n_array_aqc))
errors_aqc_unsatisfiable = np.zeros(len(n_array_aqc))
median_aqc_durations_satisfiable = np.zeros(len(n_array_aqc))
median_aqc_durations_unsatisfiable = np.zeros(len(n_array_aqc))

crosson_qw_probs = adams_quantum_walk_data_crosson()
mean_crosson_qw_prob = np.mean(crosson_qw_probs)
error_crosson_qw = np.std(crosson_qw_probs, ddof=1) / np.sqrt(len(crosson_qw_probs))
median_crosson_qw_prob = np.median(crosson_qw_probs)

for i, n in enumerate(n_array_qw):
    qw_probs = adams_quantum_walk_data(n)
    satisfiable_list = get_satisfiable_list(n).astype(int)

    qw_probs_satisfiable = np.delete(qw_probs, np.where(satisfiable_list == 0))
    qw_probs_unsatisfiable = np.delete(qw_probs, np.where(satisfiable_list == 1))

    num_satisfiable = len(qw_probs_satisfiable)
    num_unsatisfiable = len(qw_probs_unsatisfiable)

    print(f'Number of satisfiable instances for n={n} is {num_satisfiable} (out of 10000)')

    mean_qw_probs_satisfiable[i] = np.mean(qw_probs_satisfiable)
    mean_qw_probs_unsatisfiable[i] = np.mean(qw_probs_unsatisfiable)

    median_qw_probs_satisfiable[i] = np.median(qw_probs_satisfiable)
    median_qw_probs_unsatisfiable[i] = np.median(qw_probs_unsatisfiable)

    errors_qw_satisfiable[i] = np.std(qw_probs_satisfiable, ddof=1) / np.sqrt(len(qw_probs_satisfiable))
    errors_qw_unsatisfiable[i] = np.std(qw_probs_unsatisfiable, ddof=1) / np.sqrt(len(qw_probs_unsatisfiable))

for i, n in enumerate(n_array_aqc):
    aqc_durations = adams_adiabatic_data(n)
    satisfiable_list = get_satisfiable_list(n).astype(int)
    
    aqc_durations_satisfiable = np.delete(aqc_durations, np.where(satisfiable_list == 0))
    aqc_durations_unsatisfiable = np.delete(aqc_durations, np.where(satisfiable_list == 1))
    
    num_satisfiable = len(aqc_durations_satisfiable)
    num_unsatisfiable = len(aqc_durations_unsatisfiable)
    
    mean_aqc_durations_satisfiable[i] = np.mean(remove_nan(aqc_durations_satisfiable))
    mean_aqc_durations_unsatisfiable[i] = np.mean(remove_nan(aqc_durations_unsatisfiable))

    median_aqc_durations_satisfiable[i] = np.median(nan_to_largest(aqc_durations_satisfiable))
    median_aqc_durations_unsatisfiable[i] = np.median(nan_to_largest(aqc_durations_unsatisfiable))

    errors_aqc_satisfiable[i] = np.std(aqc_durations_satisfiable, ddof=1) / np.sqrt(len(aqc_durations_satisfiable))
    errors_aqc_unsatisfiable[i] = np.std(aqc_durations_unsatisfiable, ddof=1) / np.sqrt(len(aqc_durations_unsatisfiable))

# %%
# plot scaling graphs on log-linear and log-log
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(6, 4.75))

m_size = 15
m_size_res = 35

gs1 = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
# gs1.update(hspace=0.25)

# plot QW scaling log-linear
ax = plt.subplot(gs1[0])
par,cov=optimize.curve_fit(line, n_array_qw, np.log2(median_qw_probs_satisfiable))
m_satisfiable=par[0],np.sqrt(cov[0,0])
c_satisfiable=par[1],np.sqrt(cov[1,1])
par,cov=optimize.curve_fit(line, n_array_qw, np.log2(median_qw_probs_unsatisfiable))
m_unsatisfiable=par[0],np.sqrt(cov[0,0])
c_unsatisfiable=par[1],np.sqrt(cov[1,1])
plt.scatter(n_array_qw, np.log2(median_qw_probs_satisfiable), color='red', s=m_size)
plt.scatter(n_array_qw, np.log2(median_qw_probs_unsatisfiable), color='green', s=m_size)
plt.plot(n_array_qw, line(n_array_qw, m_satisfiable[0], c_satisfiable[0]), color='red')
plt.plot(n_array_qw, line(n_array_qw, m_unsatisfiable[0], c_unsatisfiable[0]), color='green')
x_ticks = np.arange(5, 25, 5)
x_tick_labels = ['${}$'.format(x) for x in x_ticks]
plt.xticks(x_ticks, x_tick_labels)
y_ticks = np.arange(-9, 0, 2)
y_tick_labels = ['$2^{'+'{}'.format(y) +'}$' for y in y_ticks]
plt.yticks(y_ticks, y_tick_labels)
# ax.set_xlabel(r'$n$', fontsize=15)
ax.set_ylabel(r"$\mathrm{median}(\overline{P}(0, 100))$", fontsize=15)
ax.tick_params(axis='both', labelsize=13)
ax.set_ylim((-9.378, -0.821))

# plot QW scaling log-log
ax = plt.subplot(gs1[1])
par,cov=optimize.curve_fit(line, np.log10(n_array_qw), np.log2(median_qw_probs_satisfiable))
m_satisfiable=par[0],np.sqrt(cov[0,0])
c_satisfiable=par[1],np.sqrt(cov[1,1])
par,cov=optimize.curve_fit(line, np.log10(n_array_qw), np.log2(median_qw_probs_unsatisfiable))
m_unsatisfiable=par[0],np.sqrt(cov[0,0])
c_unsatisfiable=par[1],np.sqrt(cov[1,1])
plt.scatter(np.log10(n_array_qw), np.log2(median_qw_probs_satisfiable), color='red', s=m_size)
plt.scatter(np.log10(n_array_qw), np.log2(median_qw_probs_unsatisfiable), color='green', s=m_size)
plt.plot(np.log10(n_array_qw), line(np.log10(n_array_qw), m_satisfiable[0], c_satisfiable[0]), color='red')
plt.plot(np.log10(n_array_qw), line(np.log10(n_array_qw), m_unsatisfiable[0], c_unsatisfiable[0]), color='green')
x_tick_labels = ['${}$'.format(x) for x in x_ticks]
x_ticks = np.log10(np.arange(5, 25, 5))
plt.xticks(x_ticks, x_tick_labels)
# y_ticks = np.arange(-9, 0, 2)
# y_tick_labels = ['$2^{'+'{}'.format(y) +'}$' for y in y_ticks]
plt.yticks([], [])
# ax.set_xlabel(r'$n$', fontsize=15)
# ax.set_ylabel(r"$\mathrm{median}(\overline{P}(0, 100))$", fontsize=15)
ax.tick_params(axis='both', labelsize=13)
ax.set_ylim((-9.378, -0.821))

# plot AQC scaling log-linear
ax = plt.subplot(gs1[2])
par,cov=optimize.curve_fit(line, n_array_aqc, np.log2(median_aqc_durations_satisfiable))
m_satisfiable=par[0],np.sqrt(cov[0,0])
c_satisfiable=par[1],np.sqrt(cov[1,1])
par,cov=optimize.curve_fit(line, n_array_aqc, np.log2(median_aqc_durations_unsatisfiable))
m_unsatisfiable=par[0],np.sqrt(cov[0,0])
c_unsatisfiable=par[1],np.sqrt(cov[1,1])
plt.scatter(n_array_aqc, np.log2(median_aqc_durations_satisfiable), color='red', s=m_size)
plt.scatter(n_array_aqc, np.log2(median_aqc_durations_unsatisfiable), color='green', s=m_size)
plt.plot(n_array_aqc, line(n_array_aqc, m_satisfiable[0], c_satisfiable[0]), color='red')
plt.plot(n_array_aqc, line(n_array_aqc, m_unsatisfiable[0], c_unsatisfiable[0]), color='green')
x_ticks = np.arange(5, 20, 5)
x_tick_labels = ['${}$'.format(n) for n in np.arange(5, 20, 5)]
plt.xticks(x_ticks, x_tick_labels)
y_ticks = np.arange(5, 7, 1)
y_tick_labels = ['$2^{'+'{}'.format(y) +'}$' for y in y_ticks]
plt.yticks(y_ticks, y_tick_labels)
ax.set_ylabel(r"$\mathrm{median}(T_{0.99})$", fontsize=15)
ax.tick_params(axis='both', labelsize=13)
# plt.xticklabels([])
ax.set_ylim((4.345, 6.563))

# residuals
divider = make_axes_locatable(ax)
ax2 = divider.append_axes("bottom", size="40%", pad=0)
ax.figure.add_axes(ax2)
x_limits = (4.5, 15.5)
ax2.hlines(0, x_limits[0], x_limits[1], color='black', linestyle='--')
ax2.set_xlim(x_limits)
residuals_satisfiable = np.log2(median_aqc_durations_satisfiable) - line(n_array_aqc, m_satisfiable[0], c_satisfiable[0])
residuals_unsatisfiable = np.log2(median_aqc_durations_unsatisfiable) - line(n_array_aqc, m_unsatisfiable[0], c_unsatisfiable[0])
# frame2.set_ylim([,])
ax2.scatter(n_array_aqc, residuals_satisfiable, color='red', marker='+', s=m_size_res)
ax2.scatter(n_array_aqc, residuals_unsatisfiable, color='green', marker='+', s=m_size_res)
ax2.plot(n_array_aqc, residuals_satisfiable, color='red', linestyle='--')
ax2.plot(n_array_aqc, residuals_unsatisfiable, color='green', linestyle='--')
ax2.set_xlabel(r'$n$', fontsize=15)
ax2.set_ylabel(r"$\mathrm{Residual}$", fontsize=15)
ax2.tick_params(axis='both', labelsize=13)
x_ticks = np.arange(5, 20, 5)
x_tick_labels = ['${}$'.format(n) for n in np.arange(5, 20, 5)]
plt.xticks(x_ticks, x_tick_labels)
res_ylim = 0.085
ax2.set_ylim((-res_ylim, res_ylim))
ax2.set_yticks([-0.05, 0.05])

# plot AQC scaling log-log
ax = plt.subplot(gs1[3])
par,cov=optimize.curve_fit(line, np.log10(n_array_aqc), np.log2(median_aqc_durations_satisfiable))
m_satisfiable=par[0],np.sqrt(cov[0,0])
c_satisfiable=par[1],np.sqrt(cov[1,1])
par,cov=optimize.curve_fit(line, np.log10(n_array_aqc), np.log2(median_aqc_durations_unsatisfiable))
m_unsatisfiable=par[0],np.sqrt(cov[0,0])
c_unsatisfiable=par[1],np.sqrt(cov[1,1])
plt.scatter(np.log10(n_array_aqc), np.log2(median_aqc_durations_satisfiable), color='red', s=m_size)
plt.scatter(np.log10(n_array_aqc), np.log2(median_aqc_durations_unsatisfiable), color='green', s=m_size)
plt.plot(np.log10(n_array_aqc), line(np.log10(n_array_aqc), m_satisfiable[0], c_satisfiable[0]), color='red')
plt.plot(np.log10(n_array_aqc), line(np.log10(n_array_aqc), m_unsatisfiable[0], c_unsatisfiable[0]), color='green')
x_ticks = np.log10(np.arange(5, 20, 5))
x_tick_labels = ['${}$'.format(n) for n in np.arange(5, 20, 5)]
plt.xticks(x_ticks, x_tick_labels)
# y_ticks = np.arange(5, 7, 1)
# y_tick_labels = ['$2^{'+'{}'.format(y) +'}$' for y in y_ticks]
plt.yticks([], [])
ax.set_xlabel(r'$n$', fontsize=15)
# ax.set_ylabel(r"$\mathrm{median}(T_{0.99})$", fontsize=15)
ax.tick_params(axis='both', labelsize=13)
# plt.xticklabels([])
ax.set_ylim((4.345, 6.563))

# residuals
divider = make_axes_locatable(ax)
ax2 = divider.append_axes("bottom", size="40%", pad=0)
ax.figure.add_axes(ax2)
x_limits = (0.6751, 1.2)
ax2.hlines(0, x_limits[0], x_limits[1], color='black', linestyle='--')
ax2.set_xlim(x_limits)
residuals_satisfiable = np.log2(median_aqc_durations_satisfiable) - line(np.log10(n_array_aqc), m_satisfiable[0], c_satisfiable[0])
residuals_unsatisfiable = np.log2(median_aqc_durations_unsatisfiable) - line(np.log10(n_array_aqc), m_unsatisfiable[0], c_unsatisfiable[0])
# frame2.set_ylim([,])
ax2.scatter(np.log10(n_array_aqc), residuals_satisfiable, color='red', marker='+', s=m_size_res)
ax2.scatter(np.log10(n_array_aqc), residuals_unsatisfiable, color='green', marker='+', s=m_size_res)
ax2.plot(np.log10(n_array_aqc), residuals_satisfiable, color='red', linestyle='--')
ax2.plot(np.log10(n_array_aqc), residuals_unsatisfiable, color='green', linestyle='--')
ax2.set_xlabel(r'$n$', fontsize=15)
# ax2.set_ylabel(r"$\mathrm{Residual}$", fontsize=15)
ax2.tick_params(axis='both', labelsize=13)
x_ticks = np.log10(np.arange(5, 20, 5))
x_tick_labels = ['${}$'.format(n) for n in np.arange(5, 20, 5)]
plt.xticks(x_ticks, x_tick_labels)
ax2.set_ylim((-res_ylim, res_ylim))
ax2.set_yticks([])

fig.tight_layout()
# plt.savefig('scalings_satisfiable_vs_unsatisfiable_windows.pdf', dpi=200)
plt.show()
