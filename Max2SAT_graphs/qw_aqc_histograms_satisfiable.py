import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def adams_quantum_walk_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 2].astype(float)


def adams_adiabatic_data(n):
    '''returns time required to get 0.99 success probability'''
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


def zero_to_nan(array):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in array]


def get_satisfiable_list(n):
    data = np.genfromtxt('./../Max2SAT/m2s_satisfiable.csv', delimiter=',', skip_header=1, dtype=str)
    satisfiable_data = data[:, 1]
    m = n - 5
    return satisfiable_data[m*10000:(m+1)*10000]


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=13)

    fig = plt.figure()

    gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs.update(hspace=0.45, wspace=0.4)
    axs = list()
    axs.append(fig.add_subplot(gs[0]))
    axs.append(fig.add_subplot(gs[1]))
    axs.append(fig.add_subplot(gs[2]))
    axs.append(fig.add_subplot(gs[3]))

    n_list = [5, 15]
    num_bins = 20

    for i, n in enumerate(n_list):
        qw_probs_list = []
        aqc_times_list = []

        qw_probs = adams_quantum_walk_data(n)
        aqc_times = adams_adiabatic_data(n)
        satisfiable_list = get_satisfiable_list(n).astype(int)

        probs_satisfiable = np.delete(qw_probs, np.where(satisfiable_list == 0))
        probs_unsatisfiable = np.delete(qw_probs, np.where(satisfiable_list == 1))

        times_satisfiable = np.delete(aqc_times, np.where(satisfiable_list == 0))
        times_unsatisfiable = np.delete(aqc_times, np.where(satisfiable_list == 1))

        # delete NaN values for AQC
        times_satisfiable = np.delete(times_satisfiable, np.where(np.isnan(times_satisfiable)))
        times_unsatisfiable = np.delete(times_unsatisfiable, np.where(np.isnan(times_unsatisfiable)))
        
        min_prob = np.min((np.min(probs_satisfiable), np.min(probs_unsatisfiable)))
        max_prob = np.max((np.max(probs_satisfiable), np.max(probs_unsatisfiable)))

        min_time = np.min((np.min(times_satisfiable), np.min(times_unsatisfiable)))
        max_time = np.max((np.max(times_satisfiable), np.max(times_unsatisfiable)))

        x_qw = np.linspace(min_prob-0.00001, max_prob+0.00001, num=num_bins+1)
        x_aqc = np.linspace(min_time-0.1, max_time+0.1, num=num_bins+1)

        x_logarithmic = np.ones(num_bins+1) * min_time
        multiply_factor = 2**((np.log2(max_time)-np.log2(min_time))/num_bins)
        x_logarithmic = [x_logarithmic[i] * multiply_factor**i for i in range(len(x_logarithmic))]
        x_logarithmic[-1] += 1
        print(max_time)

        sat_average_qw = np.median(probs_satisfiable)
        unsat_average_qw = np.median(probs_unsatisfiable)

        sat_average_aqc = np.median(times_satisfiable)
        unsat_average_aqc = np.median(times_unsatisfiable)

        axs[i].hist(probs_unsatisfiable, x_qw, color='green', alpha=0.75, density=True, label='unsatisfiable')
        axs[i].hist(probs_satisfiable, x_qw, color='red', alpha=0.75, density=True, label='satisfiable')
        ylim = axs[i].get_ylim()
        axs[i].vlines(sat_average_qw, 0, ylim[1], color='black', linestyle='--')
        axs[i].vlines(unsat_average_qw, 0, ylim[1], color='black')
        axs[i].set_ylim(ylim)
        axs[i].set_xlabel(r"$\overline{P}(0, 100)$")
        axs[i].set_ylabel(r"$p(\overline{P}(0, 100))$")
        axs[i].set_xlim((min_prob, max_prob))
        # axs[i].legend()


        if i == 0:
            axs[i+2].hist(times_unsatisfiable, x_aqc, color='green', alpha=0.75, density=True, label='unsatisfiable')
            axs[i+2].hist(times_satisfiable, x_aqc, color='red', alpha=0.75, density=True, label='satisfiable')
            ylim = axs[i+2].get_ylim()
            axs[i+2].vlines(sat_average_aqc, 0, ylim[1], color='black', linestyle='--')
            axs[i+2].vlines(unsat_average_aqc, 0, ylim[1], color='black')
            axs[i+2].set_ylim(ylim)
            axs[i+2].set_xlabel(r'$T_{0.99}$')
            axs[i+2].set_ylabel(r'$p(T_{0.99})$')
            axs[i+2].set_xlim((min_time, max_time))
            # axs[i].legend()
        else:
            # ############################ LOGARITHMIC PLOT ###############################
            h, b = np.histogram(np.log10(times_unsatisfiable), bins=num_bins, density=True)
            db = b[1:]-b[:-1]
            b = (b[1:]+b[:-1])/2
            htot = np.dot(h, db)
            h = (h/htot)
            plt.bar(b, h, width=db*1.0, alpha=0.75,
                    color='green', label='unsatisfaiable')

            h, b = np.histogram(np.log10(times_satisfiable), bins=num_bins, density=True)
            db = b[1:]-b[:-1]
            b = (b[1:]+b[:-1])/2
            htot = np.dot(h, db)
            h = (h/htot)
            plt.bar(b, h, width=db*1.0, alpha=0.75,
                    color='red', label='satisfiable')
             
            ylim = axs[i+2].get_ylim()
            axs[i+2].vlines(np.log10(sat_average_aqc), 0, ylim[1], color='black', linestyle='--')
            axs[i+2].vlines(np.log10(unsat_average_aqc), 0, ylim[1], color='black')
            axs[i+2].set_ylim(ylim)
            # axs[i+2].set_xscale('log')
            axs[i+2].set_xlabel(r'$\log_{10}(T_{0.99})$')
            axs[i+2].set_ylabel(r'$p(\log_{10}(T_{0.99}))$')
            axs[i+2].set_xlim((np.log10(min_time), np.log10(max_time)))


    plt.tick_params(direction='in', top=True, right=True, which='both')
    plt.tight_layout()
    # plt.savefig('hardness_histograms_satisfiable_vs_unsatisfiable.pdf', dpi=200)
    plt.show()
