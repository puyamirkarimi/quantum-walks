'''Plots to see if there are common properties for hard instances, and whether
those properties change for different algorithms.'''

# %%
# imports

from matplotlib import pyplot as plt
import numpy as np

# %%
# new functions

def get_all_formulae(n):
    '''returns all instances of a given n'''
    print(f'Getting all the formulae of size n={n}')
    instance_names = get_instance_names(n)
    instances = []
    for name in instance_names:
        instances.append(get_2sat_formula(name))
    return np.array(instances)


def get_hardest_formulae_qw(n, frac):
    '''returns the hardest "frac" fraction of instances for QW at a given n'''
    print(f'Getting the hardest {frac} fraction of formulae of size n={n} for QW')
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    hardest_indices = np.argsort(success_probs)[:num_instances]
    hardest_instance_names = instance_names[hardest_indices]
    hardest_instances = []
    for name in hardest_instance_names:
        hardest_instances.append(get_2sat_formula(name))
    return np.array(hardest_instances)


def get_hardest_formulae_aqc(n, frac):
    '''returns the hardest "frac" fraction of instances for AQC at a given n'''
    print(f'Getting the hardest {frac} fraction of formulae of size n={n} for AQC')
    durations = adams_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    hardest_indices = np.argsort(durations)[(10000-num_instances):]
    hardest_instance_names = instance_names[hardest_indices]
    hardest_instances = []
    for name in hardest_instance_names:
        hardest_instances.append(get_2sat_formula(name))
    return np.array(hardest_instances)


def get_crosson_formulae():
    '''returns the instances that were data-mined to be hard for QA by Crosson'''
    instances = []
    for i in range(137):
        instance_name = str(i)
        instances.append(np.loadtxt("./../../instances_crosson/" + instance_name + ".m2s").astype(int))
    return np.array(instances)


def num_satisfiable():
    '''given a set of instances, returns the number which are satisfiable'''
    return None


def single_instance_variable_negations():
    '''returns the number of times each variable has been negated'''
    return None


def get_instance_names(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 0].astype(str)


def variable_occurances(formula, n):
    '''returns the number of occurances of each variable in a formula'''
    occurances = np.zeros(n, dtype=np.int32)
    for clause in formula:
        var_1 = clause[1]
        var_2 = clause[3]
        occurances[var_1] += 1
        occurances[var_2] += 1
    return occurances


def variable_occurances_many_formulae(formulae, n):
    '''returns the number of times variables occured X times for each X across many formulae'''
    # occurances_hist = np.zeros(3*n+1, dtype=np.int32)
    occurances = np.zeros((len(formulae[:, 0, 0]), n))
    for i, formula in enumerate(formulae):
        occurances[i, :] = variable_occurances(formula, n)
    return occurances.flatten()


def min_variable_occurance(formula, n):
    return np.min(variable_occurances(formula, n))


def min_variable_occurances(formulae, n):
    return [min_variable_occurance(formula, n) for formula in formulae]


def average_min_variable_occurance(formulae, n):
    return np.mean(min_variable_occurances(formulae, n))


def max_variable_occurance(formula, n):
    return np.max(variable_occurances(formula, n))


def max_variable_occurances(formulae, n):
    return [max_variable_occurance(formula, n) for formula in formulae]


def average_max_variable_occurance(formulae, n):
    return np.mean(max_variable_occurances(formulae, n))


# reused functions below

def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return instance_data[:, 0], instance_data[:, 1]


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../instances_original/" + instance_name + ".m2s")  # path of instance files in adam's format
    return out.astype(int)


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


# %%

n = 20
fraction = 0.05

all_formulae = get_all_formulae(n)
hardest_for_qw = get_hardest_formulae_qw(n, fraction)
# hardest_for_aqc = get_hardest_formulae_aqc(n, fraction)
crosson = get_crosson_formulae()


# %%
# analysis for min variable occurances

bins = np.arange(-0.5, 9.5, step=1)
centres = [val + 0.5 for val in bins[:-1]]
plt.figure()
all_hist = plt.hist(min_variable_occurances(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
hardest_qw_hist = plt.hist(min_variable_occurances(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# hardest_aqc_hist = plt.hist(min_variable_occurances(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
crosson_hist = plt.hist(min_variable_occurances(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
plt.plot(centres, all_hist, color='red')
plt.plot(centres, hardest_qw_hist, color='forestgreen')
# plt.plot(centres, hardest_aqc_hist, color='gold')
plt.plot(centres, crosson_hist, color='purple')
plt.show()

# %%
# analysis for max variable occurances

bins = np.arange(5.5, 18.5, step=1)
centres = [val + 0.5 for val in bins[:-1]]
plt.figure()
all_hist = plt.hist(max_variable_occurances(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
hardest_qw_hist = plt.hist(max_variable_occurances(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# hardest_aqc_hist = plt.hist(max_variable_occurances(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
crosson_hist = plt.hist(max_variable_occurances(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
plt.plot(centres, all_hist, color='red')
plt.plot(centres, hardest_qw_hist, color='forestgreen')
# plt.plot(centres, hardest_aqc_hist, color='gold')
plt.plot(centres, crosson_hist, color='purple')
plt.show()

# %%
# plot average distributions of variable occurances

bins = np.arange(-0.5, (3*n)+0.5, step=1)
centres = [val + 0.5 for val in bins[:-1]]
all_hist = plt.hist(variable_occurances_many_formulae(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
hardest_qw_hist = plt.hist(variable_occurances_many_formulae(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# hardest_aqc_hist = plt.hist(variable_occurances_many_formulae(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
crosson_hist = plt.hist(variable_occurances_many_formulae(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
plt.plot(centres, all_hist, color='red')
plt.plot(centres, hardest_qw_hist, color='forestgreen')
# plt.plot(centres, hardest_aqc_hist, color='gold')
plt.plot(centres, crosson_hist, color='purple')
plt.xlim([0, 20])
plt.show()
