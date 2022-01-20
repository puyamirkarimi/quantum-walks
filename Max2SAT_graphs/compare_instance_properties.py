'''Plots to see if there are common properties for hard instances, and whether
those properties change for different algorithms.'''

# %%
# imports

from matplotlib import pyplot as plt
import numpy as np

# %%
# initialisations

n_array_qw = np.arange(5, 21)
n_array_aqc = np.arange(5, 16)

deciles = np.arange(10, dtype=int)

decile_colors_1 = np.zeros((10, 3))
for i in range(10):
    cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
    decile_colors_1[9-i, :] = (0.0, cn1, cn2)

decile_colors_2 = np.zeros((10, 3))
for i in range(10):
    cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
    decile_colors_2[9-i, :] = (cn1, cn2, 0.0)

plt.rc('text', usetex=True)
plt.rc('font', size=14)

# %%
# function definitions

def get_instance_energy_spreads(n):
    return np.loadtxt(f'./../energy_spreads/energy_spread_n_{n}.txt', skiprows=1)


def get_average_energy_spreads(n, indices=None):
    if indices is None:
        return np.mean(get_instance_energy_spreads(n))
    else:
        return np.mean(get_instance_energy_spreads(n)[indices])


def get_all_formulae(n):
    '''returns all instances of a given n'''
    print(f'Getting all the formulae of size n={n}')
    instance_names = get_instance_names(n)
    instances = []
    for name in instance_names:
        instances.append(get_2sat_formula(name))
    return np.array(instances)


def get_hardest_formulae_qw(n, frac, return_indices=False):
    '''returns the hardest "frac" fraction of instances for QW at a given n'''
    print(f'Getting the hardest {frac} fraction of formulae of size n={n} for QW')
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    hardest_indices = np.argsort(success_probs)[:num_instances]
    if return_indices:
        return hardest_indices
    hardest_instance_names = instance_names[hardest_indices]
    hardest_instances = []
    for name in hardest_instance_names:
        hardest_instances.append(get_2sat_formula(name))
    return np.array(hardest_instances)


def get_easiest_formulae_qw(n, frac, return_indices=False):
    '''returns the easiest "frac" fraction of instances for QW at a given n'''
    print(f'Getting the easiest {frac} fraction of formulae of size n={n} for QW')
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    easiest_indices = np.argsort(success_probs)[(10000-num_instances):]
    if return_indices:
        return easiest_indices
    easiest_instance_names = instance_names[easiest_indices]
    easiest_instances = []
    for name in easiest_instance_names:
        easiest_instances.append(get_2sat_formula(name))
    return np.array(easiest_instances)


def get_deciled_formulae_qw(n, return_indices=False):
    '''returns instances of a given n organised by QW decile'''
    print(f'Getting the formulae of size n={n} organised by QW decile')
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    indices_by_hardness = np.argsort(success_probs)
    deciled_instances = []
    for decile in range(10):
        print(f'Doing decile {decile+1}')
        deciled_instances.append([])
        end = int((10-decile) * (10000/10))
        start = int((9-decile) * (10000/10))
        indices = indices_by_hardness[start:end]
        if return_indices:
            deciled_instances[-1] = indices
        else:
            decile_instance_names = instance_names[indices]
            for name in decile_instance_names:
                deciled_instances[-1].append(get_2sat_formula(name))
    return np.array(deciled_instances)


def get_hardest_formulae_aqc(n, frac, return_indices=False):
    '''returns the hardest "frac" fraction of instances for AQC at a given n'''
    print(f'Getting the hardest {frac} fraction of formulae of size n={n} for AQC')
    durations = adams_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    hardest_indices = np.argsort(durations)[(10000-num_instances):]
    if return_indices:
        return hardest_indices
    hardest_instance_names = instance_names[hardest_indices]
    hardest_instances = []
    for name in hardest_instance_names:
        hardest_instances.append(get_2sat_formula(name))
    return np.array(hardest_instances)


def get_easiest_formulae_aqc(n, frac, return_indices=False):
    '''returns the easiest "frac" fraction of instances for AQC at a given n'''
    print(f'Getting the easiest {frac} fraction of formulae of size n={n} for AQC')
    durations = adams_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    easiest_indices = np.argsort(durations)[:num_instances]
    if return_indices:
        return easiest_indices
    easiest_instance_names = instance_names[easiest_indices]
    easiest_instances = []
    for name in easiest_instance_names:
        easiest_instances.append(get_2sat_formula(name))
    return np.array(easiest_instances)


def get_deciled_formulae_aqc(n, return_indices=False):
    '''returns instances of a given n organised by QW decile'''
    print(f'Getting the formulae of size n={n} organised by QW decile')
    durations = adams_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    indices_by_hardness = np.argsort(durations)
    deciled_instances = []
    for decile in range(10):
        print(f'Doing decile {decile+1}')
        deciled_instances.append([])
        start = int(decile * (10000/10))
        end = int((decile + 1) * (10000/10))
        indices = indices_by_hardness[start:end]
        if return_indices:
            deciled_instances[-1] = indices
        else:
            decile_instance_names = instance_names[indices]
            for name in decile_instance_names:
                deciled_instances[-1].append(get_2sat_formula(name))
    return np.array(deciled_instances)


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


# variable occurances functions

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


def variable_with_small_occurance_exists(formula, n, occurances=1):
    return np.min(variable_occurances(formula, n)) <= occurances


def fraction_of_formulae_with_small_occurance_variable(formulae, n, occurances=1):
    count = 0
    for formula in formulae:
        if variable_with_small_occurance_exists(formula, n, occurances=occurances):
            count += 1
    return count/len(formulae[:, 0, 0])


# positive vs negative literal functions

def positive_vs_negative_differences(formula, n):
    '''returns the difference between the number of times a literal is + vs - for each variable in a formula'''
    positives = np.zeros(n, dtype=np.int32)
    negatives = np.zeros(n, dtype=np.int32)
    for clause in formula:
        var_1 = clause[1]
        sign_1 = clause[0]
        var_2 = clause[3]
        sign_2 = clause[2]

        if sign_1 == 1:
            positives[var_1] += 1
        elif sign_1 == -1:
            negatives[var_1] += 1
        else:
            print('ERROR: literal sign is not +1 or -1')

        if sign_2 == 1:
            positives[var_2] += 1
        elif sign_2 == -1:
            negatives[var_2] += 1
        else:
            print('ERROR: literal sign is not +1 or -1')

    return np.abs(positives - negatives)


def positive_vs_negative_differences_many_formulae(formulae, n):
    '''returns the number of times variables occured X times for each X across many formulae'''
    differences = np.zeros((len(formulae[:, 0, 0]), n))
    for i, formula in enumerate(formulae):
        differences[i, :] = positive_vs_negative_differences(formula, n)
    return differences.flatten()


def fraction_small_differences_many_formulae(formulae, n, max_difference=1):
    '''returns the fraction of variables which have small "differences" for many formulae '''
    differences = positive_vs_negative_differences_many_formulae(formulae, n)
    return np.count_nonzero(differences <= max_difference) / len(differences)


# reused functions below

def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return instance_data[:, 0], instance_data[:, 1]


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../instances_original/" + instance_name + ".m2s")  # path of instance files in adam's format
    return out.astype(int)


def adams_quantum_walk_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1+(n-5)*10000, usecols=2, max_rows=10000, dtype=str).astype(float)


def adams_adiabatic_data(n):
    '''returns time required to get 0.99 success probability'''
    a = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', missing_values='', skip_header=1+(n-5)*10000, usecols=10, max_rows=10000, dtype=str)
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


# # %%

# n = 15
# fraction = 0.05

# all_formulae = get_all_formulae(n)
# hardest_for_qw = get_hardest_formulae_qw(n, fraction)
# hardest_for_aqc = get_hardest_formulae_aqc(n, fraction)
# crosson = get_crosson_formulae()

# # %%
# # analysis of min variable occurances for specific n

# bins = np.arange(-0.5, 9.5, step=1)
# centres = [val + 0.5 for val in bins[:-1]]
# plt.figure()
# all_hist = plt.hist(min_variable_occurances(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
# hardest_qw_hist = plt.hist(min_variable_occurances(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# # hardest_aqc_hist = plt.hist(min_variable_occurances(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
# crosson_hist = plt.hist(min_variable_occurances(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
# plt.plot(centres, all_hist, color='red')
# plt.plot(centres, hardest_qw_hist, color='forestgreen')
# # plt.plot(centres, hardest_aqc_hist, color='gold')
# plt.plot(centres, crosson_hist, color='purple')
# plt.show()

# # %%
# # analysis of max variable occurances for specific n

# bins = np.arange(5.5, 18.5, step=1)
# centres = [val + 0.5 for val in bins[:-1]]
# plt.figure()
# all_hist = plt.hist(max_variable_occurances(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
# hardest_qw_hist = plt.hist(max_variable_occurances(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# # hardest_aqc_hist = plt.hist(max_variable_occurances(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
# crosson_hist = plt.hist(max_variable_occurances(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
# plt.plot(centres, all_hist, color='red')
# plt.plot(centres, hardest_qw_hist, color='forestgreen')
# # plt.plot(centres, hardest_aqc_hist, color='gold')
# plt.plot(centres, crosson_hist, color='purple')
# plt.show()

# # %%
# # plot average distributions of variable occurances

# bins = np.arange(-0.5, (3*n)+0.5, step=1)
# centres = [val + 0.5 for val in bins[:-1]]
# all_hist = plt.hist(variable_occurances_many_formulae(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
# hardest_qw_hist = plt.hist(variable_occurances_many_formulae(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# # hardest_aqc_hist = plt.hist(variable_occurances_many_formulae(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
# crosson_hist = plt.hist(variable_occurances_many_formulae(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
# plt.plot(centres, all_hist, color='red')
# plt.plot(centres, hardest_qw_hist, color='forestgreen')
# # plt.plot(centres, hardest_aqc_hist, color='gold')
# plt.plot(centres, crosson_hist, color='purple')
# plt.xlim([0, 20])
# plt.show()

# # %%

# n = 20
# qw_deciled_formulae = get_deciled_formulae_qw(n)
# aqc_deciled_formulae = get_deciled_formulae_aqc(n)

# %%
# get deciled & easiest/hardest formulae for QW and AQC

frac = 0.05

# QW
qw_deciled_formulae = []    # list of arrays of formulae
qw_hardest_formulae = []    # list of formulae
qw_easiest_formulae = []    # list of formulae
for n in n_array_qw:
    qw_deciled_formulae.append(get_deciled_formulae_qw(n))
    qw_hardest_formulae.append(get_hardest_formulae_qw(n, frac))
    qw_easiest_formulae.append(get_easiest_formulae_qw(n, frac))

# AQC
aqc_deciled_formulae = []    # list of arrays of formulae
aqc_hardest_formulae = []    # list of formulae
aqc_easiest_formulae = []    # list of formulae
for n in n_array_aqc:
    aqc_deciled_formulae.append(get_deciled_formulae_aqc(n))
    aqc_hardest_formulae.append(get_hardest_formulae_aqc(n, frac))
    aqc_easiest_formulae.append(get_easiest_formulae_aqc(n, frac))

# %%
# Get crosson formulae

crosson_formulae = get_crosson_formulae()

# %%
# get deciled small occurance variable data for QW, AQC and Crosson

# QW
small_occurance_fraction_qw_deciles = np.zeros((10, len(n_array_qw)))
small_occurance_fraction_qw_hardest = np.zeros(len(n_array_qw))
small_occurance_fraction_qw_easiest = np.zeros(len(n_array_qw))
for i, n in enumerate(n_array_qw):
    for decile in deciles:
        small_occurance_fraction_qw_deciles[decile, i] = fraction_of_formulae_with_small_occurance_variable(qw_deciled_formulae[i][decile], n)
    small_occurance_fraction_qw_hardest[i] = fraction_of_formulae_with_small_occurance_variable(qw_hardest_formulae[i], n)
    small_occurance_fraction_qw_easiest[i] = fraction_of_formulae_with_small_occurance_variable(qw_easiest_formulae[i], n)

# AQC
small_occurance_fraction_aqc_deciles = np.zeros((10, len(n_array_aqc)))
small_occurance_fraction_aqc_hardest = np.zeros(len(n_array_aqc))
small_occurance_fraction_aqc_easiest = np.zeros(len(n_array_aqc))
for i, n in enumerate(n_array_aqc):
    for decile in deciles:
        small_occurance_fraction_aqc_deciles[decile, i] = fraction_of_formulae_with_small_occurance_variable(aqc_deciled_formulae[i][decile], n)
    small_occurance_fraction_aqc_hardest[i] = fraction_of_formulae_with_small_occurance_variable(aqc_hardest_formulae[i], n)
    small_occurance_fraction_aqc_easiest[i] = fraction_of_formulae_with_small_occurance_variable(aqc_easiest_formulae[i], n)

# Crosson
small_occurance_fraction_crosson = fraction_of_formulae_with_small_occurance_variable(crosson_formulae, 20)

# %%
# plot small occurance graphs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for decile in deciles:
    ax1.plot(n_array_qw, small_occurance_fraction_qw_deciles[decile, :], color=decile_colors_1[decile])
ax1.plot(n_array_qw, small_occurance_fraction_qw_hardest, color='blue')
ax1.plot(n_array_qw, small_occurance_fraction_qw_easiest, color='gold')
ax1.scatter(20, small_occurance_fraction_crosson, color='red')
ax1.set_xlabel(r'$n$')
ax1.set_ylabel(r'Fraction of instances with a\\variable that only shows up once')

for decile in deciles:
    ax2.plot(n_array_aqc, small_occurance_fraction_aqc_deciles[decile, :], color=decile_colors_1[decile])
ax2.plot(n_array_aqc, small_occurance_fraction_aqc_hardest, color='blue')
ax2.plot(n_array_aqc, small_occurance_fraction_aqc_easiest, color='gold')
ax2.scatter(20, small_occurance_fraction_crosson, color='red')
ax2.set_xlabel(r'n')
# ax2.set_ylabel(r'Fraction of instances with a\\variable that only shows up once')
plt.tight_layout()
plt.savefig('single_occurance_variable_fraction.pdf', dpi=200)
plt.show()

# # %%
# # plot average distributions of positive vs negative differences

# bins = np.arange(-0.5, (3*n)+0.5, step=1)
# centres = [val + 0.5 for val in bins[:-1]]
# all_hist = plt.hist(positive_vs_negative_differences_many_formulae(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
# hardest_qw_hist = plt.hist(positive_vs_negative_differences_many_formulae(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
# hardest_aqc_hist = plt.hist(positive_vs_negative_differences_many_formulae(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
# # crosson_hist = plt.hist(positive_vs_negative_differences_many_formulae(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
# plt.plot(centres, all_hist, color='red')
# plt.plot(centres, hardest_qw_hist, color='forestgreen')
# plt.plot(centres, hardest_aqc_hist, color='gold')
# # plt.plot(centres, crosson_hist, color='purple')
# plt.xlim([0, 20])
# plt.show()

# %%
# get deciled small differences data for QW, AQC and Crosson

max_difference = 1

# QW
small_differences_fraction_qw_deciles = np.zeros((10, len(n_array_qw)))
small_differences_fraction_qw_hardest = np.zeros(len(n_array_qw))
small_differences_fraction_qw_easiest = np.zeros(len(n_array_qw))
for i, n in enumerate(n_array_qw):
    for decile in deciles:
        small_differences_fraction_qw_deciles[decile, i] = fraction_small_differences_many_formulae(qw_deciled_formulae[i][decile], n, max_difference)
    small_differences_fraction_qw_hardest[i] = fraction_small_differences_many_formulae(qw_hardest_formulae[i], n, max_difference)
    small_differences_fraction_qw_easiest[i] = fraction_small_differences_many_formulae(qw_easiest_formulae[i], n, max_difference)

# AQC
small_differences_fraction_aqc_deciles = np.zeros((10, len(n_array_aqc)))
small_differences_fraction_aqc_hardest = np.zeros(len(n_array_aqc))
small_differences_fraction_aqc_easiest = np.zeros(len(n_array_aqc))
for i, n in enumerate(n_array_aqc):
    for decile in deciles:
        small_differences_fraction_aqc_deciles[decile, i] = fraction_small_differences_many_formulae(aqc_deciled_formulae[i][decile], n, max_difference)
    small_differences_fraction_aqc_hardest[i] = fraction_small_differences_many_formulae(aqc_hardest_formulae[i], n, max_difference)
    small_differences_fraction_aqc_easiest[i] = fraction_small_differences_many_formulae(aqc_easiest_formulae[i], n, max_difference)

# Crosson
small_differences_fraction_crosson = fraction_small_differences_many_formulae(crosson_formulae, 20, max_difference)

# %%
# plot small differences graphs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for decile in deciles:
    ax1.plot(n_array_qw, small_differences_fraction_qw_deciles[decile, :], color=decile_colors_1[decile])
ax1.plot(n_array_qw, small_differences_fraction_qw_hardest, color='blue')
ax1.plot(n_array_qw, small_differences_fraction_qw_easiest, color='gold')
ax1.scatter(20, small_differences_fraction_crosson, color='red')
ax1.set_xlabel(r'$n$')
ax1.set_ylabel(r"Fraction of variables which are\\'balanced' up to a difference of 1")

for decile in deciles:
    ax2.plot(n_array_aqc, small_differences_fraction_aqc_deciles[decile, :], color=decile_colors_1[decile])
ax2.plot(n_array_aqc, small_differences_fraction_aqc_hardest, color='blue')
ax2.plot(n_array_aqc, small_differences_fraction_aqc_easiest, color='gold')
ax2.scatter(20, small_differences_fraction_crosson, color='red')
ax2.set_xlabel(r'$n$')
plt.tight_layout()
plt.savefig('balanced_variable_fraction.pdf', dpi=200)
plt.show()


# %%
# get deciled & easiest/hardest formulae *indices* for QW and AQC

frac = 0.05

# QW
qw_deciled_formulae_indices = []
qw_hardest_formulae_indices = []
qw_easiest_formulae_indices = []
for n in n_array_qw:
    qw_deciled_formulae_indices.append(get_deciled_formulae_qw(n, return_indices=True))
    qw_hardest_formulae_indices.append(get_hardest_formulae_qw(n, frac, return_indices=True))
    qw_easiest_formulae_indices.append(get_easiest_formulae_qw(n, frac, return_indices=True))

# AQC
aqc_deciled_formulae_indices = []
aqc_hardest_formulae_indices = []
aqc_easiest_formulae_indices = []
for n in n_array_aqc:
    aqc_deciled_formulae_indices.append(get_deciled_formulae_aqc(n, return_indices=True))
    aqc_hardest_formulae_indices.append(get_hardest_formulae_aqc(n, frac, return_indices=True))
    aqc_easiest_formulae_indices.append(get_easiest_formulae_aqc(n, frac, return_indices=True))

# %%

n_array_energy_spreads = np.arange(5, 12)

# QW
average_energy_spreads_qw_deciles = np.zeros((10, len(n_array_energy_spreads)))
average_energy_spreads_qw_hardest = np.zeros(len(n_array_energy_spreads))
average_energy_spreads_qw_easiest = np.zeros(len(n_array_energy_spreads))
for i, n in enumerate(n_array_energy_spreads):
    for decile in deciles:
        average_energy_spreads_qw_deciles[decile, i] = get_average_energy_spreads(n, qw_deciled_formulae_indices[i][decile])
    average_energy_spreads_qw_hardest[i] = get_average_energy_spreads(n, qw_hardest_formulae_indices[i])
    average_energy_spreads_qw_easiest[i] = get_average_energy_spreads(n, qw_easiest_formulae_indices[i])

# AQC
average_energy_spreads_aqc_deciles = np.zeros((10, len(n_array_energy_spreads)))
average_energy_spreads_aqc_hardest = np.zeros(len(n_array_energy_spreads))
average_energy_spreads_aqc_easiest = np.zeros(len(n_array_energy_spreads))
for i, n in enumerate(n_array_energy_spreads):
    for decile in deciles:
        average_energy_spreads_aqc_deciles[decile, i] = get_average_energy_spreads(n, aqc_deciled_formulae_indices[i][decile])
    average_energy_spreads_aqc_hardest[i] = get_average_energy_spreads(n, aqc_hardest_formulae_indices[i])
    average_energy_spreads_aqc_easiest[i] = get_average_energy_spreads(n, aqc_easiest_formulae_indices[i])

# %%
# plot energy spreads graphs

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
for decile in deciles:
    ax1.plot(n_array_energy_spreads, average_energy_spreads_qw_deciles[decile, :], color=decile_colors_1[decile])
ax1.plot(n_array_energy_spreads, average_energy_spreads_qw_hardest, color='blue')
ax1.plot(n_array_energy_spreads, average_energy_spreads_qw_easiest, color='gold')
ax1.set_xlabel(r'$n$')
ax1.set_ylabel(r'$\left\langle E^{(P)}_{N-1} - E^{(P)}_0 \right\rangle$')

for decile in deciles:
    ax2.plot(n_array_energy_spreads, average_energy_spreads_aqc_deciles[decile, :], color=decile_colors_1[decile])
ax2.plot(n_array_energy_spreads, average_energy_spreads_aqc_hardest, color='blue')
ax2.plot(n_array_energy_spreads, average_energy_spreads_aqc_easiest, color='gold')
ax2.set_xlabel(r'$n$')
plt.tight_layout()
plt.savefig('instance_energy_spread.pdf', dpi=200)
plt.show()