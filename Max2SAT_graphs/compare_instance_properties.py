'''Plots to see if there are common properties for hard instances, and whether
those properties change for different algorithms.'''

# %%
# imports

from matplotlib import pyplot as plt
import numpy as np

# %%
# function definitions

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


def get_easiest_formulae_qw(n, frac):
    '''returns the easiest "frac" fraction of instances for QW at a given n'''
    print(f'Getting the easiest {frac} fraction of formulae of size n={n} for QW')
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    easiest_indices = np.argsort(success_probs)[(10000-num_instances):]
    easiest_instance_names = instance_names[easiest_indices]
    easiest_instances = []
    for name in easiest_instance_names:
        easiest_instances.append(get_2sat_formula(name))
    return np.array(easiest_instances)


def get_deciled_formulae_qw(n):
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
        decile_instance_names = instance_names[indices]
        for name in decile_instance_names:
            deciled_instances[-1].append(get_2sat_formula(name))
    return np.array(deciled_instances)


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


def get_easiest_formulae_aqc(n, frac):
    '''returns the easiest "frac" fraction of instances for AQC at a given n'''
    print(f'Getting the easiest {frac} fraction of formulae of size n={n} for AQC')
    durations = adams_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    num_instances = int(frac * 10000)
    easiest_indices = np.argsort(durations)[:num_instances]
    easiest_instance_names = instance_names[easiest_indices]
    easiest_instances = []
    for name in easiest_instance_names:
        easiest_instances.append(get_2sat_formula(name))
    return np.array(easiest_instances)


def get_deciled_formulae_aqc(n):
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

n = 15
fraction = 0.02

all_formulae = get_all_formulae(n)
hardest_for_qw = get_hardest_formulae_qw(n, fraction)
hardest_for_aqc = get_hardest_formulae_aqc(n, fraction)
crosson = get_crosson_formulae()


# %%
# analysis of min variable occurances for specific n

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
# analysis of max variable occurances for specific n

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

# %%

n = 20
qw_deciled_formulae = get_deciled_formulae_qw(n)
aqc_deciled_formulae = get_deciled_formulae_aqc(n)

# %%
# get qw deciled small occurance variable data

deciles = np.arange(10, dtype=int)
n_array_qw = np.arange(5, 21)

small_occurance_fraction_qw = np.zeros((10, len(n_array_qw)))

frac = 0.05
small_occurance_fraction_qw_hardest = np.zeros(len(n_array_qw))
small_occurance_fraction_qw_easiest = np.zeros(len(n_array_qw))

for i, n in enumerate(n_array_qw):
    qw_deciled_formulae = get_deciled_formulae_qw(n)
    for decile in deciles:
        small_occurance_fraction_qw[decile, i] = fraction_of_formulae_with_small_occurance_variable(qw_deciled_formulae[decile], n)
    qw_hardest_formulae = get_hardest_formulae_qw(n, frac)
    qw_easiest_formulae = get_easiest_formulae_qw(n, frac)
    small_occurance_fraction_qw_hardest[i] = fraction_of_formulae_with_small_occurance_variable(qw_hardest_formulae, n)
    small_occurance_fraction_qw_easiest[i] = fraction_of_formulae_with_small_occurance_variable(qw_easiest_formulae, n)


# %%
# get aqc deciled small occurance variable data

deciles = np.arange(10, dtype=int)
n_array_aqc = np.arange(5, 16)

small_occurance_fraction_aqc = np.zeros((10, len(n_array_aqc)))

frac = 0.05
small_occurance_fraction_aqc_hardest = np.zeros(len(n_array_aqc))
small_occurance_fraction_aqc_easiest = np.zeros(len(n_array_aqc))

for i, n in enumerate(n_array_aqc):
    aqc_deciled_formulae = get_deciled_formulae_aqc(n)
    for decile in deciles:
        small_occurance_fraction_aqc[decile, i] = fraction_of_formulae_with_small_occurance_variable(aqc_deciled_formulae[decile], n)
    aqc_hardest_formulae = get_hardest_formulae_aqc(n, frac)
    aqc_easiest_formulae = get_easiest_formulae_aqc(n, frac)
    small_occurance_fraction_aqc_hardest[i] = fraction_of_formulae_with_small_occurance_variable(aqc_hardest_formulae, n)
    small_occurance_fraction_aqc_easiest[i] = fraction_of_formulae_with_small_occurance_variable(aqc_easiest_formulae, n)

# %%

small_occurance_fraction_crosson = fraction_of_formulae_with_small_occurance_variable(crosson, 20)

# %%

decile_colors_1 = []
for i in range(10):
    cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
    decile_colors_1.append((0.0, cn1, cn2))

decile_colors_2 = []
for i in range(10):
    cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
    decile_colors_2.append((cn1, cn2, 0.0))

plt.figure()
for decile in deciles:
    plt.plot(n_array_qw, small_occurance_fraction_qw[decile, :], color=decile_colors_1[decile])
plt.plot(n_array_qw, small_occurance_fraction_qw_hardest, color='gold')
plt.plot(n_array_qw, small_occurance_fraction_qw_easiest, color='blue')
plt.scatter(20, small_occurance_fraction_crosson, color='red')
plt.xlabel('n')
plt.show()

plt.figure()
for decile in deciles:
    plt.plot(n_array_aqc, small_occurance_fraction_aqc[decile, :], color=decile_colors_1[decile])
plt.plot(n_array_aqc, small_occurance_fraction_aqc_hardest, color='gold')
plt.plot(n_array_aqc, small_occurance_fraction_aqc_easiest, color='blue')
plt.scatter(20, small_occurance_fraction_crosson, color='red')
plt.xlabel('n')
plt.show()

# plt.figure()
# for decile in deciles:
#     plt.plot(n_array_qw, small_occurance_fraction_qw[decile, :], color=decile_colors_1[decile])
    # plt.plot(n_array_aqc, small_occurance_fraction_aqc[decile, :], color=decile_colors_2[decile])
# plt.scatter(20, small_occurance_fraction_crosson, color='red')
# plt.xlabel('n')
# plt.show()


# %%

n = 15
fraction = 0.02

all_formulae = get_all_formulae(n)
hardest_for_qw = get_hardest_formulae_qw(n, fraction)
hardest_for_aqc = get_hardest_formulae_aqc(n, fraction)
crosson = get_crosson_formulae()


# %%
# plot average distributions of positive vs negative differences

bins = np.arange(-0.5, (3*n)+0.5, step=1)
centres = [val + 0.5 for val in bins[:-1]]
all_hist = plt.hist(positive_vs_negative_differences_many_formulae(all_formulae, n), density=True, alpha=0.5, bins=bins, color='red')[0]
hardest_qw_hist = plt.hist(positive_vs_negative_differences_many_formulae(hardest_for_qw, n), density=True, alpha=0.3, bins=bins, color='forestgreen')[0]
hardest_aqc_hist = plt.hist(positive_vs_negative_differences_many_formulae(hardest_for_aqc, n), density=True, alpha=0.3, bins=bins, color='gold')[0]
# crosson_hist = plt.hist(positive_vs_negative_differences_many_formulae(crosson, n), density=True, alpha=0.3, bins=bins, color='purple')[0]
plt.plot(centres, all_hist, color='red')
plt.plot(centres, hardest_qw_hist, color='forestgreen')
plt.plot(centres, hardest_aqc_hist, color='gold')
# plt.plot(centres, crosson_hist, color='purple')
plt.xlim([0, 20])
plt.show()

# %%
# get deciled data for qw small differences

small_differences_fraction_qw = np.zeros((10, len(n_array_qw)))

frac = 0.05
small_differences_fraction_qw_hardest = np.zeros(len(n_array_qw))
small_differences_fraction_qw_easiest = np.zeros(len(n_array_qw))

for i, n in enumerate(n_array_qw):
    qw_deciled_formulae = get_deciled_formulae_qw(n)
    for decile in deciles:
        small_differences_fraction_qw[decile, i] = fraction_small_differences_many_formulae(qw_deciled_formulae[decile], n, max_difference=1)
    qw_hardest_formulae = get_hardest_formulae_qw(n, frac)
    qw_easiest_formulae = get_easiest_formulae_qw(n, frac)
    small_differences_fraction_qw_hardest[i] = fraction_small_differences_many_formulae(qw_hardest_formulae, n, max_difference=1)
    small_differences_fraction_qw_easiest[i] = fraction_small_differences_many_formulae(qw_easiest_formulae, n, max_difference=1)

# %%

small_differences_fraction_aqc = np.zeros((10, len(n_array_aqc)))

frac = 0.05
small_differences_fraction_aqc_hardest = np.zeros(len(n_array_aqc))
small_differences_fraction_aqc_easiest = np.zeros(len(n_array_aqc))

for i, n in enumerate(n_array_aqc):
    aqc_deciled_formulae = get_deciled_formulae_aqc(n)
    for decile in deciles:
        small_differences_fraction_aqc[decile, i] = fraction_small_differences_many_formulae(aqc_deciled_formulae[decile], n, max_difference=1)
    aqc_hardest_formulae = get_hardest_formulae_aqc(n, frac)
    aqc_easiest_formulae = get_easiest_formulae_aqc(n, frac)

    small_differences_fraction_aqc_hardest[i] = fraction_small_differences_many_formulae(aqc_hardest_formulae, n, max_difference=1)
    small_differences_fraction_aqc_easiest[i] = fraction_small_differences_many_formulae(aqc_easiest_formulae, n, max_difference=1)

# %%
# get the 'small differences fraction' for the Crosson instances

small_differences_fraction_crosson = fraction_small_differences_many_formulae(crosson, 20, max_difference=1)

# %%

decile_colors_1 = []
for i in range(10):
    cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
    decile_colors_1.append((0.0, cn1, cn2))

decile_colors_2 = []
for i in range(10):
    cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
    decile_colors_2.append((cn1, cn2, 0.0))

plt.figure()
for decile in deciles:
    plt.plot(n_array_qw, small_differences_fraction_qw[decile, :], color=decile_colors_1[decile])
plt.plot(n_array_qw, small_differences_fraction_qw_hardest, color='gold')
plt.plot(n_array_qw, small_differences_fraction_qw_easiest, color='blue')
plt.scatter(20, small_differences_fraction_crosson, color='red')
plt.xlabel('n')
plt.show()

plt.figure()
for decile in deciles:
    plt.plot(n_array_aqc, small_differences_fraction_aqc[decile, :], color=decile_colors_1[decile])
plt.plot(n_array_aqc, small_differences_fraction_aqc_hardest, color='gold')
plt.plot(n_array_aqc, small_differences_fraction_aqc_easiest, color='blue')
plt.scatter(20, small_differences_fraction_crosson, color='red')
plt.xlabel('n')
plt.show()

# plt.figure()
# for decile in deciles:
#     plt.plot(n_array_qw, small_differences_fraction_qw[decile, :], color=decile_colors_1[decile])
#     plt.plot(n_array_aqc, small_differences_fraction_aqc[decile, :], color=decile_colors_2[decile])
# plt.scatter(20, small_differences_fraction_crosson, color='red')
# plt.xlabel('n')
# plt.show()
