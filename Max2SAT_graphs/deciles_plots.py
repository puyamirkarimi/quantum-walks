# %%
# imports

from scipy import optimize
from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.transforms as mtransforms

# %%
# initialisations

n_array_qw = np.arange(5, 21)
n_array_aqc = np.arange(5, 16)
n_array_aqc_reduced = np.arange(5, 15)

line = lambda x, m, c: m*x + c

deciles = np.arange(10, dtype=int)
decile_boundaries = np.arange(9, dtype=int)

# decile_colors_1 = np.zeros((10, 3))
# for i in range(10):
#     cn1, cn2 = 0.25 + ((i/10)*(0.75)), ((i/10)*(76/255)/1)
#     decile_colors_1[9-i, :] = (0.0, cn1, cn2)

decile_colors_1 = ['#99daff', '#88c5ea', '#77b1d5', '#669dc0', '#5689ac', '#467698', '#366485', '#265272', '#14405f', '#00304d']

blue = '#0072B2'
orange = '#EF6900'
green = '#009E73'

plt.rc('text', usetex=True)
plt.rc('font', size=14)

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


def get_hardest_formulae_qw(n, frac, return_indices=False):
    '''returns the hardest "frac" fraction of instances for QW at a given n'''
    print(
        f'Getting the hardest {frac} fraction of formulae of size n={n} for QW')
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
    print(
        f'Getting the easiest {frac} fraction of formulae of size n={n} for QW')
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


def get_hardest_boundary_formula_qw(n, frac, return_index=False):
    '''returns the instance where exactly "frac" fraction of instances are harder for QW'''
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    instance = int(frac * 10000)
    boundary_index = np.argsort(success_probs)[instance]
    if return_index:
        return boundary_index
    boundary_instance_name = instance_names[boundary_index]
    return get_2sat_formula(boundary_instance_name)


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


def get_decile_boundary_formulae_qw(n, return_indices=False):
    '''returns the nine formulae just below the boundaries of the QW deciles
       (i.e. the hardest formula of each decile, other than decile 10)'''
    print(f'Getting the QW decile boundary formulae of size n={n}')
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    indices_by_hardness = np.argsort(success_probs)
    boundary_instances = []
    for decile in range(9):
        print(f'Doing decile {decile+1}')
        boundary = int((9-decile) * (10000/10))
        index = indices_by_hardness[boundary]
        if return_indices:
            boundary_instances.append(index)
        else:
            boundary_instances.append(get_2sat_formula(instance_names[index]))
    return boundary_instances


def get_hardest_formulae_aqc(n, frac, return_indices=False):
    '''returns the hardest "frac" fraction of instances for AQC at a given n'''
    print(
        f'Getting the hardest {frac} fraction of formulae of size n={n} for AQC')
    durations = rerun_adiabatic_data(n)
    durations = nan_to_largest(durations, addition=1)
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
    print(
        f'Getting the easiest {frac} fraction of formulae of size n={n} for AQC')
    durations = rerun_adiabatic_data(n)
    durations = nan_to_largest(durations, addition=1)
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


def get_hardest_boundary_formula_aqc(n, frac, return_index=False):
    '''returns the instance where exactly "frac" fraction of instances are harder for AQC'''
    durations = rerun_adiabatic_data(n)
    durations = nan_to_largest(durations, addition=1)
    instance_names = get_instance_names(n)
    instance = int(frac * 10000)
    boundary_index = np.argsort(durations)[9999-instance]
    if return_index:
        return boundary_index
    boundary_instance_name = instance_names[boundary_index]
    return get_2sat_formula(boundary_instance_name)


def get_deciled_formulae_aqc(n, return_indices=False):
    '''returns instances of a given n organised by QW decile'''
    print(f'Getting the formulae of size n={n} organised by QW decile')
    durations = rerun_adiabatic_data(n)
    durations = nan_to_largest(durations, addition=1)
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


def get_decile_boundary_formulae_aqc(n, return_indices=False):
    '''returns the nine formulae just below the boundaries of the AQC deciles
       (i.e. the hardest formula of each decile, other than decile 10)'''
    print(f'Getting the AQC decile boundary formulae of size n={n}')
    durations = rerun_adiabatic_data(n)
    durations = nan_to_largest(durations, addition=1)
    instance_names = get_instance_names(n)
    indices_by_hardness = np.argsort(durations)
    boundary_instances = []
    for decile in range(9):
        print(f'Doing decile {decile+1}')
        boundary = int((decile + 1) * (10000/10) - 1)
        index = indices_by_hardness[boundary]
        if return_indices:
            boundary_instances.append(index)
        else:
            boundary_instances.append(get_2sat_formula(instance_names[index]))
    return boundary_instances


def get_crosson_formulae():
    '''returns the instances that were data-mined to be hard for QA by Crosson'''
    instances = []
    for i in range(137):
        instance_name = str(i)
        instances.append(np.loadtxt(
            "./../../instances_crosson/" + instance_name + ".m2s").astype(int))
    return np.array(instances)


def get_instance_names(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 0].astype(str)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt(
        'm2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return instance_data[:, 0], instance_data[:, 1]


def get_2sat_formula(instance_name):
    # path of instance files in adam's format
    out = np.loadtxt("./../../instances_original/" + instance_name + ".m2s")
    return out.astype(int)


def adams_quantum_walk_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1+(n-5)*10000, usecols=2, max_rows=10000, dtype=str).astype(float)


def adams_quantum_walk_data_crosson():
    '''average success prob from T=0 to T=100 for the crosson instances'''
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug_crosson.csv', delimiter=',', skip_header=1, dtype=str)[:, 2].astype(float)


def get_instance_success_prob(n, instance):
    return float(np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1+(n-5)*10000+instance, usecols=2, max_rows=1, dtype=str))


def adams_adiabatic_data(n):
    '''returns time required to get 0.99 success probability'''
    a = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',',
                      missing_values='', skip_header=1+(n-5)*10000, usecols=10, max_rows=10000, dtype=str)
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


def rerun_adiabatic_data(n):
    '''returns time required to get 0.99 success probability'''
    a = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/aqc_times_rerun.csv', delimiter=',',
                      skip_header=1+(n-5)*10000, usecols=2, max_rows=10000, dtype=str)
    b = []
    skipped = 0
    for i, element in enumerate(a):
        if element != 'None':
            b.append(float(element))
        else:
            b.append(float('nan'))
            skipped += 1
    print("n:", n, " skipped:", skipped)
    return np.array(b)


def get_instance_duration_adams(n, instance):
    val = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',',
                        missing_values='', skip_header=1+(n-5)*10000+instance, usecols=10, max_rows=1, dtype=str)
    if val != '':
        return float(val)
    return float('nan')


def get_instance_duration(n, instance):
    val = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/aqc_times_rerun.csv', delimiter=',',
                        skip_header=1+(n-5)*10000+instance, usecols=2, max_rows=1, dtype=str)
    if val != 'None':
        return float(val)
    return float('nan')


def nan_to_largest(x, addition=1):
    max = np.max(x[np.logical_not(np.isnan(x))])
    return np.nan_to_num(x, nan=max+addition)


def nan_to_pos_inf(x):
    return np.nan_to_num(x, nan=float('inf'))


def remove_nan(x):
    return x[np.logical_not(np.isnan(x))]


# %%
# get decile boundary indices for QW and AQC

# QW
qw_decile_boundary_indices = []    # list of list of indices
for n in n_array_qw:
    qw_decile_boundary_indices.append(
        get_decile_boundary_formulae_qw(n, return_indices=True))

# AQC
aqc_decile_boundary_indices = []   # list of list of indices
for n in n_array_aqc:
    aqc_decile_boundary_indices.append(
        get_decile_boundary_formulae_aqc(n, return_indices=True))

# %%
# get hardest fraction boundary indices for QW and AQC

fraction = 0.01

qw_hardest_fraction_boundary_indices = []    # list of indices
for n in n_array_qw:
    qw_hardest_fraction_boundary_indices.append(
        get_hardest_boundary_formula_qw(n, fraction, return_index=True))

aqc_hardest_fraction_boundary_indices = []    # list of indices
for n in n_array_aqc:
    aqc_hardest_fraction_boundary_indices.append(
        get_hardest_boundary_formula_aqc(n, fraction, return_index=True))

# %%
# Get crosson data

crosson_formulae = get_crosson_formulae()

crosson_probs = adams_quantum_walk_data_crosson()
mean_crosson_prob = np.mean(crosson_probs)
median_crosson_prob = np.median(crosson_probs)

# %%
# get decile boundary success probabilities and durations

success_probabilities_qw_decile_boundaries = np.zeros((9, len(n_array_qw)))

for i, n in enumerate(n_array_qw):
    for decile in decile_boundaries:
        success_probabilities_qw_decile_boundaries[decile, i] = get_instance_success_prob(
            n, qw_decile_boundary_indices[i][decile])

success_probabilities_aqc_decile_boundaries = np.zeros((9, len(n_array_aqc)))

durations_qw_decile_boundaries = np.zeros((9, len(n_array_aqc)))
durations_aqc_decile_boundaries = np.zeros((9, len(n_array_aqc)))

for i, n in enumerate(n_array_aqc):
    for decile in decile_boundaries:
        success_probabilities_aqc_decile_boundaries[decile, i] = get_instance_success_prob(
            n, aqc_decile_boundary_indices[i][decile])
        durations_qw_decile_boundaries[decile, i] = get_instance_duration(
            n, qw_decile_boundary_indices[i][decile])
        durations_aqc_decile_boundaries[decile, i] = get_instance_duration(
            n, aqc_decile_boundary_indices[i][decile])

# %%
# get hardest fraction boundary success probabilities and durations

success_probabilities_qw_hardest_fraction_boundary = np.zeros(len(n_array_qw))
for i, n in enumerate(n_array_qw):
    success_probabilities_qw_hardest_fraction_boundary[i] = get_instance_success_prob(
        n, qw_hardest_fraction_boundary_indices[i])


success_probabilities_aqc_hardest_fraction_boundary = np.zeros(len(n_array_aqc))

durations_qw_hardest_fraction_boundary = np.zeros(len(n_array_aqc))
durations_aqc_hardest_fraction_boundary = np.zeros(len(n_array_aqc))

for i, n in enumerate(n_array_aqc):
    success_probabilities_aqc_hardest_fraction_boundary[i] = get_instance_success_prob(
        n, aqc_hardest_fraction_boundary_indices[i])
    durations_qw_hardest_fraction_boundary[i] = get_instance_duration(
        n, qw_hardest_fraction_boundary_indices[i])
    durations_aqc_hardest_fraction_boundary[i] = get_instance_duration(
        n, aqc_hardest_fraction_boundary_indices[i])

# %%
# get decile indices for QW and AQC

# QW
qw_deciled_indices = []    # list of arrays of indices arrays
for n in n_array_qw:
    qw_deciled_indices.append(
        get_deciled_formulae_qw(n, return_indices=True))

qw_hardest_fraction_indices = []
for n in n_array_qw:
    qw_hardest_fraction_indices.append(
        get_hardest_formulae_qw(n, fraction, return_indices=True))

# AQC
aqc_deciled_indices = []   # list of arrays of indices arrays
for n in n_array_aqc:
    aqc_deciled_indices.append(
        get_deciled_formulae_aqc(n, return_indices=True))

aqc_hardest_fraction_indices = []
for n in n_array_aqc:
    aqc_hardest_fraction_indices.append(
        get_hardest_formulae_aqc(n, fraction, return_indices=True))

# %%
# get mean/median decile success probabilities and durations (and for hardest fractions)

mean_success_probabilities_qw_deciles = np.zeros((10, len(n_array_qw)))
median_success_probabilities_qw_deciles = np.zeros((10, len(n_array_qw)))
mean_success_probabilities_qw_hardest_fraction = np.zeros(len(n_array_qw))
median_success_probabilities_qw_hardest_fraction = np.zeros(len(n_array_qw))

for i, n in enumerate(n_array_qw):
    success_probs = adams_quantum_walk_data(n)
    for decile in deciles:
        qw_decile_success_probs = success_probs[qw_deciled_indices[i][decile]]
        mean_success_probabilities_qw_deciles[decile, i] = np.mean(
            qw_decile_success_probs)
        median_success_probabilities_qw_deciles[decile, i] = np.median(
            qw_decile_success_probs)

    qw_hardest_success_probs = success_probs[qw_hardest_fraction_indices[i]]
    mean_success_probabilities_qw_hardest_fraction[i] = np.mean(
        qw_hardest_success_probs)
    median_success_probabilities_qw_hardest_fraction[i] = np.median(
        qw_hardest_success_probs)

mean_success_probabilities_aqc_deciles = np.zeros((10, len(n_array_aqc)))
median_success_probabilities_aqc_deciles = np.zeros((10, len(n_array_aqc)))
mean_success_probabilities_aqc_hardest_fraction = np.zeros(len(n_array_aqc))
median_success_probabilities_aqc_hardest_fraction = np.zeros(len(n_array_aqc))

mean_durations_qw_deciles = np.zeros((10, len(n_array_aqc)))
median_durations_qw_deciles = np.zeros((10, len(n_array_aqc)))
mean_durations_aqc_deciles = np.zeros((10, len(n_array_aqc)))
median_durations_aqc_deciles = np.zeros((10, len(n_array_aqc)))
mean_durations_qw_hardest_fraction = np.zeros(len(n_array_aqc))
median_durations_qw_hardest_fraction = np.zeros(len(n_array_aqc))
mean_durations_aqc_hardest_fraction = np.zeros(len(n_array_aqc))
median_durations_aqc_hardest_fraction = np.zeros(len(n_array_aqc))

success_probabilities_aqc_hardest_fraction = np.zeros((len(n_array_aqc), int(fraction*10000)))
durations_qw_hardest_fraction = np.zeros((len(n_array_aqc), int(fraction*10000)))

for i, n in enumerate(n_array_aqc):
    success_probs = adams_quantum_walk_data(n)
    durations = rerun_adiabatic_data(n)
    for decile in deciles:
        aqc_decile_success_probs = success_probs[aqc_deciled_indices[i][decile]]
        mean_success_probabilities_aqc_deciles[decile, i] = np.mean(
            aqc_decile_success_probs)
        median_success_probabilities_aqc_deciles[decile, i] = np.median(
            aqc_decile_success_probs)

        qw_decile_durations = durations[qw_deciled_indices[i][decile]]
        mean_durations_qw_deciles[decile, i] = np.mean(remove_nan(qw_decile_durations))
        median_durations_qw_deciles[decile, i] = np.median(nan_to_pos_inf(qw_decile_durations))
        if median_durations_qw_deciles[decile, i] == float('inf'):
            median_durations_qw_deciles[decile, i] = float('nan')

        aqc_decile_durations = durations[aqc_deciled_indices[i][decile]]
        mean_durations_aqc_deciles[decile, i] = np.mean(remove_nan(aqc_decile_durations))
        median_durations_aqc_deciles[decile, i] = np.median(nan_to_pos_inf(aqc_decile_durations))
        if median_durations_aqc_deciles[decile, i] == float('inf'):
            median_durations_aqc_deciles[decile, i] = float('nan')

    aqc_hardest_success_probs = success_probs[aqc_hardest_fraction_indices[i]]
    mean_success_probabilities_aqc_hardest_fraction[i] = np.mean(aqc_hardest_success_probs)
    median_success_probabilities_aqc_hardest_fraction[i] = np.median(aqc_hardest_success_probs)
    success_probabilities_aqc_hardest_fraction[i, :] = aqc_hardest_success_probs

    qw_hardest_durations = durations[qw_hardest_fraction_indices[i]]
    mean_durations_qw_hardest_fraction[i] = np.mean(remove_nan(qw_hardest_durations))
    median_durations_qw_hardest_fraction[i] = np.median(nan_to_pos_inf(qw_hardest_durations))
    if median_durations_qw_hardest_fraction[i] == float('inf'):
        median_durations_qw_hardest_fraction[i] = float('nan')
    durations_qw_hardest_fraction[i, :] = nan_to_pos_inf(qw_hardest_durations)
    
    aqc_hardest_durations = durations[aqc_hardest_fraction_indices[i]]
    mean_durations_aqc_hardest_fraction[i] = np.mean(remove_nan(aqc_hardest_durations))
    median_durations_aqc_hardest_fraction[i] = np.median(nan_to_pos_inf(aqc_hardest_durations))
    if median_durations_aqc_hardest_fraction[i] == float('inf'):
        median_durations_aqc_hardest_fraction[i] = float('nan')
    
    if np.isnan(durations_aqc_hardest_fraction_boundary[i]):
        # cannot calculate the median QW success probabilty for the hardest
        # instances when more than 100 instances were skipped for AQC
        median_success_probabilities_aqc_hardest_fraction[i] = float('nan')

# %%
# pickle the important data

# data = [mean_crosson_prob,
#         median_crosson_prob,
#         success_probabilities_qw_decile_boundaries,
#         success_probabilities_aqc_decile_boundaries,
#         durations_qw_decile_boundaries,
#         durations_aqc_decile_boundaries,
#         success_probabilities_qw_hardest_fraction_boundary,
#         success_probabilities_aqc_hardest_fraction_boundary,
#         durations_qw_hardest_fraction_boundary,
#         durations_aqc_hardest_fraction_boundary,
#         mean_success_probabilities_qw_deciles,
#         median_success_probabilities_qw_deciles,
#         mean_success_probabilities_qw_hardest_fraction,
#         median_success_probabilities_qw_hardest_fraction,
#         mean_success_probabilities_aqc_deciles,
#         median_success_probabilities_aqc_deciles,
#         mean_success_probabilities_aqc_hardest_fraction,
#         median_success_probabilities_aqc_hardest_fraction,
#         mean_durations_qw_deciles,
#         median_durations_qw_deciles,
#         mean_durations_aqc_deciles,
#         median_durations_aqc_deciles,
#         mean_durations_qw_hardest_fraction,
#         median_durations_qw_hardest_fraction,
#         mean_durations_aqc_hardest_fraction,
#         median_durations_aqc_hardest_fraction,
#         success_probabilities_aqc_hardest_fraction,
#         durations_qw_hardest_fraction
#         ]

# with open('decile_plots.pkl', 'wb') as f:
#     pkl.dump(data, f)

# %%
# unpickle the important data

with open('decile_plots.pkl', 'rb') as f:
    (mean_crosson_prob,
    median_crosson_prob,
    success_probabilities_qw_decile_boundaries,
    success_probabilities_aqc_decile_boundaries,
    durations_qw_decile_boundaries,
    durations_aqc_decile_boundaries,
    success_probabilities_qw_hardest_fraction_boundary,
    success_probabilities_aqc_hardest_fraction_boundary,
    durations_qw_hardest_fraction_boundary,
    durations_aqc_hardest_fraction_boundary,
    mean_success_probabilities_qw_deciles,
    median_success_probabilities_qw_deciles,
    mean_success_probabilities_qw_hardest_fraction,
    median_success_probabilities_qw_hardest_fraction,
    mean_success_probabilities_aqc_deciles,
    median_success_probabilities_aqc_deciles,
    mean_success_probabilities_aqc_hardest_fraction,
    median_success_probabilities_aqc_hardest_fraction,
    mean_durations_qw_deciles,
    median_durations_qw_deciles,
    mean_durations_aqc_deciles,
    median_durations_aqc_deciles,
    mean_durations_qw_hardest_fraction,
    median_durations_qw_hardest_fraction,
    mean_durations_aqc_hardest_fraction,
    median_durations_aqc_hardest_fraction,
    success_probabilities_aqc_hardest_fraction,
    durations_qw_hardest_fraction) = pkl.load(f)

# %%
# percentile plots for QW/AQC using corresponding hardness

fig, axs = plt.subplots(2, 2, figsize=(6, 4.3))

# log-linear plot of QW probabilities/QW deciles using boundary values

scalings = np.zeros_like(decile_boundaries, dtype=np.float64)
scalings_error = np.zeros_like(decile_boundaries, dtype=np.float64)
for decile in decile_boundaries:
    y = np.log2(success_probabilities_qw_decile_boundaries[decile, :])
    par, cov = optimize.curve_fit(line, n_array_qw, y)
    m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
    fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_qw])
    scalings[decile] = m[0]
    scalings_error[decile] = m[1]

    axs[0, 0].scatter(
        n_array_qw, success_probabilities_qw_decile_boundaries[decile, :], color=decile_colors_1[decile], s=10)
    axs[0, 0].plot(n_array_qw, fit, color=decile_colors_1[decile], linewidth=1)
    axs[0, 0].scatter(20, median_crosson_prob, color=orange, marker='s', s=10)

y = np.log2(success_probabilities_qw_hardest_fraction_boundary)
par, cov = optimize.curve_fit(line, n_array_qw, y)
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
scaling_hardest = m[0]
scaling_hardest_error = m[1]
fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_qw])
axs[0, 0].scatter(
    n_array_qw, success_probabilities_qw_hardest_fraction_boundary, color=green, marker='^', s=10)
axs[0, 0].plot(n_array_qw, fit, color=green, linewidth=1)
axs[0, 0].set_yscale('log', base=2)
axs[0, 0].set_ylabel('$\overline{P}(0, 100)$', fontsize=15)
axs[0, 0].set_xlabel('$n$', fontsize=15)
axs[0, 0].set_xticks(np.arange(5, 25, 5))
axs[0, 0].set_yticks([2**(-9), 2**(-6), 2**(-3)])
axs[0, 0].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[0, 0].text(0, 0.3, '(a)', transform=axs[0, 0].transAxes + trans, verticalalignment='top', fontsize=15)

for i in range(9):
    axs[0, 1].scatter(10 * (decile_boundaries[i]+1), scalings[i], color=decile_colors_1[i], s=10)
    axs[0, 1].errorbar(10*(decile_boundaries[i]+1), scalings[i], yerr=scalings_error[i], capsize=1.7, fmt='none', color=decile_colors_1[i])
    if i < 8:
        axs[0, 1].plot(10 * (decile_boundaries[i:i+2]+1), scalings [i:i+2],
                    color=decile_colors_1[i], linestyle='--', linewidth=1)
axs[0, 1].scatter(10 * 9.99, scaling_hardest, color=green, marker='^', s=10)
axs[0, 1].errorbar(10*9.99, scaling_hardest, yerr=scaling_hardest_error, capsize=1.7, fmt='none', ecolor=green)
axs[0, 1].set_ylabel(r'$\kappa$', fontsize=15)
axs[0, 1].set_xlabel('QW difficulty percentile', fontsize=14)
axs[0, 1].set_xticks(np.arange(20, 120, 20))
axs[0, 1].set_yticks([-0.55, -0.5, -0.45, -0.4])
axs[0, 1].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[0, 1].text(0, 0.3, '(b)', transform=axs[0, 1].transAxes + trans, verticalalignment='top', fontsize=15)

print('Scaling exponents for QW deciles:')
for i in range(len(deciles)-1):
    print(f'{scalings[i]} +- {scalings_error[i]}')
print(f'Scaling exponent for the 99th QW percentile: {scaling_hardest} +- {scaling_hardest_error}')

# log-linear plot of AQC durations/AQC deciles using boundary values

scalings = np.zeros_like(decile_boundaries, dtype=np.float64)
scalings_error = np.zeros_like(decile_boundaries, dtype=np.float64)
for decile in decile_boundaries:
    y = np.log2(durations_aqc_decile_boundaries[decile, :])
    par, cov = optimize.curve_fit(line, n_array_aqc, y)
    m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
    fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
    scalings[decile] = m[0]
    scalings_error[decile] = m[1]

    axs[1, 0].scatter(n_array_aqc,
                      durations_aqc_decile_boundaries[decile, :], color=decile_colors_1[decile], s=10)
    axs[1, 0].plot(n_array_aqc, fit, color=decile_colors_1[decile], linewidth=1)

y = np.log2(durations_aqc_hardest_fraction_boundary)
y = y[~np.isnan(y)]
par, cov = optimize.curve_fit(line, n_array_aqc[:len(y)], y)
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
scaling_hardest = m[0]
scaling_hardest_error = m[1]
fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
axs[1, 0].scatter(n_array_aqc,
                  durations_aqc_hardest_fraction_boundary, color=green, marker='^', s=10)
axs[1, 0].plot(n_array_aqc, fit, color=green, linewidth=1)
axs[1, 0].set_yscale('log', base=2)
axs[1, 0].set_ylabel('$t_{0.99}$', fontsize=15)
axs[1, 0].set_xlabel('$n$', fontsize=15)
axs[1, 0].set_xticks(np.arange(5, 20, 5))
axs[1, 0].set_yticks([2**(6), 2**(9), 2**(12)])
axs[1, 0].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[1, 0].text(0, 0.96, '(c)', transform=axs[1, 0].transAxes + trans, verticalalignment='top', fontsize=15)

for i in range(9):
    axs[1, 1].scatter(10 * (decile_boundaries[i]+1), scalings[i], color=decile_colors_1[i], s=10)
    axs[1, 1].errorbar(10*(decile_boundaries[i]+1), scalings[i], yerr=scalings_error[i], capsize=1.7, fmt='none', color=decile_colors_1[i])
    if i < 8:
        axs[1, 1].plot(10 * (decile_boundaries[i:i+2]+1), scalings [i:i+2],
                    color=decile_colors_1[i], linestyle='--', linewidth=1)
axs[1, 1].scatter(10 * 9.99, scaling_hardest, color=green, marker='^', s=10)
axs[1, 1].errorbar(10*9.99, scaling_hardest, yerr=scaling_hardest_error, capsize=1.7, fmt='none', ecolor=green)
axs[1, 1].set_ylabel(r'$\kappa$', fontsize=15)
axs[1, 1].set_xlabel('AQC difficulty percentile', fontsize=14)
axs[1, 1].set_xticks(np.arange(20, 120, 20))
axs[1, 1].set_yticks([0.2, 0.4, 0.6])
axs[1, 1].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[1, 1].text(0, 0.96, '(d)', transform=axs[1, 1].transAxes + trans, verticalalignment='top', fontsize=15)

print('\nScaling exponents for AQC deciles:')
for i in range(len(deciles)-1):
    print(f'{scalings[i]} +- {scalings_error[i]}')
print(f'Scaling exponent for the 99th AQC percentile: {scaling_hardest} +- {scaling_hardest_error}')

for axs_arr in axs:
    for ax in axs_arr:
        ax.tick_params(direction='in', which='both')

fig.tight_layout()
# plt.savefig('qw_aqc_decile_boundaries_windows.pdf', dpi=200)
plt.show()

# %%
# percentile plots for QW/AQC using other algorithm's hardness

fig, axs = plt.subplots(2, 2, figsize=(6, 4.3))

# log-linear plot of QW probabilities/AQC deciles using median values

scalings = np.zeros_like(deciles, dtype=np.float64)
scalings_error = np.zeros_like(deciles, dtype=np.float64)
for decile in deciles:
    y = np.log2(median_success_probabilities_aqc_deciles[decile, :])
    par, cov = optimize.curve_fit(line, n_array_aqc, y)
    m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
    fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
    scalings[decile] = m[0]
    scalings_error[decile] = m[1]

    axs[0, 0].scatter(n_array_aqc,
                      median_success_probabilities_aqc_deciles[decile, :], color=decile_colors_1[decile], s=10)
    axs[0, 0].plot(n_array_aqc, fit, color=decile_colors_1[decile], linewidth=1)

y = np.log2(median_success_probabilities_aqc_hardest_fraction)
num_nans = np.count_nonzero(np.isnan(y))
par, cov = optimize.curve_fit(line, n_array_aqc[:-num_nans], y[:-num_nans])
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
scaling_hardest = m[0]
scaling_hardest_error = m[1]
fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
axs[0, 0].scatter(n_array_aqc,
                  median_success_probabilities_aqc_hardest_fraction, marker='^', color=green, s=10)
axs[0, 0].plot(n_array_aqc, fit, color=green, linewidth=1)
axs[0, 0].set_yscale('log', base=2)
axs[0, 0].set_ylabel('$\overline{P}(0, 100)$', fontsize=15)
axs[0, 0].set_xlabel('$n$', fontsize=15)
axs[0, 0].set_xticks(np.arange(5, 20, 5))
axs[0, 0].set_yticks([2**(-7), 2**(-5), 2**(-3)])
axs[0, 0].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[0, 0].text(0, 0.3, '(a)', transform=axs[0, 0].transAxes + trans, verticalalignment='top', fontsize=15)

for i in range(10):
    axs[0, 1].errorbar(deciles[i]+1, scalings[i], yerr=scalings_error[i], capsize=1.7, fmt='none', ecolor=decile_colors_1[i])
    axs[0, 1].scatter(deciles[i]+1, scalings[i], color=decile_colors_1[i], s=10)
    if i < 9:
        axs[0, 1].plot(deciles[i:i+2]+1, scalings[i:i+2], color=decile_colors_1[i], linestyle='--', linewidth=1)
axs[0, 1].set_ylabel(r'$\kappa$', fontsize=15)
axs[0, 1].set_xlabel('AQC difficulty decile', fontsize=14, loc='left')
axs[0, 1].set_xticks(range(1, 11, 2))
ylims = (-0.627, -0.3709738872006789)
axs[0, 1].set_ylim(ylims)
axs[0, 1].tick_params(axis='both', labelsize=12)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[0, 1].text(0, 0.3, '(b)', transform=axs[0, 1].transAxes + trans, verticalalignment='top', fontsize=15)

# divider for hardest 1%
divider = make_axes_locatable(axs[0, 1])
ax2 = divider.append_axes("right", size="10%", pad=0)
axs[0, 1].figure.add_axes(ax2)
ax2.scatter(1, scaling_hardest, marker='^', color=green, s=10)
ax2.errorbar(1, scaling_hardest, yerr=scaling_hardest_error, capsize=1.7, fmt='none', marker='^', ecolor=green)
ax2.set_ylim(ylims)
ax2.set_xticks([1])
ax2.set_xticklabels(['Top 1\%'], rotation=45, fontsize=12)
ax2.set_yticks([])
ax2.tick_params(direction='in', which='both')

print('\nScaling exponents for the deciles (top plot):')
for i in range(len(deciles)-1):
    print(f'{scalings[i]} +- {scalings_error[i]}')
print(f'Scaling exponent for the hardest 1% (top plot): {scaling_hardest} +- {scaling_hardest_error}')

# log-linear plot of AQC durations/QW deciles using median values

scalings = np.zeros_like(deciles, dtype=np.float64)
scalings_error = np.zeros_like(deciles, dtype=np.float64)
for decile in deciles:
    y = np.log2(median_durations_qw_deciles[decile, :])
    par, cov = optimize.curve_fit(line, n_array_aqc, y)
    m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
    fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
    scalings[decile] = m[0]
    scalings_error[decile] = m[1]

    axs[1, 0].scatter(n_array_aqc,
                      median_durations_qw_deciles[decile, :], color=decile_colors_1[decile], s=10)
    axs[1, 0].plot(n_array_aqc, fit, color=decile_colors_1[decile], linewidth=1)

y = np.log2(median_durations_qw_hardest_fraction)
par, cov = optimize.curve_fit(line, n_array_aqc, y)
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
scaling_hardest = m[0]
scaling_hardest_error = m[1]
fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
axs[1, 0].scatter(n_array_aqc,
                  median_durations_qw_hardest_fraction, marker='^', color=green, s=10)
axs[1, 0].plot(n_array_aqc, fit, color=green, linewidth=1)
axs[1, 0].set_yscale('log', base=2)
axs[1, 0].set_ylabel('$t_{0.99}$', fontsize=15)
axs[1, 0].set_xlabel('$n$', fontsize=15)
axs[1, 0].set_yticks([2**6, 2**8, 2**10])
axs[1, 0].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[1, 0].text(0, 0.96, '(c)', transform=axs[1, 0].transAxes + trans, verticalalignment='top', fontsize=15)

for i in range(10):
    axs[1, 1].errorbar(deciles[i]+1, scalings[i], yerr=scalings_error[i], capsize=1.7, fmt='none', ecolor=decile_colors_1[i])
    axs[1, 1].scatter(deciles[i]+1, scalings[i], color=decile_colors_1[i], s=10)
    if i < 9:
        axs[1, 1].plot(deciles[i:i+2]+1, scalings[i:i+2], color=decile_colors_1[i], linestyle='--', linewidth=1)
axs[1, 1].set_ylabel(r'$\kappa$', fontsize=15)
axs[1, 1].set_xlabel('QW difficulty decile', fontsize=14, loc='left')
axs[1, 1].set_xticks(range(1, 11, 2))
axs[1, 1].set_yticks(np.arange(0.1, 0.6, 0.1))
ylims = (0.060755245404130416, 0.5)
axs[1, 1].set_ylim(ylims)
axs[1, 1].tick_params(axis='both', labelsize=13)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[1, 1].text(0, 0.96, '(d)', transform=axs[1, 1].transAxes + trans, verticalalignment='top', fontsize=15)

# divider for hardest 1%
divider = make_axes_locatable(axs[1, 1])
ax2 = divider.append_axes("right", size="10%", pad=0)
axs[1, 1].figure.add_axes(ax2)
ax2.scatter(1, scaling_hardest, marker='^', color=green, s=10)
ax2.errorbar(1, scaling_hardest, yerr=scaling_hardest_error, capsize=1.7, fmt='none', marker='^', ecolor=green)
ax2.set_ylim(ylims)
ax2.set_xticks([1])
ax2.set_xticklabels(['Top 1\%'], rotation=45, fontsize=12)
ax2.set_yticks([])
ax2.tick_params(direction='in', which='both')

for axs_arr in axs:
    for ax in axs_arr:
        ax.tick_params(direction='in', which='both')

print('\nScaling exponents for the deciles (bottom plot):')
for i in range(len(deciles)-1):
    print(f'{scalings[i]} +- {scalings_error[i]}')
print(f'Scaling exponent for the hardest 1% (bottom plot): {scaling_hardest} +- {scaling_hardest_error}')

fig.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.5)
# plt.savefig('qw_aqc_deciles_cross_comparison_windows.pdf', dpi=200)
plt.show()

# %%
# plot to check difference between decile boundary and mean

# fig, axs = plt.subplots(3, 2, figsize=(14, 15))

# line = lambda x, m, c: m*x + c

# # log-linear plot using averages

# for decile in deciles:
#     y = np.log2(mean_durations_aqc_deciles[decile, :])
#     par, cov = optimize.curve_fit(line, n_array_aqc, y)
#     m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
#     print(f'EXP {decile+1}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}')
#     fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])

#     axs[0, 0].scatter(n_array_aqc,
#                       mean_durations_aqc_deciles[decile, :], color=decile_colors_1[decile])
#     axs[0, 0].plot(n_array_aqc, fit, color=decile_colors_1[decile])

# axs[0, 0].set_yscale('log', base=2)

# axs[0, 0].set_ylabel('$\mathrm{Mean}(t_{0.99})$')
# axs[0, 0].set_xlabel('$n$')

# # log-log plot using averages

# for decile in deciles:
#     y = np.log2(mean_durations_aqc_deciles[decile, :])
#     x = np.log2(n_array_aqc)
#     par, cov = optimize.curve_fit(line, x, y)
#     m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
#     # fit = 2**np.array([line(x, m[0], c[0]) for x in np.log2(n_array_aqc)])
#     fit = 2**c[0] * n_array_aqc**m[0]
#     axs[0, 1].scatter(n_array_aqc,
#                       mean_durations_aqc_deciles[decile, :], color=decile_colors_1[decile])
#     axs[0, 1].plot(n_array_aqc, fit, color=decile_colors_1[decile])

# axs[0, 1].set_xscale('log', base=2)
# axs[0, 1].set_yscale('log', base=2)

# x_ticks = np.arange(5, 12)
# x_tick_labels = [f'${x}$' for x in x_ticks]
# axs[0, 1].set_xticks(x_ticks)
# axs[0, 1].set_xticklabels(x_tick_labels)

# axs[0, 1].set_ylabel('$\mathrm{Mean}(t_{0.99})$')
# axs[0, 1].set_xlabel('$n$')

# # log-linear plot using boundary values

# for decile in decile_boundaries:
#     y = np.log2(durations_aqc_decile_boundaries[decile, :])
#     par, cov = optimize.curve_fit(line, n_array_aqc, y)
#     m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
#     print(f'EXP {decile+1}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}')
#     fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])

#     axs[1, 0].scatter(n_array_aqc,
#                       durations_aqc_decile_boundaries[decile, :], color=decile_colors_1[decile])
#     axs[1, 0].plot(n_array_aqc, fit, color=decile_colors_1[decile])

# y = np.log2(durations_aqc_hardest_fraction_boundary)
# par, cov = optimize.curve_fit(line, n_array_aqc, y)
# m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
# fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])
# axs[1, 0].scatter(n_array_aqc,
#                   durations_aqc_hardest_fraction_boundary, color='blue')
# axs[1, 0].plot(n_array_aqc, fit, color='blue')

# axs[1, 0].set_yscale('log', base=2)

# axs[1, 0].set_ylabel('$\mathrm{DecileBoundary}(t_{0.99})$')
# axs[1, 0].set_xlabel('$n$')

# # log-log plot using boundary values

# for decile in decile_boundaries:
#     y = np.log2(durations_aqc_decile_boundaries[decile, :])
#     x = np.log2(n_array_aqc)
#     par, cov = optimize.curve_fit(line, x, y)
#     m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
#     # fit = 2**np.array([line(x, m[0], c[0]) for x in np.log2(n_array_aqc)])
#     fit = 2**c[0] * n_array_aqc**m[0]
#     axs[1, 1].scatter(n_array_aqc,
#                       durations_aqc_decile_boundaries[decile, :], color=decile_colors_1[decile])
#     axs[1, 1].plot(n_array_aqc, fit, color=decile_colors_1[decile])

# y = np.log2(durations_aqc_hardest_fraction_boundary)
# x = np.log2(n_array_aqc)
# par, cov = optimize.curve_fit(line, x, y)
# m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
# # fit = 2**np.array([line(x, m[0], c[0]) for x in np.log2(n_array_aqc)])
# fit = 2**c[0] * n_array_aqc**m[0]
# axs[1, 1].scatter(n_array_aqc,
#                   durations_aqc_hardest_fraction_boundary, color='blue')
# axs[1, 1].plot(n_array_aqc, fit, color='blue')

# axs[1, 1].set_xscale('log', base=2)
# axs[1, 1].set_yscale('log', base=2)

# x_ticks = np.arange(5, 12)
# x_tick_labels = [f'${x}$' for x in x_ticks]
# axs[1, 1].set_xticks(x_ticks)
# axs[1, 1].set_xticklabels(x_tick_labels)

# axs[1, 1].set_ylabel('$\mathrm{DecileBoundary}(t_{0.99})$')
# axs[1, 1].set_xlabel('$n$')

# # log-linear plot of median

# decile = 4

# y = np.log2(durations_aqc_decile_boundaries[decile, :])
# par, cov = optimize.curve_fit(line, n_array_aqc, y)
# m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
# print(f'EXP {decile+1}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}')
# fit = 2**np.array([line(x, m[0], c[0]) for x in n_array_aqc])

# axs[2, 0].scatter(n_array_aqc,
#                   durations_aqc_decile_boundaries[decile, :], color=decile_colors_1[decile])
# axs[2, 0].plot(n_array_aqc, fit, color=decile_colors_1[decile])

# axs[2, 0].set_yscale('log', base=2)

# axs[2, 0].set_ylabel('$\mathrm{Median}(t_{0.99})$')
# axs[2, 0].set_xlabel('$n$')

# # log-log plot of median

# y = np.log2(durations_aqc_decile_boundaries[decile, :])
# x = np.log2(n_array_aqc)
# par, cov = optimize.curve_fit(line, x, y)
# m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
# # fit = 2**np.array([line(x, m[0], c[0]) for x in np.log2(n_array_aqc)])
# fit = 2**c[0] * n_array_aqc**m[0]
# axs[2, 1].scatter(n_array_aqc,
#                   durations_aqc_decile_boundaries[decile, :], color=decile_colors_1[decile])
# axs[2, 1].plot(n_array_aqc, fit, color=decile_colors_1[decile])

# axs[2, 1].set_xscale('log', base=2)
# axs[2, 1].set_yscale('log', base=2)

# x_ticks = np.arange(5, 12)
# x_tick_labels = [f'${x}$' for x in x_ticks]
# axs[2, 1].set_xticks(x_ticks)
# axs[2, 1].set_xticklabels(x_tick_labels)

# axs[2, 1].set_ylabel('$\mathrm{Median}(t_{0.99})$')
# axs[2, 1].set_xlabel('$n$')

# # plt.savefig('aqc_deciles_plots.pdf', dpi=200)
# plt.show()
