# %%
# imports

from matplotlib import pyplot as plt
import numpy as np

# %%
# initialisations

n_array_qw = np.arange(5, 21)
n_array_aqc = np.arange(5, 16)
n_array_aqc_reduced = np.arange(5, 12)

deciles = np.arange(10, dtype=int)
decile_boundaries = np.arange(9, dtype=int)

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


def get_decile_boundary_formulae_qw(n, return_indices=False):
    '''returns the nine formulae on the boundaries of the QW deciles'''
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


def get_decile_boundary_formulae_aqc(n, return_indices=False):
    '''returns the nine formulae on the boundaries of the AQC deciles'''
    print(f'Getting the AQC decile boundary formulae of size n={n}')
    durations = adams_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    indices_by_hardness = np.argsort(durations)
    boundary_instances = []
    for decile in range(9):
        print(f'Doing decile {decile+1}')
        boundary = int((decile + 1) * (10000/10))
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
        instances.append(np.loadtxt("./../../instances_crosson/" + instance_name + ".m2s").astype(int))
    return np.array(instances)


def get_instance_names(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 0].astype(str)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return instance_data[:, 0], instance_data[:, 1]


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../instances_original/" + instance_name + ".m2s")  # path of instance files in adam's format
    return out.astype(int)


def adams_quantum_walk_data(n):
    return np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1+(n-5)*10000, usecols=2, max_rows=10000, dtype=str).astype(float)


def get_instance_success_prob(n, instance):
    return float(np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', skip_header=1+(n-5)*10000+instance, usecols=2, max_rows=1, dtype=str)).astype(float)


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


def get_instance_duration(n, instance):
    val = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/heug.csv', delimiter=',', missing_values='', skip_header=1+(n-5)*10000+instance, usecols=10, max_rows=1, dtype=str)
    if val != '':
        return float(val)
    return float('nan')


# %%
# get decile boundary indices for QW and AQC

# QW
qw_decile_boundary_indices = []    # list of list of indices
for n in n_array_qw:
    qw_decile_boundary_indices.append(get_decile_boundary_formulae_qw(n, return_indices=True))

# AQC
aqc_decile_boundary_indices = []   # list of list of indices
for n in n_array_aqc_reduced:
    aqc_decile_boundary_indices.append(get_decile_boundary_formulae_aqc(n, return_indices=True))

# %%
# Get crosson formulae

crosson_formulae = get_crosson_formulae()

# %%
# get decile boundary success probabilities and durations

success_probabilities_qw_decile_boundaries = np.zeros((9, len(n_array_qw)))

for i, n in enumerate(n_array_qw):
    for decile in decile_boundaries:
        success_probabilities_qw_decile_boundaries[decile, i] = get_instance_success_prob(n, qw_decile_boundary_indices[i][decile])

success_probabilities_aqc_decile_boundaries = np.zeros((9, len(n_array_aqc)))

durations_qw_decile_boundaries = np.zeros((9, len(n_array_aqc_reduced)))
durations_aqc_decile_boundaries = np.zeros((9, len(n_array_aqc_reduced)))

for i, n in enumerate(n_array_aqc_reduced):
    for decile in decile_boundaries:
        success_probabilities_aqc_decile_boundaries[decile, i] = get_instance_success_prob(n, aqc_decile_boundary_indices[i][decile])
        durations_qw_decile_boundaries[decile, i] = get_instance_duration(n, qw_decile_boundary_indices[i][decile])
        durations_aqc_decile_boundaries[decile, i] = get_instance_duration(n, aqc_decile_boundary_indices[i][decile])

# %%
# plot

plt.figure()

for decile in decile_boundaries:
    plt.scatter(n_array_aqc_reduced, durations_aqc_decile_boundaries[decile, :], color=decile_colors_1[decile])

plt.yscale('log', base=2)
plt.show()

# %%
# get decile indices for QW and AQC

# QW
qw_deciled_indices = []    # list of arrays of indices arrays
for n in n_array_qw:
    qw_deciled_indices.append(get_deciled_formulae_qw(n, return_indices=True))

# AQC
aqc_deciled_indices = []   # list of arrays of indices arrays
for n in n_array_aqc_reduced:
    aqc_deciled_indices.append(get_deciled_formulae_aqc(n, return_indices=True))

# %%
# get decile boundary success probabilities and durations

mean_success_probabilities_qw_deciles = np.zeros((10, len(n_array_qw)))
median_success_probabilities_qw_deciles = np.zeros((10, len(n_array_qw)))

for i, n in enumerate(n_array_qw):
    success_probs = adams_quantum_walk_data(n)
    for decile in deciles:
        qw_decile_success_probs = success_probs[qw_deciled_indices[i][decile]]
        mean_success_probabilities_qw_deciles[decile, i] = np.mean(qw_decile_success_probs)
        median_success_probabilities_qw_deciles[decile, i] = np.median(qw_decile_success_probs)

mean_success_probabilities_aqc_deciles = np.zeros((10, len(n_array_aqc_reduced)))
median_success_probabilities_aqc_deciles = np.zeros((10, len(n_array_aqc_reduced)))

mean_durations_qw_deciles = np.zeros((10, len(n_array_aqc_reduced)))
median_durations_qw_deciles = np.zeros((10, len(n_array_aqc_reduced)))
mean_durations_aqc_deciles = np.zeros((10, len(n_array_aqc_reduced)))
median_durations_aqc_deciles = np.zeros((10, len(n_array_aqc_reduced)))

for i, n in enumerate(n_array_aqc_reduced):
    success_probs = adams_quantum_walk_data(n)
    durations = adams_adiabatic_data(n)
    for decile in deciles:
        aqc_decile_success_probs = success_probs[aqc_deciled_indices[i][decile]]
        mean_success_probabilities_aqc_deciles[decile, i] = np.mean(aqc_decile_success_probs)
        median_success_probabilities_aqc_deciles[decile, i] = np.median(aqc_decile_success_probs)

        qw_decile_durations = durations[qw_deciled_indices[i][decile]]
        qw_decile_durations = qw_decile_durations[np.logical_not(np.isnan(qw_decile_durations))]    # remove NaN values
        mean_durations_qw_deciles[decile, i] = np.mean(qw_decile_durations)
        median_durations_qw_deciles[decile, i] = np.median(qw_decile_durations)

        aqc_decile_durations = durations[aqc_deciled_indices[i][decile]]
        aqc_decile_durations = aqc_decile_durations[np.logical_not(np.isnan(aqc_decile_durations))] # remove NaN values
        mean_durations_aqc_deciles[decile, i] = np.mean(aqc_decile_durations)
        median_durations_aqc_deciles[decile, i] = np.median(aqc_decile_durations)

# %%
# plot

from scipy import optimize

plt.figure()

line=lambda x, m, c: (x*m)+c

for decile in deciles:
    y = np.log2(mean_durations_aqc_deciles[decile, :])
    par, cov = optimize.curve_fit(line, n_array_aqc_reduced, y)
    m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
    print(f'EXP {decile+1}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}')
    fdata = np.array([line(x, m[0], c[0]) for x in n_array_aqc_reduced])

    plt.scatter(n_array_aqc_reduced, mean_durations_aqc_deciles[decile, :], color=decile_colors_1[decile])

plt.yscale('log', base=2)
plt.show()

# %%

print(mean_durations_aqc_deciles[5, :])