# %%
# imports

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

import matplotlib.transforms as mtransforms

# %%
# initialisations

plt.rc('text', usetex=True)
plt.rc('font', size=14)

blue = '#0072B2'
orange = '#EF6900'
green = '#009E73'

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
    success_probs = adams_quantum_walk_data(n)
    instance_names = get_instance_names(n)
    instance = int(frac * 10000 - 1)
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
    print(
        f'Getting the hardest {frac} fraction of formulae of size n={n} for AQC')
    durations = rerun_adiabatic_data(n)
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
    print(
        f'Getting the easiest {frac} fraction of formulae of size n={n} for AQC')
    durations = rerun_adiabatic_data(n)
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


def get_hardest_boundary_formula_aqc(n, frac, return_index=False):
    durations = rerun_adiabatic_data(n)
    durations = np.nan_to_num(durations, nan=np.max(durations)+1.0)
    instance_names = get_instance_names(n)
    instance = int(frac * 10000 - 1)
    boundary_index = np.argsort(durations)[9999-instance]
    if return_index:
        return boundary_index
    boundary_instance_name = instance_names[boundary_index]
    return get_2sat_formula(boundary_instance_name)


def get_deciled_formulae_aqc(n, return_indices=False):
    '''returns instances of a given n organised by QW decile'''
    print(f'Getting the formulae of size n={n} organised by QW decile')
    durations = rerun_adiabatic_data(n)
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
    durations = rerun_adiabatic_data(n)
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


def adams_mixbnb_data(n):
    costs = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/mixbnb.csv', delimiter=',',
                          skip_header=1+(n-5)*10000, usecols=2, max_rows=10000, dtype=str).astype(float)
    mix_iters = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/mixbnb.csv', delimiter=',',
                              skip_header=1+(n-5)*10000, usecols=4, max_rows=10000, dtype=str).astype(float)
    n_calls = costs + mix_iters
    return n_calls


def adams_mixbnb_data_crosson():
    costs = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/mixbnb_crosson.csv',
                          delimiter=',', skip_header=1, usecols=2, dtype=str).astype(float)
    mix_iters = np.genfromtxt('./../Max2SAT_quantum/qw_and_aqc_data/mixbnb_crosson.csv',
                              delimiter=',', skip_header=1, usecols=4, dtype=str).astype(float)
    n_calls = costs + mix_iters
    return n_calls


# %%
# plot MIXBnB calls against QW probability and AQC duration
from matplotlib.colors import LogNorm

n = 15

fig = plt.figure(figsize=(6, 4.75))
axs = []
# gs1 = gridspec.GridSpec(2, 6, width_ratios=[1, 0.04, 0.05, 1, 0.04, 0.05])
# gs1.update(wspace=0.25)
# gs1.update(hspace=0.45)
gs1 = gridspec.GridSpec(2, 6, width_ratios=[1, 0.04, 0.25, 1, 0.04, 0.0])
gs1.update(wspace=0.25)
gs1.update(hspace=0.45)

# plot MIXBnB against QW
axs.append(plt.subplot(gs1[0]))

bnb = adams_mixbnb_data(n)
qw = adams_quantum_walk_data(n)

hex = plt.hexbin(np.log10(qw), np.log10(bnb), gridsize=30, cmap='Blues', vmax=238, linewidths=0.05)
vals = hex.get_array()
print('max:', np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=13, size=5)
plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
plt.ylabel(r'$N_\mathrm{calls}$', fontsize=15)
xt = np.arange(-2.5, -1, 0.5)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
yt = np.arange(2, 5, 1)
ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[-1].text(0.0, 0.26, '(a)', transform=axs[-1].transAxes + trans, verticalalignment='top', fontsize=15)

xr_tmp, yr_tmp = np.argsort(np.log2(qw)), np.argsort(np.log2(bnb))
xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
xr[xr_tmp], yr[yr_tmp] = np.arange(len(qw)), np.arange(len(bnb))
covr = np.cov(xr, yr)
sr = covr[1, 0]/(np.std(xr)*np.std(yr))

line = lambda x, m, c: (x*m)+c
par, cov = curve_fit(line, np.log2(qw), np.log2(bnb))
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(qw)])

# plot MIXBnB against QW logarithmic
axs.append(plt.subplot(gs1[3]))

bnb = adams_mixbnb_data(n)
qw = adams_quantum_walk_data(n)

hex = plt.hexbin(np.log10(qw), np.log10(
    bnb), gridsize=30, cmap='Blues', norm=LogNorm(vmax=239), linewidths=0.05)
vals = hex.get_array()
print('max:', np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=13, size=5)
plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
xt = np.arange(-2.5, -1, 0.5)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
yt = np.arange(2, 5, 1)
# ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, [], fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[-1].text(0.0, 0.26, '(b)', transform=axs[-1].transAxes + trans, verticalalignment='top', fontsize=15)

# plot MIXBnB against AQC
axs.append(plt.subplot(gs1[6]))

bnb = adams_mixbnb_data(n)
aqc = rerun_adiabatic_data(n)

bnb, aqc = bnb[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

hex = plt.hexbin(np.log10(aqc), np.log10(bnb), gridsize=30, cmap='Blues', vmax=238, linewidths=0.05)
vals = hex.get_array()
print('max:', np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

plt.xlabel(r'$t_{0.99}$', fontsize=15)
plt.ylabel(r'$N_\mathrm{calls}$', fontsize=15)
xt = np.arange(1.5, 4.5, 1)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
yt = np.arange(2, 5, 1)
ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
x0, x1, y0, y1 = plt.axis()
plt.axis((x0,x1,y0,y1))

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[-1].text(0.0, 0.97, '(c)', transform=axs[-1].transAxes + trans, verticalalignment='top', fontsize=15)

xr_tmp, yr_tmp = np.argsort(np.log2(aqc)), np.argsort(np.log2(bnb))
xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
xr[xr_tmp], yr[yr_tmp] = np.arange(len(aqc)), np.arange(len(bnb))
covr = np.cov(xr, yr)
sr = covr[1, 0]/(np.std(xr)*np.std(yr))

line = lambda x, m, c: (x*m)+c
par, cov = curve_fit(line, np.log2(aqc), np.log2(bnb))
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(aqc)])

ax = fig.add_subplot(gs1[:, 1])
cb = plt.colorbar(hex, cax=ax, use_gridspec=True)
cb.ax.tick_params(labelsize=13)
cb.set_label(r'Number of MAX 2-SAT instances', fontsize=15)

# plot MIXBnB against AQC
axs.append(plt.subplot(gs1[9]))

bnb = adams_mixbnb_data(n)
aqc = rerun_adiabatic_data(n)

bnb, aqc = bnb[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

hex = plt.hexbin(np.log10(aqc), np.log10(
    bnb), gridsize=30, cmap='Blues', norm=LogNorm(vmax=239), linewidths=0.05)
vals = hex.get_array()
print('max:', np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

plt.xlabel(r'$t_{0.99}$', fontsize=15)
# plt.ylabel(r'$N_\mathrm{calls}$', fontsize=15)
xt = np.arange(1.5, 4.5, 1)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
yt = np.arange(2, 5, 1)
ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, [], fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[-1].text(0.0, 0.97, '(d)', transform=axs[-1].transAxes + trans, verticalalignment='top', fontsize=15)

ax = fig.add_subplot(gs1[:, 4])
cb = plt.colorbar(hex, cax=ax, use_gridspec=True)
cb.ax.tick_params(labelsize=13)
cb.set_label(r'Number of MAX 2-SAT instances $+1$', fontsize=15)

# gs1.tight_layout(fig)
# plt.savefig('aqcqwmixbnb_hexbins_windows.pdf', bbox_inches='tight')
plt.show()

# %%
# plot AQC duration against QW success probability for n = 15

n = 15
qw = adams_quantum_walk_data(n)
aqc = rerun_adiabatic_data(n)

qw, aqc = qw[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

fig = plt.figure(figsize=(8, 5.5))
plt.subplot()

hex = plt.hexbin(np.log10(qw), np.log10(aqc), gridsize=50, cmap='Blues')
vals = hex.get_array()
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

cb = plt.colorbar()
cb.ax.tick_params(labelsize=13, size=5)
plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
plt.ylabel(r'$t_{0.99}$', fontsize=15)
xt = np.arange(-2.5, -1, 0.5)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
yt = np.arange(1.5, 4, 0.5)
ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in', size=5)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# xr_tmp, yr_tmp = np.argsort(np.log2(x)), np.argsort(np.log2(y))
# xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
# xr[xr_tmp], yr[yr_tmp] = np.arange(len(x)), np.arange(len(y))
# covr = np.cov(xr, yr)
# sr = covr[1, 0]/(np.std(xr)*np.std(yr))

# line = lambda x, m, c: (x*m)+c
# par, cov = spo.curve_fit(line, np.log2(x), np.log2(y))
# m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
# print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
# fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(x)])

# plt.figure(figsize=(8, 5))
# plt.hexbin(np.log2(x), np.log2(y), gridsize=50, cmap='Blues')
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=13, size=5)
# plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
# plt.ylabel(r'$t_{0.99}$', fontsize=15)
# xt = np.arange(-9, -4, 1)
# xtl = ['$2^{' + f'{x}' + '}$' for x in xt]
# plt.xticks(xt, xtl, fontsize=13)
# yt = np.arange(5, 13, 1)
# ytl = ['$2^{' + f'{y}' + '}$' for y in yt]
# plt.yticks(yt, ytl, fontsize=13)
# plt.tick_params(direction='in', size=5)
# plt.xlim(-9.5, -4.8)
# plt.ylim(4.55, 12)

plt.tight_layout()
plt.show()

# %%
# plot AQC duration against QW success probability for n = 5, 15
# vertical

# fig = plt.figure(figsize=(4.3, 5.9))
fig = plt.figure(figsize=(6, 9.3))
axs = []
gs1 = gridspec.GridSpec(2, 2, width_ratios=[1, 0.04])
# gs1.update(wspace=0.25)

# n = 5 plot
axs.append(plt.subplot(gs1[0, 0]))
n = 5

qw = adams_quantum_walk_data(n)
aqc = rerun_adiabatic_data(n)

qw, aqc = qw[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

hex = plt.hexbin(np.log10(qw), np.log10(aqc), gridsize=30, vmin=0, vmax=181, cmap='Blues', linewidths=0.2)
vals = hex.get_array()
print("max:", np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

# plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
plt.ylabel(r'$t_{0.99}$', fontsize=15)
xt = np.arange(-0.8, -0.4, 0.1)
xtl = ['$10^{' + f'{np.round(x,3)}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
axs[0].tick_params(axis='x', which='major', pad=7)
yt = np.arange(1.2, 2.4, 0.4)
ytl = ['$10^{' + f'{np.round(y,3)}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

xr_tmp, yr_tmp = np.argsort(np.log2(qw)), np.argsort(np.log2(aqc))
xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
xr[xr_tmp], yr[yr_tmp] = np.arange(len(qw)), np.arange(len(aqc))
covr = np.cov(xr, yr)
sr = covr[1, 0]/(np.std(xr)*np.std(yr))

line = lambda x, m, c: (x*m)+c
par, cov = curve_fit(line, np.log2(qw), np.log2(aqc))
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(qw)])

# n = 15 plot
axs.append(plt.subplot(gs1[1, 0]))
n = 15

qw = adams_quantum_walk_data(n)
aqc = rerun_adiabatic_data(n)

qw, aqc = qw[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

hex = plt.hexbin(np.log10(qw), np.log10(aqc), gridsize=30, vmin=0, vmax=181, cmap='Blues', linewidths=0.2)
vals = hex.get_array()
print("max:", np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
plt.ylabel(r'$t_{0.99}$', fontsize=15)
xt = np.arange(-2.5, -1, 0.5)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
axs[1].tick_params(axis='x', which='major', pad=7)
yt = np.arange(1.5, 4, 0.5)
ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

ax = fig.add_subplot(gs1[:, 1])
# plt.axis('off')
cb = plt.colorbar(hex, cax=ax, use_gridspec=True)
cb.ax.tick_params(labelsize=13)
cb.set_label(r'Number of MAX 2-SAT instances', fontsize=15)
# pos = ax.get_position()
# points = pos.get_points()
# print(points)
# pos.set_points(points)
# ax.set_position(pos)

xr_tmp, yr_tmp = np.argsort(np.log2(qw)), np.argsort(np.log2(aqc))
xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
xr[xr_tmp], yr[yr_tmp] = np.arange(len(qw)), np.arange(len(aqc))
covr = np.cov(xr, yr)
sr = covr[1, 0]/(np.std(xr)*np.std(yr))

line = lambda x, m, c: (x*m)+c
par, cov = curve_fit(line, np.log2(qw), np.log2(aqc))
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(qw)])

# w=1.4
# for ax in axs:
#     ax.spines['top'].set_linewidth(w)
#     ax.spines['right'].set_linewidth(w)
#     ax.spines['bottom'].set_linewidth(w)
#     ax.spines['left'].set_linewidth(w)

plt.tight_layout()
# plt.savefig('aqcqw_hexbin_windows.pdf', bbox_inches='tight')
plt.show()

# %%
# plot AQC duration against QW success probability for n = 5, 15
# horizontal

# fig = plt.figure(figsize=(4.3, 5.9))
fig = plt.figure(figsize=(6, 3.8))
axs = []
gs1 = gridspec.GridSpec(2, 2, height_ratios=[1, 0.04])
# gs1.update(wspace=0.25)

# n = 5 plot
axs.append(plt.subplot(gs1[0, 0]))
n = 5

qw = adams_quantum_walk_data(n)
aqc = rerun_adiabatic_data(n)

qw, aqc = qw[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

hex = plt.hexbin(np.log10(qw), np.log10(aqc), gridsize=30, vmin=0, vmax=181, cmap='Blues', linewidths=0.2)
vals = hex.get_array()
print("max:", np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
plt.ylabel(r'$t_{0.99}$', fontsize=15)
xt = np.arange(-0.8, -0.4, 0.1)
xtl = ['$10^{' + f'{np.round(x,3)}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
axs[0].tick_params(axis='x', which='major', pad=7)
yt = np.arange(1.2, 2.4, 0.4)
ytl = ['$10^{' + f'{np.round(y,3)}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[-1].text(0.0, 0.2, '(a)', transform=axs[-1].transAxes + trans, verticalalignment='top', fontsize=15)

xr_tmp, yr_tmp = np.argsort(np.log2(qw)), np.argsort(np.log2(aqc))
xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
xr[xr_tmp], yr[yr_tmp] = np.arange(len(qw)), np.arange(len(aqc))
covr = np.cov(xr, yr)
sr = covr[1, 0]/(np.std(xr)*np.std(yr))

line = lambda x, m, c: (x*m)+c
par, cov = curve_fit(line, np.log2(qw), np.log2(aqc))
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(qw)])

# n = 15 plot
axs.append(plt.subplot(gs1[0, 1]))
n = 15

qw = adams_quantum_walk_data(n)
aqc = rerun_adiabatic_data(n)

qw, aqc = qw[~np.isnan(aqc)], aqc[~np.isnan(aqc)]

hex = plt.hexbin(np.log10(qw), np.log10(aqc), gridsize=30, vmin=0, vmax=181, cmap='Blues', linewidths=0.2)
vals = hex.get_array()
print("max:", np.max(vals))
centres = hex.get_offsets()
x_min, x_max = np.min(centres[:, 0]), np.max(centres[:, 0])
y_min, y_max = np.min(centres[:, 1]), np.max(centres[:, 1])

plt.xlabel(r'$\overline{P}(0, 100)$', fontsize=15)
# plt.ylabel(r'$t_{0.99}$', fontsize=15)
xt = np.arange(-2.5, -1, 0.5)
xtl = ['$10^{' + f'{x}' + '}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
axs[1].tick_params(axis='x', which='major', pad=7)
yt = np.arange(1.5, 4, 0.5)
ytl = ['$10^{' + f'{y}' + '}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
axs[-1].text(0.0, 0.2, '(b)', transform=axs[-1].transAxes + trans, verticalalignment='top', fontsize='15')

ax = fig.add_subplot(gs1[1, :])
# plt.axis('off')
cb = plt.colorbar(hex, cax=ax, use_gridspec=True, orientation='horizontal')
cb.ax.tick_params(labelsize=13)
cb.set_label(r'Number of MAX 2-SAT instances', fontsize=15)
# pos = ax.get_position()
# points = pos.get_points()
# print(points)
# pos.set_points(points)
# ax.set_position(pos)

xr_tmp, yr_tmp = np.argsort(np.log2(qw)), np.argsort(np.log2(aqc))
xr, yr = np.empty_like(xr_tmp), np.empty_like(yr_tmp)
xr[xr_tmp], yr[yr_tmp] = np.arange(len(qw)), np.arange(len(aqc))
covr = np.cov(xr, yr)
sr = covr[1, 0]/(np.std(xr)*np.std(yr))

line = lambda x, m, c: (x*m)+c
par, cov = curve_fit(line, np.log2(qw), np.log2(aqc))
m, c = (par[0], np.sqrt(cov[0, 0])), (par[1], np.sqrt(cov[1, 1]))
print(f'n={n}: m={m[0]}pm{m[1]}, c={c[0]}pm{c[1]}, SR={sr}')
fy = np.array([line(xval, m[0], c[0]) for xval in np.log2(qw)])

# w=1.4
# for ax in axs:
#     ax.spines['top'].set_linewidth(w)
#     ax.spines['right'].set_linewidth(w)
#     ax.spines['bottom'].set_linewidth(w)
#     ax.spines['left'].set_linewidth(w)

fig.tight_layout()
# plt.savefig('aqcqw_hexbin_windows.pdf', bbox_inches='tight')
plt.show()

# %%
# probability density histogram of MIXBnB calls (plotting the log on a linear x-axis)
# I think Adam's way of doing this means the probability density is changed and depends on the base of the exponent?

n = 20
bnb = adams_mixbnb_data(n)
bnb_crosson = adams_mixbnb_data_crosson()

fig = plt.figure(figsize=(6, 4.2))
plt.subplot()

h, b = np.histogram(np.log10(bnb), bins='auto', density=True)
db = b[1:]-b[:-1]
b = (b[1:]+b[:-1])/2
htot = np.dot(h, db)
h = (h/htot)
plt.bar(b, h, width=db*1.0, alpha=0.75,
        color=blue)

h, b = np.histogram(np.log10(bnb_crosson), bins='auto', density=True)
db = b[1:]-b[:-1]
b = (b[1:]+b[:-1])/2
htot = np.dot(h, db)
h = (h/htot)
plt.bar(b, h, width=db*1.0, alpha=0.75, color=orange)

plt.xlabel('$\log_{10}(N_\mathrm{calls})$', fontsize=15)
plt.ylabel('$p(\log_{10}(N_\mathrm{calls}))$', fontsize=15)
xt = np.arange(2, 5, 1)
xtl = [f'${x}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
plt.xlim(1.7, 4.8)
yt = np.arange(0, 2.5, 0.5)
ytl = [f'${y:.1f}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in', size=4)
fig.tight_layout()

# plt.savefig('mixbnb20hist_windows.pdf', dpi=200)
plt.show()

# %%
# probabilitiy histogram of MIXBnB calls (plotting the log on a linear x-axis)

n = 20
bnb = adams_mixbnb_data(n)
bnb_crosson = adams_mixbnb_data_crosson()

fig = plt.figure(figsize=(6, 4.125))
plt.subplot()

h, b = np.histogram(np.log10(bnb), bins='auto')
db = b[1:]-b[:-1]
b = (b[1:]+b[:-1])/2
htot = np.dot(h, db)
# h = (h/htot)
h = h/10000
plt.bar(b, h, width=db*1.0, alpha=0.75,
        color='green')

h, b = np.histogram(np.log10(bnb_crosson), bins='auto')
db = b[1:]-b[:-1]
b = (b[1:]+b[:-1])/2
htot = np.dot(h, db)
# h = (h/htot)
h = h/127
plt.bar(b, h, width=db*1.0, alpha=0.75, color='red')

plt.xlabel('$\log_{10}(N_\mathrm{calls})$', fontsize=20)
plt.ylabel('$P(\log_{10}(N_\mathrm{calls}))$', fontsize=20)
xt = np.arange(2, 5, 1)
xtl = [f'${x}$' for x in xt]
plt.xticks(xt, xtl, fontsize=13)
plt.xlim(1.7, 4.8)
yt = np.arange(0, 0.35, 0.05)
ytl = [f'${y:.1f}$' for y in yt]
plt.yticks(yt, ytl, fontsize=13)
plt.tick_params(direction='in', size=5)
fig.tight_layout()

plt.show()

# %%
# histogram of MIXBnB calls (plotting on a logarithmic x-axis)

n = 20

bnb = adams_mixbnb_data(n)
bnb_crosson = adams_mixbnb_data_crosson()

min_calls = np.min(np.append(bnb, bnb_crosson))
max_calls = np.max(np.append(bnb, bnb_crosson))

num_bins = 50
x = np.ones(num_bins+1) * min_calls
multiply_factor = 10**((np.log10(max_calls)-np.log10(min_calls))/num_bins)
x = [x[i] * multiply_factor**i for i in range(len(x))]
x[-1] += 1

num_bins_crosson = 15
x_crosson = np.ones(num_bins_crosson+1) * min_calls
multiply_factor = 10**((np.log10(max_calls) -
                       np.log10(min_calls))/num_bins_crosson)
x_crosson = [x_crosson[i] * multiply_factor**i for i in range(len(x_crosson))]
x_crosson[-1] += 1

fig = plt.figure(figsize=(8, 5.5))
plt.subplot()
plt.hist(bnb, x, alpha=0.75, color='green', density=True)
plt.hist(bnb_crosson, x_crosson, alpha=0.75, color='red', density=True)
plt.xscale('log')

plt.xlabel('$N_\mathrm{calls}$', fontsize=20)
plt.ylabel('$p(N_\mathrm{calls})$', fontsize=20)

plt.tight_layout()
plt.show()
