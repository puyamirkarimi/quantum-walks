# %%
# imports and functions

import numpy as np
from matplotlib import pyplot as plt


def adams_adiabatic_data(n):
    a = np.genfromtxt('./heug.csv', delimiter=',', missing_values='', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 10]
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


def adams_adiabatic_data_errors(n):
    a = np.genfromtxt('./heug.csv', delimiter=',', missing_values='', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 11]
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


def predict_errors(sorted_array):
    out = list()
    for i in range(len(sorted_array) - 1):
        out.append((sorted_array[i+1]-sorted_array[i])/2)
    out.append(float('nan'))
    return np.array(out)


# %%

n11_times = adams_adiabatic_data(11)
n11_errors = adams_adiabatic_data_errors(11)

n5_times = adams_adiabatic_data(5)
n5_errors = adams_adiabatic_data_errors(5)

n15_times = adams_adiabatic_data(15)
n15_errors = adams_adiabatic_data_errors(15)

n14_times = adams_adiabatic_data(14)
n14_errors = adams_adiabatic_data_errors(14)


# %%

times = n15_times
errors = n15_errors

rel_errors = errors/times

plt.figure()
plt.scatter(times, errors)
# plt.xscale('log')
plt.show()

# %%

sorted_times = np.unique(np.sort(times))
sorted_errors = np.unique(np.sort(errors))
my_sorted_errors = predict_errors(sorted_times)
my_rel_errors = my_sorted_errors/sorted_times

# %%

plt.figure()
# plt.scatter(sorted_times, sorted_errors)
plt.scatter(sorted_times, my_rel_errors)
# plt.xscale('log')
plt.show()

# %%

print(my_sorted_errors/sorted_times)

# %%

print(sorted_times)

# %%

for n in range(5, 16):
    print(f'n={n}')
    times = adams_adiabatic_data(n)
    sorted_times = np.unique(np.sort(times))
    my_sorted_errors = predict_errors(sorted_times)
    my_rel_errors = my_sorted_errors/sorted_times

    fig, ax = plt.subplots()
    # plt.scatter(sorted_times, sorted_errors)
    plt.scatter(sorted_times, my_rel_errors)
    # plt.xscale('log')
    # yl = ax.get_ylim()
    # max_y = np.min(())
    plt.ylim([0, 0.03])
    plt.show()

# %%
n = 11
times = adams_adiabatic_data(n)
sorted_times = np.unique(np.sort(times))
my_sorted_errors = predict_errors(sorted_times)
my_rel_errors = my_sorted_errors/sorted_times

# %%
print(sorted_times[-1])
# print(my_sorted_errors[-10:])

fig, ax = plt.subplots()
# plt.scatter(sorted_times, sorted_errors)
plt.scatter(sorted_times, my_rel_errors)
# plt.xscale('log')
# yl = ax.get_ylim()
# max_y = np.min(())
plt.ylim([0, 0.03])
plt.ylabel('0.5 * fractional distance to next higher T')
plt.xlabel('T')
plt.show()