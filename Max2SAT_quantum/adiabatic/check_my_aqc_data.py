# %%
import numpy as np
from matplotlib import pyplot as plt

def adams_adiabatic_data(n):
    a = np.genfromtxt('./../qw_and_aqc_data/heug.csv', delimiter=',', missing_values='', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 10]
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


def rerun_data(n):
    a = np.genfromtxt('./../aqc_sim_rerun/aqc_times_rerun.csv', delimiter=',', skip_header=1, dtype=str)[(n-5)*10000:(n-4)*10000, 2]
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


# %%
n = 15
data_mine = rerun_data(n)
data_adams = adams_adiabatic_data(n)

# %%

line = np.linspace(10, 5000, 10)
plt.figure()
plt.xlabel("Rerun times")
plt.ylabel("Adam's times")
# plt.xlim([10, 30])
# plt.ylim([10, 30])
plt.scatter(data_mine, data_adams, marker='.', label=f'n={n}')
plt.plot(line, line, color='red')
plt.legend()
plt.show()

