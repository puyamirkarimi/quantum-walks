import matplotlib.pyplot as plt
import numpy as np


def counts_data_adam(n):
    return np.loadtxt("adam_counts_"+str(n)+".txt").reshape((-1, 10000))


def counts_data_adam_noGT(n):
    return np.loadtxt("adam_noGT_counts_"+str(n)+".txt").reshape((-1, 10000))


def counts_data_adam_noGT_nondg(n):
    return np.loadtxt("adam_noGT_nondg_counts_"+str(n)+".txt").reshape((-1, 10000))


def compare_repeats(data):
    num_repeats = len(data[:, 0])
    num_x_vals = len(data[0, :])
    y_av = np.zeros(num_x_vals)
    y_std_error = np.zeros(num_x_vals)
    y_errors = np.zeros(num_x_vals)

    for x in range(num_x_vals):
        y_av[x] = np.mean(data[:, x])
        y_std_error[x] = np.std(data[:, x], ddof=1) / np.sqrt(num_repeats)

    for x in range(num_x_vals):
        for y in range(num_repeats):
            y_errors[x] += np.abs(data[y, x] - y_av[x])

    return y_av, y_std_error, not np.array_equal(y_errors, np.zeros(num_x_vals))


if __name__ == '__main__':
    av, std_err, err_exist = compare_repeats(counts_data_adam_noGT(20))
    print(err_exist)