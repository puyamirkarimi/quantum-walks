import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import binom
from scipy import linalg
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix
from scipy.special import comb
import math
import time


def hypercube(n_dim):
    n = n_dim - 1
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    A = sigma_i(sigma_x, 0, n)

    for i in range(1, n_dim):
        A += sigma_i(sigma_x, i, n)
    return -1 * A


def sigma_i(sigma_x, i, n):
    if i > 0:
        out = np.eye(2)
        for j_before in range(i - 1):
            out = np.kron(out, np.eye(2))
        out = np.kron(out, sigma_x)
    else:
        out = sigma_x
    for j_after in range(n - i):
        out = np.kron(out, np.eye(2))
    return out


def quantum_walk_hypercube(N, H, psi0, timesteps, normalise):
    psiN = expm_multiply(-(1j) * timesteps * H, psi0)

    prob = np.real(np.conj(psiN) * psiN)

    result = np.zeros(N + 1)
    normalise_array = np.zeros(N+1)

    for i, probability in enumerate(prob):
        binary_i = bin(i)
        i_ones = [ones for ones in binary_i[2:] if ones == '1']
        num_ones = len(i_ones)
        result[num_ones] += probability
        if normalise:
            normalise_array[num_ones] += 1

    if normalise:
        result = result/normalise_array

    return result


def run_walks(n, time_limit, gamma, normalise=False):
    P = 2 ** n  # number of positions
    A = hypercube(n)
    marked_state = np.zeros((2 ** n, 2 ** n))
    marked_state[0, 0] = 1
    H = gamma * (A - n * np.eye(2 ** n)) - marked_state  # not sure if marked_node should also be multiplied by gamma

    psi0 = np.ones(P) * (1 / np.sqrt(2 ** n))
    output = np.zeros((time_limit + 1, n + 1))
    for timesteps in range(0, time_limit + 1):
        output[timesteps] = quantum_walk_hypercube(n, H, psi0, timesteps, normalise=normalise)
        print(sum(output[timesteps]))
    return output


def plot_furthest_qubit_prob(timesteps, data):
    plt.figure()
    plt.plot(range(timesteps + 1), data[:, -1])
    plt.xlim(0, timesteps)
    plt.xticks(range(0, timesteps + 1, 5))
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def plot_nearest_qubit_prob(timesteps, data):
    plt.figure()
    plt.plot(range(timesteps + 1), data[:, 0])
    plt.xlim(0, timesteps)
    plt.xticks(range(0, timesteps + 1, 5))
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def plot_prob_heatmap(data, N, timesteps):
    fig, ax = plt.subplots()
    im = ax.imshow(data, interpolation="gaussian", cmap=plt.get_cmap('plasma'))
    ax.set_xlabel("Steps away from origin, x")
    ax.set_ylabel("Time, t", rotation=0, labelpad=18)
    plt.xticks(range(0, N + 1, 2))
    plt.yticks(range(0, timesteps + 1, 3))
    ax.invert_yaxis()

    cb = fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('plasma')))
    cb.set_label('Probability, P(x, t)')

    plt.show()


def optimal_gamma(n):
    N = 2**n
    gam = 0
    for r in range(1, n+1):
        gam += comb(n, r) * (1/r)
    gam = (1/2) * (1/N) * gam
    return gam


def first_max_index(array):
    index = len(array) - 1
    previous_value = array[0] - 1
    for i in range(len(array)):
        if array[i] < previous_value:
            index = i - 1
            break
        previous_value = array[i]
    return index, np.max(array)


if __name__ == '__main__':
    time_start = time.time()
    max_timesteps = [10, 10, 10, 20, 20, 20, 30, 40, 50, 60]
    data = []
    num_dims = 10

    for n in range(1, num_dims+1):
        gamma = optimal_gamma(n)    # hopping rate
        print("gamma:", gamma)
        print("n:", n)
        data.append(run_walks(n, max_timesteps[n-1], gamma))  # 2D array of probability with dims [time, distance]

    time_end = time.time()
    print("runtime:", time_end - time_start)

    peak_times = np.zeros(num_dims)
    peak_probs = np.zeros(num_dims)
    for i in range(num_dims):
        peak_times[i] = first_max_index(data[i][:, 0])[0]
        peak_probs[i] = first_max_index(data[i][:, 0])[1]

    plt.figure()

    # plt.plot(np.arange(num_dims)+1, peak_probs)
    # plt.xlim(1, num_dims)
    # plt.xlabel("Number of qubits, n")
    # plt.ylabel("Maximum probability of marked state, $P_{max}(n)$")

    n_array = np.arange(float(num_dims)) + 1
    for index, num in enumerate(n_array):
        n_array[index] = np.sqrt(2**num)


    plt.plot(n_array, peak_times)
    plt.scatter(n_array, peak_times)
    #plt.xlim(0, n_array[-1])
    # plt.xticks(range(0, timesteps + 1, 5))
    plt.xlabel("$\sqrt{2^{n}}$")
    plt.ylim(0, max_timesteps[-1])
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylabel("Time until first probability maximum of marked state")

    plt.show()
