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

    ground_state = np.zeros(len(psiN))
    ground_state[0] = 1

    prob_ground = np.abs(np.dot(np.conjugate(ground_state), psiN)) ** 2

    # result = np.zeros(N + 1)
    # normalise_array = np.zeros(N+1)
    #
    # for i, probability in enumerate(prob):
    #     binary_i = bin(i)
    #     i_ones = [ones for ones in binary_i[2:] if ones == '1']
    #     num_ones = len(i_ones)
    #     result[num_ones] += probability
    #     if normalise:
    #         normalise_array[num_ones] += 1
    #
    # if normalise:
    #     result = result/normalise_array

    return prob_ground


def run_many_walks(n, time_limit, gamma, normalise=False):
    P = 2 ** n  # number of positions
    A = hypercube(n)
    marked_state = np.zeros((2 ** n, 2 ** n))
    marked_state[0, 0] = 1
    H = gamma * (A - n * np.eye(2 ** n)) - marked_state  # not sure if marked_node should also be multiplied by gamma

    psi0 = np.ones(P) * (1 / np.sqrt(2 ** n))
    output = np.zeros(time_limit + 1)
    for timestep in range(0, time_limit + 1):
        output[timestep] = quantum_walk_hypercube(n, H, psi0, timestep, normalise=normalise)
        print(timestep)
    return output


def run_many_walks_2(n, timesteps, limit, gamma, normalise=False):
    P = 2 ** n  # number of positions
    A = hypercube(n)
    marked_state = np.zeros((2 ** n, 2 ** n))
    marked_state[0, 0] = 1
    H = gamma * (A - n * np.eye(2 ** n)) - marked_state  # not sure if marked_node should also be multiplied by gamma

    psi0 = np.ones(P) * (1 / np.sqrt(2 ** n))
    output = np.zeros(timesteps + 1)
    for step in range(0, timesteps + 1):
        timestep = (step/timesteps)*np.sqrt(2 ** n)*limit
        output[step] = quantum_walk_hypercube(n, H, psi0, timestep, normalise=normalise)
        print(step)
    return output


def plot_furthest_qubit_prob(timesteps, data):
    plt.figure()
    plt.plot(range(timesteps + 1), data[:, -1])
    plt.xlim(0, timesteps)
    plt.xticks(range(0, timesteps + 1, 5))
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()


def plot_nearest_qubit_prob(ax, timesteps, data, colors):
    for i in range(len(data[:,0])):
        ax.plot(range(timesteps + 1), data[i,:], color=colors[i])
    ax.set_xlim(0, timesteps)
    ax.set_xticks(range(0, timesteps + 1, 20))
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel("$t_f$")
    ax.set_ylabel("$P(t_f)$")


def plot_nearest_qubit_prob_2(ax, time_array, data, colors):
    for i in range(len(data[:,0])):
        ax.plot(time_array, data[i,:], color=colors[i])
    ax.set_xlim(0, time_array[-1])
    # ax.set_xticks(range(0, timesteps + 1, 20))
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$t_f / \sqrt{N}$")


def optimal_gamma(n):
    N = 2**n
    gam = 0
    for r in range(1, n+1):
        gam += comb(n, r) * (1/r)
    gam = (1/2) * (1/N) * gam
    return gam


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (9.6, 4.8)

    time_start = time.time()

    n_list = [5, 9]  # number of dimensions of hypercube
    timesteps = 100
    sqrt_limit = 5.48
    data = np.zeros((len(n_list), timesteps + 1))
    data_2 = np.zeros((len(n_list), timesteps + 1))

    for i, n in enumerate(n_list):
        gamma = optimal_gamma(n)  # hopping rate
        print("n:", n, "gamma:", gamma)
        data[i, :] = run_many_walks(n, timesteps, gamma, normalise=False)  # 2D array of [timesteps_run, probability_at_distance_of_index]

    for i, n in enumerate(n_list):
        gamma = optimal_gamma(n)  # hopping rate
        print("n:", n, "gamma:", gamma)
        data_2[i, :] = run_many_walks_2(n, timesteps, sqrt_limit, gamma, normalise=False)  # 2D array of [timesteps_run, probability_at_distance_of_index]

    time_end = time.time()
    print("runtime:", time_end - time_start)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.tick_params(direction='in', top=True, right=True, which='both')
    ax2.tick_params(direction='in', top=True, right=True, which='both', labelleft=False)
    colors = ['forestgreen', 'red']

    plot_nearest_qubit_prob(ax1, timesteps, data, colors)
    times = np.linspace(0, sqrt_limit, num=101)
    plot_nearest_qubit_prob_2(ax2, times, data_2, colors)
    # plt.show()
    plt.savefig('continuous_time_search.png', dpi=200)
