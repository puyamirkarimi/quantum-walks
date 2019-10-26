import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import binom
from scipy import linalg
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix
from scipy.special import comb
import math


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


def quantum_walk_hypercube(N, timesteps, gamma, normalise):
    P = 2**N  # number of positions

    A = hypercube(N)
    marked_node = np.zeros((2**N, 2**N))
    marked_node[0, 0] = -1
    H = gamma * (A - N * np.eye(2 ** N)) + marked_node      # not sure if marked_node should also be multiplied by gamma

    psi0 = np.ones(P) * (1/np.sqrt(2**N))

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


def run_many_walks(N, time_limit, gamma, normalise=True):
    output = np.zeros((time_limit + 1, N + 1))
    for timesteps in range(0, time_limit + 1):
        output[timesteps] = quantum_walk_hypercube(N, timesteps, gamma, normalise)
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


if __name__ == '__main__':
    n = 8  # number of dimensions of hypercube
    timesteps = 40
    gamma = optimal_gamma(n)    # hopping rate
    print("gamma:", gamma)

    data = run_many_walks(n, timesteps, gamma)  # 2D array of [timesteps_run, probability_at_distance_of_index]

    #plot_furthest_qubit_prob(timesteps, data)
    plot_nearest_qubit_prob(timesteps, data)
    plot_prob_heatmap(data, n, timesteps)
