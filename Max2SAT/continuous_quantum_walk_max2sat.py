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


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances_original/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


def first_eigv(A):
    """returns ground state of matrix A"""
    return np.linalg.eigh(A)[1][:, 0]


def hypercube(n_dim):
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    A = sigma_i(sigma_x, 0, n)

    for i in range(1, n_dim):
        A += sigma_i(sigma_x, i, n_dim)
    return -1 * A


def sigma_i(sigma, i, n_dim):
    n = n_dim -1            # because of i starting from 0 rather than 1
    if i > 0:
        out = np.eye(2)
        for j_before in range(i - 1):
            out = np.kron(out, np.eye(2))
        out = np.kron(out, sigma)
    else:
        out = sigma
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


def hamiltonian_2sat(n, formula):
    N = 2 ** n
    out = np.zeros((N, N))
    sigma_z = np.array([[1, 0],
                        [0, -1]])
    sigma_identity = np.eye(N)
    sigma_z_i = np.zeros((n, N, N))
    for i in range(n):
        sigma_z_i[i] = sigma_i(sigma_z, i, n)
    for clause in formula:
        v_1 = clause[1]
        v_2 = clause[3]
        sign_1 = -1 * clause[0]                 # -1 because signs should be opposite in Hamiltonian
        sign_2 = -1 * clause[2]
        out += (1/4) * (sign_1*sign_2*sigma_z_i[v_1]*sigma_z_i[v_2]
                        + sign_1*sigma_z_i[v_1] + sign_2*sigma_z_i[v_2] + sigma_identity)
    print(first_eigv(out))
    return out


def run_many_walks(formula, n, time_limit, gamma, normalise=False):
    N = 2 ** n  # number of positions
    A = hypercube(n)
    H_problem = hamiltonian_2sat(n, formula)
    H = gamma * (A - n * np.eye(2 ** n)) + H_problem

    psi0 = np.ones(N) * (1 / np.sqrt(N))
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
    plt.xlabel("Time, $t$")
    plt.show()


def plot_nearest_qubit_prob(timesteps, data):
    plt.figure()
    plt.plot(range(timesteps + 1), data[:, 0])
    plt.xlim(0, timesteps)
    plt.xticks(range(0, timesteps + 1, 5))
    plt.ylim(0, 1)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Time, $t$")
    plt.ylabel("Probability of being in ground state of $H_{prob}$")
    plt.show()


def plot_prob_heatmap(data, N, timesteps):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=plt.get_cmap('plasma'))
    ax.set_xlabel("Hamming distance, $d$")
    ax.set_ylabel("Time, $t$", rotation=0, labelpad=25)
    plt.xticks(range(0, N + 1, 2))
    plt.yticks(range(0, timesteps + 1, 5))
    ax.invert_yaxis()

    cb = fig.colorbar(cm.ScalarMappable(cmap=plt.get_cmap('plasma')))
    cb.set_label('Normalised probability, $P(d, t)$')

    plt.show()


def optimal_gamma(n):
    N = 2**n
    gam = 0
    for r in range(1, n+1):
        gam += comb(n, r) * (1/r)
    gam = (1/2) * (1/N) * gam
    return gam


if __name__ == '__main__':
    instance_names, instance_n_bits = get_instances()
    instance_name = instance_names[0]
    sat_formula = get_2sat_formula(instance_name)
    n = instance_n_bits[0]                   # number of variables/qubits

    timesteps = 30
    gamma = optimal_gamma(n)    # hopping rate
    print("gamma:", gamma)

    time_start = time.time()
    data = run_many_walks(sat_formula, n, timesteps, gamma, normalise=True)  # 2D array of [timesteps_run, probability_at_distance_of_index]
    time_end = time.time()

    print("runtime:", time_end - time_start)

    #plot_furthest_qubit_prob(timesteps, data)
    plot_nearest_qubit_prob(timesteps, data)
    plot_prob_heatmap(data, n, timesteps)
