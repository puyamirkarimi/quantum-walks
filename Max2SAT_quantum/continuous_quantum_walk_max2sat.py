import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
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


def first_eigvec(A):
    """returns ground state of matrix A"""
    return np.linalg.eigh(A)[1][:, 0]


def hypercube(n_dim):
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    A = sigma_i(sigma_x, 0, n_dim)

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
    return out


def run_many_walks(formula, n, time_limit, normalise=False):
    N = 2 ** n  # number of positions
    A = hypercube(n)
    H_problem = hamiltonian_2sat(n, formula)
    gamma = heuristic_gamma(n)
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
    plt.xlabel("Time, $t$", labelpad=10)
    plt.show()


def plot_nearest_qubit_prob(timesteps, data):
    plt.figure()
    plt.plot(range(timesteps + 1), data[:, 0])
    plt.xlim(0, timesteps)
    plt.xticks(range(0, timesteps + 1, 5))
    plt.ylim(0, 0.2)
    plt.yticks([0, 0.05, 0.1, 0.15])
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("$t$")
    plt.ylabel("$P(t_f)$")
    plt.tight_layout()
    plt.show()


def plot_nearest_qubit_prob_2(timesteps, data, ax, color):
    ax.plot(range(timesteps + 1), data[:, 0], color=color)
    ax.set_xlim(0, timesteps)
    ax.set_xticks(range(0, timesteps + 1, 10))
    ax.set_ylim(0, 0.6)
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel("$t_f$")


def plot_prob_heatmap(data, N, timesteps):
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=plt.get_cmap('plasma'), norm=colors.Normalize(vmin=0, vmax=0.2))
    ax.set_xlabel("Hamming distance, $d$")
    ax.set_ylabel("Time, $t$", rotation=0, labelpad=25)
    plt.xticks(range(0, N + 1, 2))
    plt.yticks(range(0, timesteps + 1, 5))
    ax.invert_yaxis()

    m = cm.ScalarMappable(cmap=plt.get_cmap('plasma'))
    m.set_clim(0., 0.2)
    cb = fig.colorbar(m, extend='max')
    cb.set_label('Normalised probability, $P(d, t)$')
    plt.tight_layout()
    plt.show()


def optimal_gamma(n):
    return 0.77


def heuristic_gamma(n):
    out = "haven't defined heuristic gamma for given n"
    if n == 5:
        out = 0.56503
    if n == 6:
        out = 0.587375
    if n == 7:
        out = 0.5984357142857143
    if n == 8:
        out = 0.60751875
    if n == 9:
        out = 0.6139833333333333
    if n == 10:
        out = 0.619345
    if n == 11:
        out = 0.6220136363636364
    print("heuristic gamma: ", out)
    return out


if __name__ == '__main__':
    plt.rc('text', usetex=True)
    plt.rc('font', size=16)
    plt.rcParams["figure.figsize"] = (9.6, 4.8)

    fig, axs = plt.subplots(1, 2)
    colors = ['forestgreen', 'deeppink']
    axs[0].tick_params(direction='in', top=True, right=True, which='both')
    axs[1].tick_params(direction='in', top=True, right=True, which='both', labelleft=False)
    axs[0].set_ylabel("$P(t_f)$")
    axs[0].set_yticks(np.arange(0.1, 0.7, 0.1))

    timesteps = 50
    instance_names, instance_n_bits = get_instances()
    instance_nums = [1, 50031] #50031
    for i, instance_num in enumerate(instance_nums):
        # instance_name = instance_names[32070]
        instance_name = instance_names[instance_num]
        sat_formula = get_2sat_formula(instance_name)
        n = instance_n_bits[instance_num]                   # number of variables/qubits
        print("n:", n)

        time_start = time.time()
        data = run_many_walks(sat_formula, n, timesteps, normalise=True)  # 2D array of [timesteps_run, probability_at_distance_of_index]
        time_end = time.time()

        print("runtime:", time_end - time_start)

        plot_nearest_qubit_prob_2(timesteps, data, axs[i], colors[i])
        # plot_prob_heatmap(data, n, timesteps)

    # plt.savefig('inst_success_probs_n_5_10.png', dpi=200)
    plt.show()