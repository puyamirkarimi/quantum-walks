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
from scipy.optimize import minimize


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances_original/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


def first_eig_vec(A):
    """returns ground state of matrix A"""
    return np.linalg.eigh(A)[1][:, 0]


def eig_vec(A, i):
    """returns ith eigenvector of matrix A, ordered by increasing eigenvalue"""
    return np.linalg.eigh(A)[1][:, i]


def eig_vecs(A):
    """returns all eigenvectors of matrix A, ordered by increasing eigenvalue"""
    return np.linalg.eigh(A)[1]


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
    return out


def inner_product(psi1, psi2):
    return np.dot(np.conjugate(psi1), psi2)


def inner_product_sq(psi1, psi2):
    return np.abs(np.dot(np.conjugate(psi1), psi2)) ** 2


def inf_time_av_prob(N, H_prob, H_tot, psi_0):
    out = 0
    ground_state = first_eig_vec(H_prob)
    eig_vectors = eig_vecs(H_tot)
    for a in range(N):
        #a_state = eig_vec(H_tot, a)
        a_state = eig_vectors[:, a]
        out += inner_product_sq(ground_state, a_state) * inner_product_sq(a_state, psi_0)
    return out


def opt_func(gamma, N, H_prob, H_lap, psi_0):
    """ args = (N, H_problem, H_lap, psi_0) """
    H_tot = gamma * H_lap + H_prob
    out = 0
    ground_state = first_eig_vec(H_prob)
    eig_vectors = eig_vecs(H_tot)
    for a in range(N):
        #a_state = eig_vec(H_tot, a)
        a_state = eig_vectors[:, a]
        out += inner_product_sq(ground_state, a_state) * inner_product_sq(a_state, psi_0)
    return -1 * out
    # return -1 * inf_time_av_prob(args[0], args[1], H_tot, args[3])


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


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    plt.rc('font', size=14)

    instance_names, instance_n_bits = get_instances()
    n_list = [5, 10]
    colors = ['forestgreen', 'red']
    linestyles = ['solid', 'dashed']

    start = 630
    end = start+2

    instances_per_n = end - start

    gamma_step = 0.02
    gamma_limit = 2
    gammas = np.arange(0, gamma_limit+gamma_step, gamma_step)
    num_gammas = len(gammas)

    probs = np.zeros((len(n_list), instances_per_n, num_gammas))    # probability[n, instance, gamma]
    heur_gammas = np.zeros(len(n_list))

    for n_index, n in enumerate(n_list):
        N = 2 ** n
        n_shifted = n - 5
        A = hypercube(n)
        psi_0 = np.ones(N) * (1 / np.sqrt(N))
        heur_gammas[n_index] = heuristic_gamma(n)

        H_lap = A - n * np.eye(2 ** n)

        for loop, i in enumerate(range(n_shifted*10000+start, n_shifted*10000+end)):
            instance_name = instance_names[i]
            sat_formula = get_2sat_formula(instance_name)
            H_problem = hamiltonian_2sat(n, sat_formula)
            for j, gamma in enumerate(gammas):
                H_total = gamma * H_lap + H_problem
                probs[n_index, loop, j] = inf_time_av_prob(N, H_problem, H_total, psi_0)
                print(j)
            print("loop:", loop)

    plt.figure()
    for n_index in range(len(n_list)):
        for instance_i in range(instances_per_n):
            plt.plot(gammas, probs[n_index, instance_i, :], color=colors[n_index], linestyle=linestyles[instance_i])
        plt.vlines(heur_gammas[n_index], 0, 0.3, colors=colors[n_index], linestyles='dotted')
    plt.xlim([np.min(gammas), np.max(gammas)])
    plt.ylim([0, 0.3])
    plt.xlabel("$\gamma$")
    plt.ylabel("$P_\infty$")
    plt.savefig('p_infty_gamma_n_5_10.png', dpi=200)
    # plt.show()
