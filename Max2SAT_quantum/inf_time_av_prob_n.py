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


def first_eig_vec(A):
    """returns ground state of matrix A"""
    return np.linalg.eigh(A)[1][:, 0]


def eig_vec(A, i):
    """returns ith eigenvector of matrix A, ordered by increasing eigenvalue"""
    return np.linalg.eigh(A)[1][:, i]


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


def inf_time_av_prob(gamma, N, H_prob, H_tot, psi_0):
    out = 0
    ground_state = first_eig_vec(H_prob)
    for a in range(N):
        a_state = eig_vec(H_tot, a)
        out += inner_product_sq(ground_state, a_state) * inner_product_sq(a_state, psi_0)
    return out


def optimal_gamma(n):
    N = 2**n
    gam = 0
    for r in range(1, n+1):
        gam += comb(n, r) * (1/r)
    gam = (1/2) * (1/N) * gam
    return gam


if __name__ == "__main__":
    plt.rc('text', usetex=True)
    #plt.rc('font', size=16)

    instance_names, instance_n_bits = get_instances()
    instance_num = 38000
    instance_name = instance_names[instance_num]
    sat_formula = get_2sat_formula(instance_name)
    n = instance_n_bits[instance_num]  # number of variables/qubits
    print("n:", n)
    N = 2**n
    H_problem = hamiltonian_2sat(n, sat_formula)
    A = hypercube(n)
    psi_0 = np.ones(N) * (1 / np.sqrt(N))

    probs = []
    for i in range(0, 100):
        print(i)
        gamma = i / 50
        H_total = gamma * (A - n * np.eye(2 ** n)) + H_problem

        probs.append(inf_time_av_prob(gamma, N, H_problem, H_total, psi_0))

    plt.figure()
    plt.plot(np.arange(0, 100)/50, np.array(probs))
    plt.xlim([0, 2])
    plt.ylim([0, 0.2])
    plt.xlabel("$\gamma$")
    plt.ylabel("$P_{\infty}$")
    plt.show()
