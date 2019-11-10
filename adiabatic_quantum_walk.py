import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix
from scipy.special import comb
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


def quantum_walk(n, T, M, normalise=True):
    psi0 = np.ones(2**n) * (1 / np.sqrt(2 ** n))
    psiN = adiabatic(psi0, T, M)
    prob = np.real(np.conj(psiN) * psiN)

    result = np.zeros(n + 1)
    normalise_array = np.zeros(n + 1)

    for i, probability in enumerate(prob):
        binary_i = bin(i)
        i_ones = [ones for ones in binary_i[2:] if ones == '1']
        num_ones = len(i_ones)
        result[num_ones] += probability
        if normalise:
            normalise_array[num_ones] += 1

    if normalise:
        result = result / normalise_array

    return result


def optimal_gamma(n):
    N = 2 ** n
    gam = 0
    for r in range(1, n+1):
        gam += comb(n, r) * (1/r)
    gam = (1/2) * (1/N) * gam
    return gam


def driver_hamiltonian(n, gamma):
    A = hypercube(n)
    return gamma * (A - n * np.eye(2 ** n))


def problem_hamiltonian(n):
    marked_state = np.zeros((2 ** n, 2 ** n))
    marked_state[0, 0] = -1
    return marked_state


def evolution_operator(H_driver, H_problem, T, M):
    H = H_driver
    U = expm((-1j*T/M) * H)
    for i in range(1, M):
        t = i * (T/M)
        H = (1 - t/T)*H_driver + (t/T)*H_problem
        U = np.matmul(U, expm((-1.0j*T/M) * H))
    print("found U")
    return U


def hamiltonian(t, T):
    return (1 - t/T)*H_driver + (t/T)*H_problem


def adiabatic(psi0, T, M):
    psiN = psi0
    for i in range(1, M+1):
        t = i * (T / M)
        H = hamiltonian(t, T)
        U = expm(-1j * (T / M) * H)
        psiN = np.dot(U, psiN)
    return psiN


if __name__ == '__main__':
    time_start = time.time()
    M = 100     # number of slices

    t_finish = 100
    n = 3  # number of dimensions of hypercube
    gamma = optimal_gamma(n)    # hopping rate
    print("gamma:", gamma)

    H_driver = driver_hamiltonian(n, gamma)
    H_problem = problem_hamiltonian(n)

    probs = quantum_walk(n, t_finish, M)

    time_end = time.time()
    print("runtime:", time_end - time_start)

    plt.figure()
    plt.plot(np.arange(n+1), probs)
    plt.xlabel("Hamming distance, d")
    plt.xticks(range(n+1))
    plt.xlim(0, n)
    plt.ylabel("Normalised probability of states, P(d)")
    plt.show()

