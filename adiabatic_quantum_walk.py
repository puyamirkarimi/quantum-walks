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


def quantum_walk(n, T, M, H_driver, H_problem, normalise=True):
    N = 2**n
    psi0 = np.ones(N) * (1 / np.sqrt(N))
    psiN = psi0
    H = H_driver

    prob_ground_H = np.zeros(M+1)
    prob_ground_H[0] = np.abs(np.dot(first_eigv(H), psiN))**2

    prob_ground_H_driv = np.zeros(M + 1)
    ground_state_driv = first_eigv(H_driver)
    prob_ground_H_driv[0] = np.abs(np.dot(np.conjugate(ground_state_driv), psiN))**2

    prob_ground_H_prob = np.zeros(M + 1)
    ground_state_prob = first_eigv(H_problem)
    prob_ground_H_prob[0] = np.abs(np.dot(np.conjugate(ground_state_prob), psiN)) ** 2

    for i in range(1, M + 1):
        t = i * (T / M)
        H = hamiltonian(t, T, H_driver, H_problem)
        U = expm(-1j * (T / M) * H)
        psiN = np.dot(U, psiN)
        prob_ground_H[i] = np.abs(np.dot(np.conjugate(first_eigv(H)), psiN))**2
        prob_ground_H_driv[i] = np.abs(np.dot(np.conjugate(ground_state_driv), psiN))**2
        prob_ground_H_prob[i] = np.abs(np.dot(np.conjugate(ground_state_prob), psiN)) ** 2

    return prob_ground_H, prob_ground_H_driv, prob_ground_H_prob


def hamiltonian(t, T, H_driver, H_problem):
    print("t/T:", t/T)
    return (1 - t/T)*H_driver + (t/T)*H_problem


def first_eigv(A):
    return np.linalg.eigh(A)[1][:, 0]


def optimal_gamma(n):
    N = 2 ** n
    gam = 0
    for r in range(1, n+1):
        gam += comb(n, r) * (1/r)
    gam = (1/2) * (1/N) * gam
    return gam


def driver_hamiltonian(n, gamma):
    A = hypercube(n)
    return (A + n * np.eye(2 ** n))/2      # plus or minus??? keep the half?


def problem_hamiltonian(n):
    marked_state = np.eye(2 ** n)
    marked_state[0, 0] = 0
    return marked_state


if __name__ == '__main__':
    time_start = time.time()
    M = 300     # number of slices
    t_finish = 300
    timestep = t_finish / M

    n = 4  # number of dimensions of hypercube
    gamma = optimal_gamma(n)    # hopping rate
    print("gamma:", gamma)

    H_driver = driver_hamiltonian(n, gamma)
    H_problem = problem_hamiltonian(n)

    prob_H_total, prob_H_driv, prob_H_prob = quantum_walk(n, t_finish, M, H_driver, H_problem)

    time_end = time.time()
    print("runtime:", time_end - time_start)

    plt.figure()
    plt.plot(np.arange(0, t_finish + timestep, timestep), prob_H_total, label='$H(t)$')
    plt.plot(np.arange(0, t_finish + timestep, timestep), prob_H_driv, label='$H_{driver}$')
    plt.plot(np.arange(0, t_finish + timestep, timestep), prob_H_prob, label='$H_{problem}$')
    plt.xlabel("Time, t")
    plt.xlim(0, t_finish)
    plt.ylim(0, 1)
    plt.ylabel("Probability of being in ground state of hamiltonian")
    plt.legend()
    plt.show()

