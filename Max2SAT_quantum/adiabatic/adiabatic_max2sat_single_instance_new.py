import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import eigsh
from scipy.special import comb
import time


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../../instances_original/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('../m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


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


def adiabatic(n, T, M, H_driver, H_problem, normalise=True):
    N = 2**n
    psiN = np.ones(N) * (1 / np.sqrt(N))
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
        # U = expm(-1j * (T / M) * H)
        # psiN = np.dot(U, psiN)
        A = -1j * (T / M) * H
        psiN = expm_multiply(A, psiN)
        prob_ground_H[i] = np.abs(np.dot(np.conjugate(first_eigv(H)), psiN))**2
        prob_ground_H_driv[i] = np.abs(np.dot(np.conjugate(ground_state_driv), psiN))**2
        prob_ground_H_prob[i] = np.abs(np.dot(np.conjugate(ground_state_prob), psiN)) ** 2

    return prob_ground_H, prob_ground_H_driv, prob_ground_H_prob


def hamiltonian(t, T, H_driver, H_problem):
    print("t/T:", t/T)
    return (1 - t/T)*H_driver + (t/T)*H_problem


def first_eigv(A):
    return eigsh(A, k=1, which='SM')[1][:, 0]


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
    time_start = time.time()
    M = 500     # number of slices
    t_finish = 500
    timestep = t_finish / M

    instance_names, instance_n_bits = get_instances()

    n = 10
    gamma = heuristic_gamma(n)    # hopping rate

    instance_name = instance_names[50005]
    sat_formula = get_2sat_formula(instance_name)

    H_driver = sparse.csc_matrix(driver_hamiltonian(n, gamma))
    H_problem = sparse.csc_matrix(hamiltonian_2sat(n, sat_formula))

    prob_H_total, prob_H_driv, prob_H_prob = adiabatic(n, t_finish, M, H_driver, H_problem)

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

