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


def adiabatic(n, T, M, H_driver, H_problem, ground_state_prob, normalise=True, sprs=True):
    N = 2**n
    psiN = np.ones(N) * (1 / np.sqrt(N))
    H = H_driver

    for i in range(1, M + 1):
        t = i * (T / M)
        H = hamiltonian(t, T, H_driver, H_problem)
        if sprs:
            A = -1j * (T / M) * H
            psiN = expm_multiply(A, psiN)
        else:
            U = expm(-1j * (T / M) * H)
            psiN = np.dot(U, psiN)

    return np.abs(np.dot(np.conjugate(ground_state_prob), psiN)) ** 2


def hamiltonian(t, T, H_driver, H_problem):
    return (1 - t/T)*H_driver + (t/T)*H_problem


def first_eigv(A, sprs=True):
    if sprs:
        return eigsh(A, k=1, which='SM')[1][:, 0]
    else:
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
    M = 100     # number of slices
    max_T = 512

    instance_names, instance_n_bits = get_instances()

    n = 7
    sprs = True

    gamma = heuristic_gamma(n)    # hopping rate
    n_shifted = n - 5  # n_shifted runs from 0 to 15 instead of 5 to 20

    if sprs:
        H_driver = sparse.csc_matrix(driver_hamiltonian(n, gamma))
    else:
        H_driver = driver_hamiltonian(n, gamma)
    ground_state_calculated = False
    ground_state_prob = None

    times_array = np.zeros(10000)

    for loop, i in enumerate(range(n_shifted * 10000, (n_shifted + 1) * 10000)):  # 10000 instances per value of n
        abandon = False
        success_prob = 0
        t_finish_old = 1
        t_finish = 1
        success_prob_old = 0
        T = 0
        instance_name = instance_names[i]
        sat_formula = get_2sat_formula(instance_name)
        if sprs:
            H_problem = sparse.csc_matrix(hamiltonian_2sat(n, sat_formula))
        else:
            H_problem = hamiltonian_2sat(n, sat_formula)
        if not ground_state_calculated:
            ground_state_prob = first_eigv(H_problem, sprs=sprs)
            ground_state_calculated = True

        while success_prob < 0.99 and not abandon:
            success_prob_old = success_prob
            t_finish_old = t_finish
            t_finish *= 2
            if t_finish > max_T:
                abandon = True
                T = -1
                break
            success_prob = adiabatic(n, t_finish, M, H_driver, H_problem, ground_state_prob, sprs=sprs)

        if not abandon:
            while t_finish - t_finish_old > 1:
                t_mid = int((t_finish + t_finish_old)/2)
                success_prob = adiabatic(n, t_mid, M, H_driver, H_problem, ground_state_prob, sprs=sprs)
                if success_prob < 0.99:
                    t_finish_old = t_mid
                else:
                    t_finish = t_mid
            T = t_finish

        # print(loop, success_prob, T)
        times_array[loop] = T

        if loop % 10 == 0:
            print("Instance:", loop)

    time_end = time.time()
    print("runtime:", time_end - time_start)

    with open("adiabatic_time_n_"+str(n)+".txt", "ab") as f:         # saves runtimes using time.time()
        np.savetxt(f, times_array)


