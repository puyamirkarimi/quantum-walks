import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1]


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances_original/" + instance_name + ".m2s")
    return out.astype(int)


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


def energy_spread(H):
    energies = np.diagonal(H)
    return np.max(energies)-np.min(energies)


if __name__ == "__main__":
    instance_names, instance_n_bits = get_instances()

    num_samples = 1000              # max 10000

    energy_spread_array = np.zeros(num_samples)

    n = 10
    n_shifted = n - 5  # n_shifted runs from 0 to 15 instead of 5 to 20

    for loop, i in enumerate(range(n_shifted * 10000, n_shifted * 10000 + num_samples)):  # 10000 instances per value of n
        instance_name = instance_names[i]
        formula = get_2sat_formula(instance_name)
        H = hamiltonian_2sat(n, formula)
        energy_spread_array[loop] = energy_spread(H)

        if loop % 10 == 0:
            print("loop:", loop)

    with open("energy_spread_n_"+str(n)+".txt", "ab") as f:         # saves runtimes using time.time()
        f.write(b"\n")
        np.savetxt(f, energy_spread_array)
