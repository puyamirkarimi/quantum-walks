import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import comb
from scipy.integrate import complex_ode
from pathlib import Path


def get_2sat_formula(path_to_instance):
    out = np.loadtxt(path_to_instance)
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('../m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


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


def schrodinger(t, psi, T, H_driver, H_problem):
    return -1j * ((1 - t/T)*H_driver + (t/T)*H_problem).dot(psi)


def adiabatic(n, T, H_driver, H_problem, ground_state_prob, normalise=True, sprs=True, n_steps=4000):
    N = 2**n
    psi0 = np.ones(N) * (1 / np.sqrt(N))
    newschro = lambda t, y: schrodinger(t, y, T, H_driver, H_problem)
    r = complex_ode(newschro)
    r.set_integrator("dop853", nsteps=n_steps)
    r.set_initial_value(psi0, 0)
    # r.set_f_params(T, H_driver, H_problem)
    psiN = r.integrate(T)
    return np.abs(np.conjugate(ground_state_prob).dot(psiN)) ** 2, r.successful()


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
    if n == 12:
        out = 0.6255458333333334
    if n == 13:
        out = 0.6292038461538462
    if n == 14:
        out = 0.6291071428571428
    if n == 15:
        out = 0.6301333333333333
    if n == 16:
        out = 0.633021875
    if n == 17:
        out = 0.6346411764705883
    if n == 18:
        out = 0.6349472222222222
    if n == 19:
        out = 0.6361394736842105
    if n == 20:
        out = 0.63659
    return out


def run(instance_name, instances_folder, n, sparse_matrix=True, max_T=8192, n_steps=4000):
    instances_path = Path(instances_folder)
    instance_name += ".m2s"
    instance_path = instances_path / instance_name
    max_T = 8192

    sprs = sparse_matrix

    gamma = heuristic_gamma(n)  # hopping rate

    if sprs:
        H_driver = sparse.csc_matrix(driver_hamiltonian(n, gamma))
    else:
        H_driver = driver_hamiltonian(n, gamma)

    abandon = False
    success_prob = 0
    t_finish_old = 1
    t_finish = 1
    T = 0
    sat_formula = get_2sat_formula(instance_path)
    if sprs:
        H_problem = sparse.csc_matrix(hamiltonian_2sat(n, sat_formula))
    else:
        H_problem = hamiltonian_2sat(n, sat_formula)
    ground_state_prob = np.zeros(2**n)
    ground_state_prob[0] = 1
    successful_integration = False
    success = False

    while success_prob < 0.99 and not abandon:
        t_finish_old = t_finish
        t_finish *= 2
        if t_finish > max_T:
            abandon = True
            T = -1
            break
        success_prob, successful_integration = adiabatic(n, t_finish, H_driver, H_problem, ground_state_prob, sprs=sprs, n_steps=n_steps)
        success = successful_integration

    if not abandon:
        while t_finish - t_finish_old > 1:
            t_mid = int((t_finish + t_finish_old) / 2)
            success_prob, successful_integration = adiabatic(n, t_mid, H_driver, H_problem, ground_state_prob, sprs=sprs)
            if success_prob < 0.99:
                t_finish_old = t_mid
            else:
                t_finish = t_mid
                success = successful_integration
        T = t_finish

    return T, success


if __name__ == '__main__':
    instance_names, instance_n_bits = get_instances()
    print(run(instance_names[0], "../../../instances_original/", 5, sparse_matrix=False))

