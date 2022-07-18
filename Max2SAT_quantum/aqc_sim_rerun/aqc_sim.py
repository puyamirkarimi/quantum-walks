# %%t
# imports and function definitions

import numpy as np
from scipy.integrate import complex_ode
from scipy import sparse


def sigma_i_sparse(sigma, i, n):
    """Returns the Pauli matrix 'sigma' acting on the i-th qubit as a sparse
    matrix. Note that the i-th qubit is the i-th qubit from the right, whereas
    sometimes sigma_i is defined on the i-th qubit from the left.

    Args:
        sigma: single-qubit Pauli matrix (i.e. X, Y, Z)
        i: qubit to act on
        n: total number of qubits

    Raises:
        Exception: when i > n - 1

    Returns:
        The Pauli matrix acting on the i-th qubit
    """
    if i > n - 1:
        raise Exception('Qubit index too high for given n in sigma_i')
    if i < n - 1:
        out = sparse.eye(2**(n-1-i), format='csc', dtype=int)
        out = sparse.kron(out, sigma)
        out = sparse.kron(out, sparse.eye(2**i, format='csc', dtype=int))
    else:
        # when i = n - 1
        out = sparse.kron(sigma, sparse.eye(2**(n-1), format='csc', dtype=int))
    return out


def hypercube_sparse(n, X):
    """Returns the hypercube (transverse field) Hamiltonian A
    as a sparse matrix.
    A = - \sum_{i=0}^{n-1} sigma_i^x
    """
    A = sigma_i_sparse(X, 0, n)

    for i in range(1, n):
        A += sigma_i_sparse(X, i, n)
    
    return A


def driver_hamiltonian_transverse_field(n, X):
    """Returns the transverse field driver Hamiltonian.
    """
    A = -1 * hypercube_sparse(n, X)
    return A


def hamiltonian(t, T, H_driver, H_problem):
    """Returns the total Hamiltonian."""
    return (1 - t/T)*H_driver + (t/T)*H_problem


def schrodinger(t, psi, T, H_driver, H_problem):
    """Returns the dH/dt according to the Schrodinger equation."""
    return -1j * hamiltonian(t, T, H_driver, H_problem).dot(psi)


def aqc_success_prob(n, T, H_driver, H_problem, integrator_steps=10000000, psi0=None):
    """Simulates AQC and returns the final success probability.

    Args:
        n: number of qubits
        T: total time of sweep
        H_driver: driver Hamiltonian
        H_problem: problem Hamiltonian
        integrator_steps (optional): max number of integrator timesteps
            (defaults to 10,000,000)
        psi0 (optional): initial state

    Returns:
        float: success probability
        bool: whether the integrator was successful
    """
    N = 2**n
    if psi0 is None:
        psi0 = np.ones(N) * (1 / np.sqrt(N))

    schro = lambda t, y: schrodinger(t, y, T, H_driver, H_problem)
    r = complex_ode(schro)
    r.set_integrator("dop853", nsteps=integrator_steps)
    r.set_initial_value(psi0, 0)
    # r.set_f_params(T, H_driver, H_problem)
    psiN = r.integrate(T)

    # ground_state = np.zeros(N)
    # ground_state[0] = 1
    # success_prob = np.abs(np.conjugate(ground_state).dot(psiN)) ** 2

    success_prob = np.real(np.conj(psiN[0]) * psiN[0])
        
    return success_prob, r.successful()


def aqc_find_T(n, Hw, Hp, psi_i, T_start=10.0, target_prob=0.99, \
tolfrac=0.01, Tmin=0.1, Tmax=10000.0, verbose=False):
    old_T = 0.0
    T = T_start
    p, success = aqc_success_prob(n, T, Hw, Hp)
    old_p = -float('inf')
    if verbose: print(f'Ran with T={T}, found p={p}')
    if p > target_prob:
        while p > target_prob:
            old_T = T
            T = T*0.5
            if T < Tmin or T > Tmax:
                return None, False
            old_p = p
            p, s = aqc_success_prob(n, T, Hw, Hp)
            if success: success = s
            if verbose: print(f'Ran with T={T}, found p={p}')
        Tlower = T
        Tupper = old_T
        plower = p
        pupper = old_p
        T = Tupper
    else:
        while p < target_prob:
            old_T = T
            T = T*2.0
            if T < Tmin or T > Tmax:
                return None, False
            old_p = p
            p, s = aqc_success_prob(n, T, Hw, Hp)
            if success: success = s
            if verbose: print(f'Ran with T={T}, found p={p}')
        Tlower = old_T
        Tupper = T
        plower = old_p
        pupper = p
    if verbose: print(f'Found range Tlower={Tlower}, Tupper={Tupper}')
    while Tupper - Tlower > 0.5*(Tupper + Tlower)*tolfrac:
        T = 0.5*(Tupper + Tlower)
        if T < Tmin or T > Tmax:
            return None, False
        p, s = aqc_success_prob(n, T, Hw, Hp)
        if success: success = s
        if verbose: print(f'Ran with T={T}, found p={p}')
        if p < target_prob:
            plower, pupper = p, pupper
            Tlower, Tupper = T, Tupper
        elif p >= target_prob:
            plower, pupper = plower, p
            Tlower, Tupper = Tlower, T
    if verbose: print('Done')
    return T, success


def get_2sat_formula(instance_name):
    out = np.loadtxt("./../../../instances_original/" + instance_name + ".m2s")
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('./../m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


def hamiltonian_2sat_sparse(n, formula, sigma_z):
    N = 2 ** n
    out = sparse.csc_matrix((N, N))
    sigma_identity = sparse.eye(N, format='csc')
    for clause in formula:
        v_1 = clause[1]
        v_2 = clause[3]
        sign_1 = -1 * clause[0]                 # -1 because signs should be opposite in Hamiltonian
        sign_2 = -1 * clause[2]
        '''below we use .multiply(), which is elementwise multiplication, rather than .dot(), which is matrix
        multiplication, becasue the matrices in the problem Hamiltonian are diagonal so the end result is the same for
        both types of multiplication, even though .dot() is technically the correct type.'''
        out += (1/4) * (sign_1*sign_2*sigma_i_sparse(sigma_z, v_1, n).multiply(sigma_i_sparse(sigma_z, v_2, n))
                        + sign_1*sigma_i_sparse(sigma_z, v_1, n) + sign_2*sigma_i_sparse(sigma_z, v_2, n) + sigma_identity)
    return out

# %%
# run

def run(instance_num):
    X_dense = np.array([[0, 1],
                        [1, 0]])
    Z_dense = np.array([[1, 0],
                        [0, -1]])
    X_sparse = sparse.csc_matrix(X_dense)
    Z_sparse = sparse.csc_matrix(Z_dense)

    instance_names, instance_n_bits = get_instances()

    instance_name = instance_names[instance_num]
    sat_formula = get_2sat_formula(instance_name)
    n = instance_n_bits[instance_num]
    print("n:", n)

    N = 2 ** n
    H_driver = driver_hamiltonian_transverse_field(n, X_sparse)
    H_problem = hamiltonian_2sat_sparse(n, sat_formula, Z_sparse)

    psi0 = np.ones(N) * (1 / np.sqrt(N))

    out = aqc_find_T(n, H_driver, H_problem, psi0, T_start=10.0, target_prob=0.99, tolfrac=0.01, Tmin=0.1, Tmax=10000.0, verbose=True)
    return instance_num, instance_name, out[0], out[1]


instance_num, instance_name, t_99, success = run(0)

with open("test.csv", "w") as output:
    output.write(str(instance_num)+','+str(instance_name)+','+str(t_99)+','+str(success))
