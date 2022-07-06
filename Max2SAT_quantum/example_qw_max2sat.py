# %%
# imports and functions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import complex_ode
from scipy.sparse import csc_matrix
from scipy import sparse
import time
from pathlib import Path
import pickle as pkl

blue = '#0072B2'
orange = '#EF6900'
green = '#009E73'

def get_2sat_formula(instance_name):
    out = np.loadtxt(Path("./../../instances_original/" + instance_name + ".m2s"))
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt(Path('./../instance_gen/m2s_nqubits.csv'), delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


def get_energy_spreads():
    """returns array of instance energy spreads"""
    data = np.genfromtxt(Path('./../instance_gen/instances_energy.csv'), delimiter=',', skip_header=1, dtype=str)
    return data[:, 1].astype(float)


def first_eigvec(A):
    """returns ground state of matrix A"""
    return np.linalg.eigh(A)[1][:, 0]


def hypercube(n_dim):
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    sigma_x = sparse.csc_matrix(sigma_x, dtype=int)
    A = sigma_i(sigma_x, 0, n_dim)

    for i in range(1, n_dim):
        A += sigma_i(sigma_x, i, n_dim)
    return -1 * A


def sigma_i(sigma, i, n_dim):
    I = sparse.eye(2, format='csc', dtype=int)
    n = n_dim -1            # because of i starting from 0 rather than 1
    if i > 0:
        out = sparse.eye(2**i, format='csc', dtype=int)
        out = sparse.kron(out, sigma)
        out = sparse.kron(out, sparse.eye(2**(n-i), format='csc', dtype=int))
    else:
        out = sparse.kron(sigma, sparse.eye(2**n, format='csc', dtype=int))
    return out


# def quantum_walk_hypercube(N, H, psi0, start_time, stop_time, num_steps):
#     psiN_list = expm_multiply(-(1j) * H, psi0, start=start_time, stop=stop_time, num=num_steps, endpoint=True)

#     success_probs = np.zeros(num_steps)

#     for i in range(num_steps):
#         success_probs[i] = np.real(np.conj(psiN_list[i][0]) * psiN_list[i][0])

#     return success_probs


def schrodinger(t, psi, H):
    return -1j * H.dot(psi)


def append_sol(t, y, success_probs, times, last_state, n):
    last_state[0] = y
    success_probs.append(np.real(np.conj(y[0]) * y[0]))
    times.append(t)
    if n > 15:
        print(t)


def quantum_walk_hypercube(n, H, psi0, start_time, stop_time, success_probs, times, last_state):
    newschro = lambda t, y: schrodinger(t, y, H)
    newappendsol = lambda t, y: append_sol(t, y, success_probs, times, last_state, n)
    r = complex_ode(newschro)
    r.set_integrator("dop853", nsteps=10000000)
    r.set_solout(newappendsol)
    r.set_initial_value(psi0, start_time)
    r.integrate(stop_time)

    print("number of solutions:", len(times))
    return np.array(times), np.array(success_probs)


def hamiltonian_2sat(n, formula):
    N = 2 ** n
    out = sparse.csc_matrix((N, N), dtype=int)
    sigma_z = np.array([[1, 0],
                        [0, -1]])
    sigma_z = sparse.csc_matrix(sigma_z, dtype=int)
    sigma_identity = sparse.eye(N, format='csc', dtype=int)
    sigma_z_i = []
    for i in range(n):
        sigma_z_i.append(sigma_i(sigma_z, i, n))
    for clause in formula:
        v_1 = clause[1]
        v_2 = clause[3]
        sign_1 = -1 * clause[0]                 # -1 because signs should be opposite in Hamiltonian
        sign_2 = -1 * clause[2]
        out += (1/4) * (sign_1*sign_2*sigma_z_i[v_1].multiply(sigma_z_i[v_2])
                        + sign_1*sigma_z_i[v_1] + sign_2*sigma_z_i[v_2] + sigma_identity)
    return out


def heuristic_gamma(n):
    out = "haven't defined heuristic gamma for given n"
    # if n == 5:
    #     out = 0.56503
    # if n == 6:
    #     out = 0.587375
    # if n == 7:
    #     out = 0.5984357142857143
    # if n == 8:
    #     out = 0.60751875
    # if n == 9:
    #     out = 0.6139833333333333
    # if n == 10:
    #     out = 0.619345
    # if n == 11:
    #     out = 0.6220136363636364
    
    energy_spreads = get_energy_spreads()
    energy_spreads_n = energy_spreads[10000*(n-5):10000*(n-4)]
    out = (1/(2*n))*np.mean(energy_spreads_n)
    print("heuristic gamma: ", out)
    return out


# %%
# simulations

instance_names, instance_n_bits = get_instances()
# instance_nums_1 = [0, 1]
# instance_nums_2 = [150000, 15001]
# instance_num = 0
instance_num = 150001

instance_name = instance_names[instance_num]
sat_formula = get_2sat_formula(instance_name)
n = instance_n_bits[instance_num]                   # number of variables/qubits
print("n:", n)

time_start = time.time()

N = 2 ** n  # number of positions
print("getting hypercube Hamiltonian")
A = hypercube(n)
print("getting problem Hamiltonian")
H_problem = hamiltonian_2sat(n, sat_formula)
print("Getting heuristic hopping rate")
gamma = heuristic_gamma(n)
print("getting total Hamiltonian")
H = gamma * (A - n * sparse.eye(N, format='csc')) + H_problem

psi0 = np.ones(N) * (1 / np.sqrt(N))

start = 0
stop = 100
print("Starting sim")
times_list = list()
success_probs_list = list()
last_state = [0]

times, success_probs = quantum_walk_hypercube(n, H, psi0, start, stop, success_probs_list, times_list, last_state)

time_end = time.time()

print("runtime:", time_end - time_start)

# %%
# pickle the data

# print(times)
# print("--------------")
# print(times_list)
# print("--------------")
# print(success_probs)
# print("--------------")
# print(success_probs_list)
# print("--------------")
# print(last_state[0])

# data = [times, success_probs]
# with open(f'examples_qw_max2sat_data_{instance_num}.pkl', 'wb') as f:
#     pkl.dump(data, f)

    
# %%
# unpickle the data and plot

times_data = list()
success_probs_data = list()

plt.rc('text', usetex=True)

fig, axs = plt.subplots(1, 2, figsize=(6, 2.7))
axs[0].tick_params(direction='in', which='both', labelsize=13)
axs[1].tick_params(direction='in', which='both', labelsize=13)
axs[0].set_ylabel("$P(t_f)$", fontsize=15)
axs[0].set_yticks(np.arange(0.2, 0.8, 0.2))
axs[1].set_yticks(np.arange(0.005, 0.02, 0.005))

timesteps = 50
instance_names, instance_n_bits = get_instances()
instance_nums_1 = [0, 1]
instance_nums_2 = [150000, 150001]

for i, instance_num in enumerate(instance_nums_1):

    with open(f'examples_qw_max2sat_data_{instance_num}.pkl', 'rb') as f:
        (times, success_probs) = pkl.load(f)
    
    start = 0
    stop = 100

    if i == 0:
        axs[0].plot(times, success_probs, color=blue)
    else:
        axs[0].plot(times, success_probs, color=orange, linestyle="--")

    axs[0].set_xlim(start, stop)
    # axs[0].set_xticks(range(0, timesteps + 1, 10))
    axs[0].set_ylim(0, 0.62)
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[0].set_xlabel("$t_f$", fontsize=15)

# ylims = axs[0].get_ylim()

for i, instance_num in enumerate(instance_nums_2):

    with open(f'examples_qw_max2sat_data_{instance_num}.pkl', 'rb') as f:
        (times, success_probs) = pkl.load(f)
    
    start = 0
    stop = 100

    if i == 0:
        axs[1].plot(times, success_probs, color=blue)
    else:
        axs[1].plot(times, success_probs, color=orange, linestyle="--")

    axs[1].set_xlim(start, stop)
    # axs[1].set_xticks(range(0, timesteps + 1, 10))
    axs[1].set_ylim(0, 0.0165)
    #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    axs[1].set_xlabel("$t_f$", fontsize=15)

fig.tight_layout()
# plt.savefig('inst_success_probs_windows.pdf', dpi=200)
plt.show()


# %%
# USING expm_multiply:
###########

# if __name__ == '__main__':
#     sols = list()
#     times = list()

#     plt.rc('text', usetex=True)
#     plt.rc('font', size=16)
#     plt.rcParams["figure.figsize"] = (9.6, 4.8)

#     fig, axs = plt.subplots(1, 2)
#     axs[0].tick_params(direction='in', which='both')
#     axs[1].tick_params(direction='in', which='both', labelleft=False)
#     axs[0].set_ylabel("$P(t_f)$")
#     axs[0].set_yticks(np.arange(0.1, 0.7, 0.1))

#     timesteps = 50
#     instance_names, instance_n_bits = get_instances()
#     instance_nums_1 = [0, 1]
#     # instance_nums_2 = [150000, 15001]
#     instance_nums_2 = [3, 4]

#     for i, instance_num in enumerate(instance_nums_1):
#         # instance_name = instance_names[32070]
#         instance_name = instance_names[instance_num]
#         sat_formula = get_2sat_formula(instance_name)
#         n = instance_n_bits[instance_num]                   # number of variables/qubits
#         print("n:", n)

#         time_start = time.time()

#         N = 2 ** n  # number of positions
#         print("getting hypercube Hamiltonian")
#         A = hypercube(n)
#         print("getting problem Hamiltonian")
#         H_problem = hamiltonian_2sat(n, sat_formula)
#         print("Getting heuristic hopping rate")
#         gamma = heuristic_gamma(n)
#         print("getting total Hamiltonian")
#         H = gamma * (A - n * sparse.eye(N, format='csc')) + H_problem

#         psi0 = np.ones(N) * (1 / np.sqrt(N))
        
#         start = 0
#         stop = 100
#         num_steps= 101
#         print("Starting sim")
#         success_probs = quantum_walk_hypercube(N, H, psi0, start, stop, num_steps)

#         time_end = time.time()

#         print("runtime:", time_end - time_start)

#         axs[0].plot(np.linspace(start, stop, num_steps, endpoint=True), success_probs)
#         axs[0].set_xlim(start, stop)
#         # axs[0].set_xticks(range(0, timesteps + 1, 10))
#         # axs[0].set_ylim(0, 0.6)
#         #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#         axs[0].set_xlabel("$t_f$")
    
#     ylims = axs[0].get_ylim()

#     for i, instance_num in enumerate(instance_nums_2):
#         # instance_name = instance_names[32070]
#         instance_name = instance_names[instance_num]
#         sat_formula = get_2sat_formula(instance_name)
#         n = instance_n_bits[instance_num]                   # number of variables/qubits
#         print("n:", n)

#         time_start = time.time()

#         N = 2 ** n  # number of positions
#         print("getting hypercube Hamiltonian")
#         A = hypercube(n)
#         print("getting problem Hamiltonian")
#         H_problem = hamiltonian_2sat(n, sat_formula)
#         print("Getting heuristic hopping rate")
#         gamma = heuristic_gamma(n)
#         print("getting total Hamiltonian")
#         H = gamma * (A - n * sparse.eye(N, format='csc')) + H_problem

#         psi0 = np.ones(N) * (1 / np.sqrt(N))
        
#         start = 0
#         stop = 100
#         num_steps= 101
#         print("Starting sim")
#         success_probs = quantum_walk_hypercube(N, H, psi0, start, stop, num_steps)

#         time_end = time.time()

#         print("runtime:", time_end - time_start)

#         axs[1].plot(np.linspace(start, stop, num_steps, endpoint=True), success_probs)
#         axs[1].set_xlim(start, stop)
#         # axs[1].set_xticks(range(0, timesteps + 1, 10))
#         axs[1].set_ylim(ylims[0], ylims[1])
#         #plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
#         axs[1].set_xlabel("$t_f$")

#     # plt.savefig('inst_success_probs_n_5_10.png', dpi=200)
#     plt.show()
