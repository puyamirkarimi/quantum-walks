import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy import linalg
import math


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
        for j_before in range(i-1):
            out = np.kron(out, np.eye(2))
        out = np.kron(out, sigma_x)
    else:
        out = sigma_x
    for j_after in range(n-i):
        out = np.kron(out, np.eye(2))
    return out

def quantum_walk_hypercube(N, timesteps):
    P = 2**N         # number of positions
    gamma = 0.5     # hopping rate

    A = hypercube(N)

    #U = linalg.expm(-(1j)*gamma*A)          # walk operator
    U = linalg.expm(-(1j) * gamma * timesteps * A)  # walk operator

    posn0 = np.zeros(P)
    posn0[0] = 1
    psi0 = posn0

    psiN = U.dot(psi0)

    prob = np.real(np.conj(psiN) * psiN)

    result = np.zeros(N+1)
    for i, probability in enumerate(prob):
        binary_i = bin(i)
        i_ones = [ones for ones in binary_i[2:] if ones=='1']
        num_ones = len(i_ones)
        result[num_ones] += probability
    print(result)
    return result

def run_walks_check_furthest_qubit(N, time_limit):
    output = np.zeros(time_limit+1)
    for timesteps in range(1, time_limit + 1):
        output[timesteps] = quantum_walk_hypercube(N, timesteps)[-1]
    return output

if __name__ == '__main__':
    N = 9           # number of dimensions of hypercube
    timesteps = 20

    data = run_walks_check_furthest_qubit(N, timesteps)
    plt.figure()
    plt.plot(range(timesteps+1), data)
    plt.show()



# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# plt.plot(positions, prob)
# # plt.xticks(range(-N, N+1, int(0.2*N)))
# ax.set_xlabel("Position, x")
# ax.set_ylabel("Probability, P(x)")
# plt.show()

