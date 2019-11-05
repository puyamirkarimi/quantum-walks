import numpy as np
from scipy.sparse.linalg import expm_multiply


def evolve_continuous(H, psi0, timesteps):
    psiT = expm_multiply(-1j * timesteps * H, psi0)
    prob = np.real(np.conj(psiT) * psiT)
    return prob


def hamming_probabilities(prob, N, normalise=False):
    result = np.zeros(N + 1)
    normalise_array = np.zeros(N + 1)
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


def continuous_walk_hypercube(n, time_limit, gamma, normalise=False):
    N = 2**n  # number of positions
    A = hypercube(n)
    marked_state = np.zeros((N, N))
    marked_state[0, 0] = 1
    H = gamma * (A - n * np.eye(N)) - marked_state  # not sure if marked_node should also be multiplied by gamma

    psi0 = np.ones(N) / np.sqrt(N)
    output = np.zeros((time_limit + 1, n + 1))
    for timesteps in range(0, time_limit + 1):
        probabilities = evolve_continuous(H, psi0, timesteps)
        output[timesteps] = hamming_probabilities(probabilities, n, normalise=normalise)
        print(sum(output[timesteps]))
    return output


def many_continuous_walks


def hypercube(n_dim):
    n = n_dim - 1
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    A = sigma_i(sigma_x, 0, n)

    for i in range(1, n_dim):
        A += sigma_i(sigma_x, i, n)
    return -1*A


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
