import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy import linalg
import math

def hypercube(ndim, diagonal=False):
    """Recursively construct the edge-connectivity of a hypercube

    Parameters
    ----------
    ndim : int
        Dimension of the hypercube
    diagonal : bool
        Value of the diagonal
        If True, vertices are considered connected to themselves

    Returns
    -------
    ndarray, [2**ndim, 2**ndim], bool
        connectivity pattern of the hypercube
    """
    if ndim == 0:
        return np.array([[diagonal]])
    else:
        D = hypercube(ndim-1, diagonal)
        I = np.eye(len(D), dtype=D.dtype)
        return np.block([
            [D, I],
            [I, D],
        ])

print hypercube(2)

'''
N = 60         # number of random steps
timesteps = 40
P = 2*N+1       # number of positions
gamma = 0.5     # hopping rate
positions = np.arange(-N, N+1)

A = np.zeros((P, P))
j = 0
for i in range(P):
    A[j, i-1] = 1
    if i==P-1:
        A[j, 0] = 1
    else:
        A[j, i+1] = 1
    j += 1

print A

U = linalg.expm(-(1j)*gamma*A)          # walk operator

posn0 = np.zeros(P)
posn0[N] = 1                                              # array indexing starts from 0, so index N is the central posn
#psi0 = np.kron(posn0,(coin0+coin1*1j)/np.sqrt(2.))        # initial state
psi0 = posn0

psiN = np.linalg.matrix_power(U, timesteps).dot(psi0)

prob = np.real(np.conj(psiN) * psiN)
print prob

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(positions, prob)
# plt.xticks(range(-N, N+1, int(0.2*N)))
ax.set_xlabel("Position, x")
ax.set_ylabel("Probability, P(x)")
plt.show()
'''
