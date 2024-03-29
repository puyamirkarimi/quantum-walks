'''
Creates a quantum walk visual to be used as a background
in powerpoint presentations
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy import linalg
import math
from matplotlib.ticker import MultipleLocator

N = 100         # number of random steps
timesteps = 40
P = 2*N+1       # number of positions
gamma = 0.5     # hopping rate
positions = np.arange(-N, N+1)

B = np.array([[0,1,0,0,0],
              [1,0,1,0,0],
              [0,1,0,1,0],
              [0,0,1,0,1],
              [0,0,0,1,0]])

A = np.zeros((P, P))
j = 0
for i in range(P):
    A[j, i-1] = 1
    if i==P-1:
        A[j, 0] = 1
    else:
        A[j, i+1] = 1
    j += 1

#U = linalg.expm(-(1j)*gamma*A)
U = linalg.expm(-(1j) * gamma * timesteps * A)
H_classical = linalg.expm(-1 * gamma * timesteps * (2*np.eye(P) - A))
print(H_classical)
prob0_classical = np.zeros(P)
prob0_classical[N] = 1

posn0 = np.zeros(P)
posn0[N] = 1                                              # array indexing starts from 0, so index N is the central posn
#psi0 = np.kron(posn0,(coin0+coin1*1j)/np.sqrt(2.))        # initial state
psi0 = posn0

#psiN = np.linalg.matrix_power(U, timesteps).dot(psi0)
psiN = U.dot(psi0)
probN_classical = H_classical.dot(prob0_classical)

prob = np.real(np.conj(psiN) * psiN)

plt.rc('text', usetex=True)
plt.rc('font', size=14)

fig = plt.figure()
ax = fig.add_subplot(111)

#plt.plot(positions, probN_classical, linestyle="--", color='grey')
plt.plot(positions, prob, color='#68246D')

# plt.scatter(positions, prob, marker=".")
# plt.scatter(positions, probN_classical, marker=".", color='purple')
# plt.xticks(range(-N, N+1, int(0.2*N)))
# ax.set_xlabel("Position, $x$")
# ax.set_ylabel("Probability, $P(x)$")
ax.set_xlim([-70, 70])
ax.set_ylim([-0.02, 0.1])
#ax.tick_params(direction='in', top=True, right=True)

plt.show()
# plt.savefig('visual.png', dpi=400)
