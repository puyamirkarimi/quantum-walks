import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

N = 100     # number of random steps
P = 2*N+1   # number of positions
#positions = np.arange(P)
positions = np.arange(-N, N+1)
evens = positions[0::2]

coin0 = np.array([1, 0])     # |0>
coin1 = np.array([0, 1])     # |1>

C00 = np.outer(coin0, coin0)      # |0><0|
C01 = np.outer(coin0, coin1)      # |0><1|
C10 = np.outer(coin1, coin0)      # |1><0|
C11 = np.outer(coin1, coin1)      # |1><1|

C_hat = (C00 + C01 + C10 - C11)/np.sqrt(2.)

ShiftPlus = np.roll(np.eye(P), 1, axis=0)                       # MAKE SURE TO UNDERSTAND THESE
ShiftMinus = np.roll(np.eye(P), -1, axis=0)
S_hat = np.kron(ShiftPlus, C00) + np.kron(ShiftMinus, C11)      # shift operator

U = S_hat.dot(np.kron(np.eye(P), C_hat))                        # walk operator

posn0 = np.zeros(P)
posn0[N] = 1                                              # array indexing starts from 0, so index N is the central posn
#psi0 = np.kron(posn0,(coin0+coin1*1j)/np.sqrt(2.))        # initial state
psi0 = np.kron(posn0, coin0)

psiN = np.linalg.matrix_power(U, N).dot(psi0)

prob = np.empty(P)
for k in range(P):
    posn = np.zeros(P)
    posn[k] = 1
    M_hat_k = np.kron(np.outer(posn,posn), np.eye(2))
    proj = M_hat_k.dot(psiN)
    prob[k] = proj.dot(proj.conjugate()).real

classical_prob = np.zeros(len(evens))
i = 0
for k in evens:
    p = 0.5
    classical_prob[i] = nCr(N,(N+k)/2) * p**((N+k)/2) * (1-p)**((N-k)/2)
    i+=1

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(evens, prob[0::2])
#plt.plot(evens, prob[0::2], 'o')
plt.plot(evens, classical_prob)
plt.xticks(range(-N, N+1, int(0.2*N)))
ax.set_xlabel("Position, x")
ax.set_ylabel("Probability, P(x)")
plt.show()
