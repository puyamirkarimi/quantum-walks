import numpy as np
import time

class FakeFunc:
    def __init__(self, func):
        self.func = func
        self.count = {}

    def __call__(self, *args, **kwargs):
        n = len(args[0])
        self.count[n] = self.count.get(n, 0) + 1
        self.count['total'] = self.count.get('total', 0) + 1
        return self.func(*args, **kwargs)

    def reset(self):
        self.count = {}


def callcount(func):
    newfunc = FakeFunc(func)
    return newfunc

@callcount
def cost(s, C, p):
    sT = np.transpose(s)
    #transpose for a column vector rather than for a row vector
    val = np.dot(s, np.transpose(np.dot(C,s))) + np.dot(p, sT)
    val = float(val)
    return val

@callcount
def branch(S):
    S_plus = S.copy()
    S_plus = np.append(S_plus, [1])
    S_minus = S.copy()
    S_minus = np.append(S_minus, [-1])
    branch = [S_plus, S_minus]
    return branch

@callcount
def bound(S_j, p, C, f_nm_list):

    m = len(S_j) #slicing the input arrays to the appropriate dimension
    n = p.shape[0]
    C_m = C[:m,:m]
    p_m = p[:m]

    A = cost(S_j, C_m, p_m)

    B = 0
    for i in range(m, n):
        B -= np.abs(np.dot(S_j, C[:m,i]))

    D = f_nm_list[(n-m)-1]

    return A + B + D

def f_min2(C, p, pres=False):
    n = p.shape[0]
    if n == 1 and pres:
        return [-np.abs(p[0])], [-int(np.sign(p[0]))]
    if n == 1 and not pres:
        return -np.abs(p[0]), -int(np.sign(p[0]))
    best = None
    best_energy = np.inf
    z = np.empty(n)

    best_energies, bests = f_min2(C[1:,1:], p[1:], pres=True)

    L = branch([])

    while len(L) > 0:
        S = L.pop()
        #for full assignments, compute cost and compare to the best current values, replacing as necessary
        if len(S) == n:
            costVal = cost(S, C, p)
            if costVal <= best_energy:
                best_energy = costVal
                best = S

        #for partial assignments compute and compare the assignment bound, storing the assignment and branching as necessary
        if len(S) < n:
            bound_val = bound(S, p, C, best_energies)
            if bound_val <= best_energy:
                plus, minus = branch(S)
                L.append(plus)
                L.append(minus)

    if pres:
        return best_energies + [best_energy], bests + [best]
    else:
        return best_energy, best

def isingify_m2s(nqubits, prob):
    J = np.zeros((nqubits, nqubits))
    h = np.zeros(nqubits)
    ic = 0.0

    for i, clause in enumerate(prob):
        s0, v0, s1, v1 = tuple([int(x) for x in clause])

        ic += 0.25
        J[v0, v1] += 0.25*s0*s1
        h[v0] -= 0.25*s0
        h[v1] -= 0.25*s1

    return J, h, ic

def solve_m2s(nqubits, prob):
    J, h, ic = isingify_m2s(nqubits, prob)
    C, p = J, h
    bound.reset()
    cost.reset()
    branch.reset()
    stime = time.time()
    stime_p = time.process_time()
    res=f_min2(C,p)
    etime_p = time.process_time()
    etime = time.time()
    duration = etime-stime
    duration_p = etime_p - stime_p
    nrg = res[0]
    sol = [int(x) for x in list(((-1*res[1])+1)/2)]
    sol_i = [str(x) for x in sol]
    sol_i.reverse()
    sol_i = ''.join(sol_i)
    sol_i = int(sol_i,2)

    newrow = {'nqubits':nqubits,'sol_index':sol_i, 'sol':sol,
    'unsatisfied':int(nrg+ic), 'duration':duration,
    'duration_processtime':duration_p}

    for j in (cost.count.keys()):
        if j == 'total':
            continue
        newrow['cost_calls_{}'.format(j)] = cost.count[j]
    for j in (bound.count.keys()):
        if j == 'total':
            continue
        newrow['bound_calls_{}'.format(j)] = bound.count[j]
    for j in (branch.count.keys()):
        if j == 'total':
            continue
        newrow['branch_calls_{}'.format(j)] = branch.count[j]
    newrow['cost_calls'] = cost.count['total']
    newrow['bound_calls'] = bound.count['total']
    newrow['branch_calls'] = branch.count['total']
    newrow['calls'] = cost.count['total']+bound.count['total']+\
    branch.count['total']

    bound.reset()
    cost.reset()
    branch.reset()

    return newrow


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


def prob_array(filename):
    return np.loadtxt('./../../instances_original/'+filename+".m2s")


if __name__ == '__main__':
    instance_names, instance_n_bits = get_instances()

    runtimes_time = np.zeros(10000)
    runtimes_processtime = np.zeros(10000)
    counts = np.zeros(10000)

    n = 10
    n_shifted = n - 5  # n_shifted runs from 0 to 15 instead of 5 to 20

    for loop, i in enumerate(range(n_shifted * 10000, (n_shifted + 1) * 10000)):  # 10000 instances per value of n
        instance_name = instance_names[i]
        solution = solve_m2s(n, prob_array(instance_names[i]))
        runtimes_time[loop] = solution['duration']
        runtimes_processtime[loop] = solution['duration_processtime']
        counts[loop] = solution['calls']
        if loop % 100 == 0:
            print("loop:", loop)

    # for loop, i in enumerate(range(n_shifted * 10000, (n_shifted + 1) * 10000)):  # 10000 instances per value of n
    #     instance_name = instance_names[i]
    #     solution = solve_m2s(n, prob_array(instance_names[i]))
    #     if 1 in solution['sol']:
    #         print("error")

    with open("adam_runtimes_time_"+str(n)+".txt", "ab") as f:         # saves time
        f.write(b"\n")
        np.savetxt(f, runtimes_time)

    with open("adam_runtimes_processtime_"+str(n)+".txt", "ab") as f:         # saves processtime
        f.write(b"\n")
        np.savetxt(f, runtimes_processtime)

    with open("adam_counts_"+str(n)+".txt", "ab") as f:         # saves counts
        f.write(b"\n")
        np.savetxt(f, counts)

