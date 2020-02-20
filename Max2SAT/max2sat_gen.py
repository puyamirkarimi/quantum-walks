import numpy as np


def sigma_identity(state, site):
    """
    Convenience function. Returns `(1, state)` for int `state`. Can also accept
    numpy arrays and gives numpy arrays back
    """
    return state ** 0, state


def sigma_z(state, site):
    """
    Returns `(sign, state)` where `sign` is 1 or -1 if the `site`th bit of the int `state`
    is 0 or 1 respectively. Can also accept numpy arrays of ints and gives numpy arrays back
    """
    return (-1.0) ** ((state >> site) & 1), state


I, Z = sigma_identity, sigma_z  # convenient aliases


def max2sat_prob(nqubits=None, nclauses=None):
    """
    Generate a max2sat problem on `nqubits` binary variables with `nclauses` clauses
    """
    cs = []  # initialise empty clause list
    while len(cs) < nclauses:  # some clauses will be rejected so use while instead of for

        q1 = np.random.randint(0, nqubits)  # get a random qubit

        q2 = q1
        while q2 == q1:  # embarrasing hack to find a different random qubit
            q2 = np.random.randint(0, nqubits)

        q1, q2 = np.min((q1, q2)), np.max((q1, q2))  # enforce lowest qubit index comes first

        s1 = (2 * np.random.randint(0, 2)) - 1  # 2 random choices of -1 or 1 (i.e clause sign)
        s2 = (2 * np.random.randint(0, 2)) - 1

        c = (s1, q1, s2, q2)  # make the clause in form (sign1, qubit1, sign2, qubit2)
        if c in cs: continue  # if we already have this clase, reject and try again

        cs.append(c)  # if we got here, the clause is new, so keep it

    cs = sorted(cs, key=lambda x: (x[1] * nqubits) + x[3])  # sort the clauses by qubit1 first, then by qubit 2
    return np.array(cs)  # give sorted clauses as an array


def max2sat_values(prob=None, nqubits=None):
    """
    Gets the number of clauses NOT satisfied by each bitstring from `0` to `2**nqubits - 1`
    in the max2sat problem defined by `prob`
    """

    prob = np.array(prob, dtype=int)  # make sure prob is defined by ints just in case
    nclauses = prob.shape[0]  # get number if clauses in the problem

    N = 2 ** nqubits  # how many bitstrings there are

    Hdiag = np.zeros(N, dtype=np.float64)  # make the empty result array:
    #     ultimately, the number of clauses NOT satisfied
    #     by a bitstring will be placed in the array
    #     element whose index has that bitstring as its
    #     binary representation

    ket = np.array(range(N))  # a convenient array containing indices

    for i in range(len(prob)):  # loops through clauses

        ######## fills in the terms from 0.25*(1-Z1)*(1-Z2) for each clause ########
        coeff0, bra0 = I(ket, 0)
        coeff1, bra1 = Z(ket, prob[i, 1])
        coeff2, bra2 = Z(ket, prob[i, 3])
        coeff3, bra3 = Z(bra2, prob[i, 1])
        Hdiag[bra0] += 0.25 * coeff0
        Hdiag[bra1] += 0.25 * (-prob[i, 0] * coeff1)
        Hdiag[bra2] += 0.25 * (-prob[i, 2] * coeff2)
        Hdiag[bra3] += 0.25 * (prob[i, 0] * prob[i, 2] * coeff2 * coeff3)
        ############################################################################

    return Hdiag


def max2sat_GT(prob, nqubits, index_to_zero):
    """
    Makes a new MAX2SAT problem with identical structure to `prob` but with
    bitstrings permuted so `index_to_zero` is mapped the zero bitstring
    (`index_to_zero` should be an int whose binary representation is the bitstring`)
    """
    idxz_bits = [i for i in range(nqubits) if Z(index_to_zero, i)[0] == -1]  # find the `1`s in the bitstring
    newprob = np.array(prob)  # make a new problem array of same shape

    for i in range(newprob.shape[0]):  # go through each clause

        if newprob[i][1] in idxz_bits:  # if the clause bits are 1 in the bitstring, invert the clause signs
            newprob[i, 0] *= -1
        if newprob[i][3] in idxz_bits:
            newprob[i, 2] *= -1

    return newprob


def generate_max2sat(nqubits, nclauses, ensure_unique_solution=True, zero_solution=True):
    if (not ensure_unique_solution) and zero_solution:
        raise ValueError("Can't make the solution zero if the solution is not unique")

    while True:
        prob = max2sat_prob(nqubits=nqubits, nclauses=nclauses)  # get a candidate problem
        if not ensure_unique_solution: break  # if we don't need unique solution, we are done
        probvals = max2sat_values(prob=prob, nqubits=nqubits)  # calculate the problem values
        nsols = probvals[probvals == np.min(probvals)].shape[0]  # coun't the number of solutions
        if nsols == 1: break  # if solution is unique, we are done
        # if we are here, we didn't find a unique solution problem, so we'll go back around

    if zero_solution:
        sol = np.argmin(probvals)  # if we are here, solution is unique, so get it
        prob = max2sat_GT(prob, nqubits, sol)  # transform the problem so solution is the zero bitstring

    return prob


if __name__ == '__main__':
    num_qubits = 4
    num_clauses = 3 * num_qubits
    generate_max2sat(nqubits= )