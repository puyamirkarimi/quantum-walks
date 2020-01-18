import numpy as np


def get_2sat_formula():
    out = np.loadtxt("../../instances/wwrencmnhuihioqjuvuhptuwkjdwac.m2s").astype(int)
    return out


def get_m_n(array):
    clauses = len(array[:, 0])  # number of clauses
    max_lit_1 = np.amax(array[:, 1]) + 1
    max_lit_2 = np.amax(array[:, 3]) + 1
    literals = max(max_lit_1, max_lit_2)  # number of literals
    return clauses, literals


# def make_stack(array):
#     out = list()
#     for element in array:
#         out.append(element)
#     return out


def mixing_method(problem):
    convergence_criteria = 0.0001
    m, n = get_m_n(problem)
    k = get_k(n)
    s = s_array(problem, m, n)
    v = np.zeros((n, k))
    z = np.zeros((m, k))
    for i in range(n):
        v[i] = random_unit_sphere_vector(k)
    for j in range(m):
        i_1 = problem[j, 1]                 # 1st non-zero element in s[j,:]
        i_2 = problem[j, 3]                 # 2nd non-zero element in s[j,:]
        z[j] = s[j, i_1]*v[i_1] + s[j, i_2]*v[i_2]

    converged = False
    while not converged:
        converged = True
        for i in range(n):
            old_v_i = np.array(v[i])
            for j in range(m):
                if s[j, i] != 0:                # is there a faster way of finding non-zero elements?
                    z[j] -= s[j, i]*v[i]
            for j in range(m):                  # can combine loops
                v[i] -= (s[j, i]/8) * z[j]
            v[i] /= np.linalg.norm(v[i])
            if i == 3:
                print("v:", v[2][0:3])
            for j in range(m):
                if s[j, i] != 0:                # is there a faster way of finding non-zero elements?
                    z[j] += s[j, i]*v[i]
            if np.amax(np.absolute(old_v_i - v[i])) > convergence_criteria:
                converged = False
    return v


def get_k(n):
    k = np.ceil(np.sqrt(n * 2))
    if k % 4:
        k = 4 * np.ceil(k/4)                    # round up to nearest multiple of 4
    return int(k)


def s_array(problem, m, n):
    s = np.zeros((m, n))
    for j in range(m):
        lit_1 = problem[j, 1]
        lit_2 = problem[j, 3]
        s[j, lit_1] = problem[j, 0]
        s[j, lit_2] = problem[j, 2]
    return s


def random_unit_sphere_vector(n_dims):
    # more efficient algorithms for this exist
    vector = np.zeros(n_dims)
    inside_sphere = False
    while not inside_sphere:
        # discard vectors outside a unit sphere
        vector = 2 * np.random.rand(n_dims) - 1
        r_squared = 0
        for i in range(n_dims):
            r_squared += vector[i]**2
        if 1 > r_squared > 0.01:
            inside_sphere = True
    vector /= np.linalg.norm(vector)        # normalise vector to 1
    return vector


if __name__ == "__main__":
    formula = get_2sat_formula()        # formula, number of clauses, number of literals
    epsilon = 0
    best = 0
    Q = list()                          # replace Q with wstack
    Q.append(formula)
    while len(Q) > 0:
        P = Q.pop()                     # new SDP root
        f_star = mixing_method(P)
        print(f_star)
        # if np.ceil(f_star - epsilon) >= best:
        Q = []