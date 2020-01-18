import numpy as np


def get_2sat_formula():
    out = np.loadtxt("../../instances/wwrencmnhuihioqjuvuhptuwkjdwac.m2s").astype(int)
    return out


def get_m_n(array):
    clauses = len(array[:, 0])  # number of clauses
    max_lit_1 = np.amax(array[:, 1]) + 1
    max_lit_2 = np.amax(array[:, 3]) + 1
    variables = max(max_lit_1, max_lit_2)  # number of variables
    return clauses, variables


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
        i_1 = problem[j, 1]  # 1st non-zero element in s[j,:]
        i_2 = problem[j, 3]  # 2nd non-zero element in s[j,:]
        z[j] = s[j, i_1] * v[i_1] + s[j, i_2] * v[i_2]

    converged = False
    while not converged:
        converged = True
        for i in range(n):
            old_v_i = np.array(v[i])
            for j in range(m):
                if s[j, i] != 0:  # is there a faster way of finding non-zero elements?
                    z[j] -= s[j, i] * v[i]
            for j in range(m):  # can combine loops
                v[i] -= (s[j, i] / 8) * z[j]
            v[i] /= np.linalg.norm(v[i])
            if i == 3:
                print("v:", v[2][0:3])
            for j in range(m):
                if s[j, i] != 0:  # is there a faster way of finding non-zero elements?
                    z[j] += s[j, i] * v[i]
            if np.amax(np.absolute(old_v_i - v[i])) > convergence_criteria:
                converged = False
    return v


def get_k(n):
    k = np.ceil(np.sqrt(n * 2))
    if k % 4:
        k = 4 * np.ceil(k / 4)  # round up to nearest multiple of 4
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
            r_squared += vector[i] ** 2
        if 1 > r_squared > 0.01:
            inside_sphere = True
    vector /= np.linalg.norm(vector)  # normalise vector to 1
    return vector

############## old code above ###############


class SatProblem:
    def __init__(self, formula):
        self.m, self.n = get_m_n(formula) # number of clauses, variables
        self.nnz = 2 * self.m   # number of literals
        self.var_len = # array of length n which stores the number of times a variable is used
        self.cls_len = # array of length m which stores the number of literals in each clause
        self.cls = # the clause to variable sparse matrix, i.e. each element contains a list of the variables in that clause
        self.var = # the variable to clause sparse matrix, i.e. each element contains a list of the clauses that variable appears in



class Soln:
    def __init__(self):
        self.answer = np.zeros(n)
        self.f = 0.0
        self.x = 0
        self.lv = 0


class WStack:
    def __init__(self, prob):
        self.m = prob.m
        self.n = prob.n
        self.pivot = np.zeros(2*self.n)
        self.watched = np.zeros(2*self.n)       # 3D array?
        self.left = np.zeros(2 * self.n)        # 3D array?
        self.cls = prob.cls
        self.cls_id = np.zeros(prob.nnz + self.n + self.m)
        self.cls_end = np.zeros(self.m)         # 2D array?

class Mixsat:
    def __init__(self, prob):



def bin_maxsat(prob, num_trial):
    m = prob.m
    n = prob.n
    lv_thres = 5
    best_soln = Soln()
    curr_soln = Soln()
    wstack = WStack(prob)
    mixsat = Mixsat(prob)

    upper_bound = np.zeros(n)
    lower_bound = np.zeros(n)

    pivot = np.zeros(2 * n)
    assign = np.zeros(n)

    # wstack_update(wstack)

    lv = n
    from_parent = 1
    upper[lv] = m
    best.x = m

    state_visited = 0
    state_expand = 0
    state_prune = 0
    diff_cnt = 0

    start_time = time.time()
    while lv <= n:
        state_visited += 1
        if state_visited % 10000 == 0:
            state_mixed = state_prune + state_expand
            # some print statement

        if not from_parent:
            lit = assign[lv]
            curr_soln.x -= backtrack(wstack, lit)
            mix_backtrack(wstack, mixsat, lit)

            action = assign[lv] == pivot[lv] and wstack.watched[lv] ^ 1
        elif lv == 0:
            if curr.x < best.x:
                save_best(curr_soln, best)
            action = 0
        elif lv > lv_thres and upper_bound[lv] > best.x:
            # do mixing
            fval = 0.0
            mix_val = 0

            by_diff = 0
            if lv != n and best.x <= np.ceil(mix.dual[0][lv]):
                mix_val = np.ceil(mix.dual[0][lv])
                by_diff = 1

            if (not by_diff) and (lv != 2) and (pivot[lv + 1] != assign[lv + 1]):
                # solve the difference problem
                fval = mixsat_set_var(mixsat, wstack, lv, 1, assign, curr_soln.x)
                fval += do_mixing(mixsat)
                if np.abs(round(fval) - fval) < 1e-4:
                    mix_val = lower_bound[lv] + round(fval)
                else:
                    mix_val = lower_bound[lv] + np.ceil(fval)
                diff_cnt += 1
                if best_soln.x <= mix_val:
                    by_diff = mix_val - best_soln.x + 1

            if not (best.x <= mix_val):
                # solve the complete problem
                fval = mixsat_set_var(mixsat, wstack, lv, 0, assign, curr_soln.x)
                fval += do_mixing(mixsat)
                fval -= mixsat.eps
                mix_val = np.ceil(curr_soln.x + (fval if fval > 0 else 0))

            if best_soln.x <= mix_val:
                # prunable
                if by_diff:
                    state_prune += 1
                lower_bound[lv] = mix_val
                action = 0
            else:
                state_expand += 1
                n_trial = num_trial * (1000 * (lv + 1) if lv == n else 10)
                maxsat_rounding(mixsat, wstack, n_trial, best_soln.x-curr_soln.x, curr_soln, best_soln, start_time)
                mixsat_push(mixsat, wstack, lv, assign)
                upper[lv] = curr_soln.x+fval
                action = 1

        else:
            # from_parent and not prunable
            action = 1

        if action:
            # try going down
            assign[lv] = pivot[lv] ^ ((not wstack.watched[pivot[lv]]) or (not from_parent))

            lit = assign[lv]
            curr_soln.x += forward(wstack, lit)
            delta = mix_forward(wstack, mixsat, lit, lv)
            upper_bound[lv-1] = upper_bound[lv] + delta
            if assign[lv] == pivot[lv] or not(wstack.watched[pivot[lv]]):
                lower_bound[lv] = m

            from_parent = 1
            lv -= 1
        else:
            # try going up
            if mixsat.ov[len] - 1 == lv:
                mixsat_pop(mixsat, wstack)
            if lv != n and lower_bound[lv+1] > lower_bound[lv]:
                lower_bound[lv+1] = lower_bound[lv]

            from_parent = 0
            lv +=1

    #assert(curr_soln.x==0)
    print(best_soln.x, state_visited, state_prune, state_expand, state_prune+state_expand, wall_time_diff(wall_clock_ns(), time_st)

    return best.x






if __name__ == "__main__":
    formula = get_2sat_formula()
    prob = SatProblem(formula)
    bin_maxsat(prob)