import numpy as np


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances_big_adam_format/" + instance_name + ".gz")  # path of instance files in adam's format
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return data[:, 0], data[:, 1]


def make_file(name, formula, num_var, num_cls):
    with open("../../instances_big_wcnf/"+name+'.txt', 'w') as file:    # path of output instance files in WCNF format
        file.write('p wcnf '+str(num_var)+' '+str(num_cls)+'\n')
        for clause in range(num_cls):
            sign_1 = formula[clause, 0]
            sign_2 = formula[clause, 2]
            v_1 = formula[clause, 1] + 1
            v_2 = formula[clause, 3] + 1
            file.write("1 "+str(int(sign_1*v_1))+" "+str(int(sign_2*v_2))+" 0\n")


if __name__ == "__main__":
    n = 65
    m = 3*n
    num_instances = 1000

    for instance_num in range(num_instances):
        instance_name = str(n) + "_" + str(instance_num)
        sat_formula = get_2sat_formula(instance_name)

        make_file(instance_name, sat_formula, n, m)

        if instance_num % 100 == 0:
            print(instance_num)