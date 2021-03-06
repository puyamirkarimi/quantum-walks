import numpy as np


def get_2sat_formula(instance_name):
    out = np.loadtxt("../../instances_original/" + instance_name + ".m2s")  # path of instance files in adam's format
    return out.astype(int)


def get_instances():
    """returns array of instance names, array of corresponding n"""
    data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)    # path of csv file
    return data[:, 0], data[:, 1]


def make_file(name, formula, num_var, num_cls):
    with open("../../instances_wcnf/"+name+'.txt', 'w') as file:    # path of output instance files in WCNF format
        file.write('p wcnf '+str(num_var)+' '+str(num_cls)+'\n')
        for clause in range(num_cls):
            sign_1 = formula[clause, 0]
            sign_2 = formula[clause, 2]
            v_1 = formula[clause, 1] + 1
            v_2 = formula[clause, 3] + 1
            file.write("1 "+str(int(sign_1*v_1))+" "+str(int(sign_2*v_2))+" 0\n")


if __name__ == "__main__":
    instance_names, instance_n_bits_str = get_instances()
    for instance_num, instance_name in enumerate(instance_names):
        sat_formula = get_2sat_formula(instance_name)

        ############ DON'T USE [0:-2] BELOW FOR THE NORMAL INSTANCES ############
        # n = int(instance_n_bits_str[instance_num][0:-2])  # number of variables
        n = int(instance_n_bits_str[instance_num])  # number of variables

        m = len(sat_formula[:, 0])  # number of clauses

        make_file(instance_name, sat_formula, n, m)

        if instance_num % 100 == 0:
            print(instance_num)