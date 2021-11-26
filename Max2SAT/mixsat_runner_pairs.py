import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)       #  can add _noGT_nondg on end
    return instance_data[:, 0], instance_data[:, 1]


if __name__ == '__main__':
    runtimes = np.zeros(10000)
    runtimes_transformed = np.zeros(10000)

    states_count = np.zeros(10000)
    states_count_transformed = np.zeros(10000)

    n = 5

    num_instances = 10000

    for i in range(num_instances):
        instance_name = str(n) + "_" + str(i)
        instance_name_transformed = str(n) + "_" + str(i) + "_transformed"

        time_start_inst = time.process_time()
        result = subprocess.run(['./../../mixsat/complete', './../../instances_pairs_5/'+instance_name+'.gz'], stdout=subprocess.PIPE)
        time_end_inst = time.process_time()
        runtime = time_end_inst - time_start_inst
        runtimes[i] = runtime
        output = str(result.stdout)
        string_start_index = output.find('state_visited ') + 14
        string_end_index = output.find(' pruned')
        states_visited = int(output[string_start_index: string_end_index])
        states_count[i] = states_visited

        time_start_inst = time.process_time()
        result = subprocess.run(['./../../mixsat/complete', './../../instances_pairs_5/'+instance_name_transformed+'.gz'], stdout=subprocess.PIPE)
        time_end_inst = time.process_time()
        runtime = time_end_inst - time_start_inst
        runtimes_transformed[i] = runtime
        output = str(result.stdout)
        string_start_index = output.find('state_visited ') + 14
        string_end_index = output.find(' pruned')
        states_visited = int(output[string_start_index: string_end_index])
        states_count_transformed[i] = states_visited

    with open("pairs_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes
        f.write(b"\n")
        np.savetxt(f, runtimes)

    with open("pairs_counts_"+str(n)+".txt", "ab") as f:         # saves counts
        f.write(b"\n")
        np.savetxt(f, states_count)
    
    with open("pairs_transformed_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes
        f.write(b"\n")
        np.savetxt(f, runtimes_transformed)

    with open("pairs_transformed_counts_"+str(n)+".txt", "ab") as f:         # saves counts
        f.write(b"\n")
        np.savetxt(f, states_count_transformed)
