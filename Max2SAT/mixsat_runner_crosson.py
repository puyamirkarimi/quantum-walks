import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)
    return instance_data[:, 0], instance_data[:, 1].astype(int)


if __name__ == '__main__':
    instance_names = np.array(range(137))
    runtimes = np.zeros(137)
    states_count = np.zeros(137)

    for instance_name_int in instance_names:
        instance_name = str(instance_name_int)
        time_start_inst = time.time()
        result = subprocess.run(['./../../mixsat/complete', './../../instances_crosson_dimacs/'+instance_name+'.txt'], stdout=subprocess.PIPE)
        time_end_inst = time.time()
        runtime = time_end_inst - time_start_inst
        runtimes[instance_name_int] = runtime
        #print(time_end_inst - time_start_inst)
        output = str(result.stdout)

        string_start_index = output.find('state_visited ') + 14
        string_end_index = output.find(' pruned')
        states_visited = int(output[string_start_index : string_end_index])
        states_count[instance_name_int] = states_visited


        # print(output)
        # failed = b'-' in output

    #print("average runtime: ", average_runtime)

    # with open("crosson_runtimes.txt", "ab") as f:         # saves runtimes using time.time()
    #     f.write(b"\n")
    #     np.savetxt(f, runtimes)

    with open("crosson_counts.txt", "ab") as f:         # saves counts
        f.write(b"\n")
        print(states_count)
        np.savetxt(f, states_count)
