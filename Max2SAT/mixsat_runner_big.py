import subprocess
import time
import numpy as np


def get_instances():
    """returns array of instance names, array of corresponding n"""
    instance_data = np.genfromtxt('m2s_nqubits.csv', delimiter=',', skip_header=1, dtype=str)       #  can add _noGT_nondg on end
    return instance_data[:, 0], instance_data[:, 1]


if __name__ == '__main__':
    instance_names, instance_n_bits_str = get_instances()

    n = 60 #11
    num_instances = 1000

    runtimes = np.zeros(num_instances)
    output_runtimes = np.zeros(num_instances)

    for i in range(num_instances):
        instance_name = str(n) + "_" + str(i)
        time_start_inst = time.process_time()
        result = subprocess.run(['./../../mixsat/complete', "./../../instances_big/"+instance_name+".gz"], stdout=subprocess.PIPE)        #  can add _noGT_nondg on end
        time_end_inst = time.process_time()
        runtime = time_end_inst - time_start_inst
        runtimes[i] = runtime

        output = str(result.stdout)
        string_slice_start = output.find('state_visited ')
        output2 = output[string_slice_start:]
        string_start_index = output2.find('time ') + 5
        output3 = output2[string_start_index:]
        string_end_index = output3.find('s')
        output_time = float(output3[:string_end_index])
        output_runtimes[i] = output_time
        if i % 100 == 0:
            print(i)

    with open("big_runtimes_"+str(n)+".txt", "ab") as f:         # saves runtimes using time.process_time()        #  can add _noGT_nondg in middle
        f.write(b"\n")
        np.savetxt(f, runtimes)

    with open("big_output_runtimes_"+str(n)+".txt", "ab") as f:         # saves counts       #  can add _noGT_nondg in middle
        f.write(b"\n")
        np.savetxt(f, output_runtimes)
