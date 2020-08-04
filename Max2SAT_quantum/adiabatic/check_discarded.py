import numpy as np

if __name__ == '__main__':
    n_list = [11]
    for n in n_list:
        print("-----",n,"------")
        if n > 8:
            if n == 9:
                data = np.genfromtxt('final_adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)
            else:
                data = np.genfromtxt('adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)
            times = data[:, 1].astype(int)
            success = data[:, 2]
        else:
            times = np.loadtxt("new_adiabatic_time_n_" + str(n) + ".txt")
            success = np.ones(10000)
        new_times = []
        num_discarded = 0
        for i, time in enumerate(times):
            failed = success[i] == 'False'
            if time > 0:
                new_times.append(time)
                if failed:
                    print("instance", i, "failed but not discarded")
            else:
                if not failed:
                    print("instance", i, "discarded but not failed")
                num_discarded += 1
                print("instance", i, time, "increase n_steps")
                # n=9, i=2361, T>T_max(16384)
        av_time = np.mean(new_times)
        print("n=" + str(n), "average time until P > 0.99: ", av_time, "discarded instances:", num_discarded)
