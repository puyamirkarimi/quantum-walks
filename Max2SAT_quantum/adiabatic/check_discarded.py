import numpy as np

if __name__ == '__main__':
    n_list = [9]
    for n in n_list:
        if n > 8:
            data = np.genfromtxt('new_new_adiabatic_time_n_' + str(n) + '.csv', delimiter=',', skip_header=1, dtype=str)
            times = data[:, 1].astype(int)
            success = data[:, 2]
        else:
            times = np.loadtxt("new_adiabatic_time_n_" + str(n) + ".txt")
        new_times = []
        num_discarded = 0
        for i, time in enumerate(times):
            if time > 0:
                new_times.append(time)
                if not success[i]:
                    print("failed but not discarded")
            else:
                num_discarded += 1
        av_time = np.mean(new_times)
        print("n=" + str(n), "average time until P > 0.99: ", av_time, "discarded instances:", num_discarded)
