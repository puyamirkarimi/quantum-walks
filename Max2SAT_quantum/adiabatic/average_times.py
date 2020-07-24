import numpy as np

if __name__ == '__main__':
    n_list = [5, 6, 7]
    for n in n_list:
        times = np.loadtxt("adiabatic_time_n_" + str(n) + ".txt")
        new_times = []
        num_discarded = 0
        for time in times:
            if time > 0:
                new_times.append(time)
            else:
                num_discarded += 1
        av_time = np.mean(new_times)
        print("n=" + str(n), "average time until P > 0.99: ", av_time, "discarded instances:", num_discarded)
