import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    n_experiments = 3
    n_treads = [1, 2, 4, 8, 16, 32, 64, 128]

    # compilando o programa
    os.system('gcc matrix-multi-openMP.c -o matrix-multi-openMP -fopenmp')

    # dicionários para guardar os tempos médios, os speed-up e as eficácias de cada running/n-threads
    avg_times = {}
    speedups = {}
    efficiency = {}
    for n in n_treads:
        # realiza experimentos distintos 
        for experiment in range(1, n_experiments+1):
            os.system('./matrix-multi-openMP '+str(n)+' '+str(n)+'.txt')
        
        times_n_file = open(str(n)+'.txt', 'r')
        times_n = [float(line[:-1]) for line in times_n_file.readlines()]
        avg_times[n] = np.mean(times_n)
        speedups[n] = avg_times[1] / avg_times[n]
        efficiency[n] = speedups[n] / n
        # os.system('rm '+str(n)+'.txt')

    # plotando gráficos
    plt.title('Expected Time')
    plt.scatter(list(range(1, n_treads+1, step)), list(avg_times.values()), color='navy')
    plt.axvline(x=8, color='black')
    plt.xlabel('Threads')
    plt.ylabel('Time (sec)')
    plt.grid()
    # plt.show()
    plt.savefig('expected_time.png')
    plt.clf()

    plt.title('Speedup')
    plt.scatter(list(range(1, n_treads+1, step)), list(speedups.values()), color='crimson')
    plt.axvline(x=8, color='black')
    plt.xlabel('Threads')
    plt.grid()
    # plt.show()
    plt.savefig('speedup.png')
    plt.clf()

    plt.title('Efficiency')
    plt.scatter(list(range(1, n_treads+1, step)), list(efficiency.values()), color='lightseagreen')
    plt.axvline(x=8, color='black')
    plt.xlabel('Threads')
    plt.grid()
    # plt.show()
    plt.savefig('efficiency.png')
    plt.clf()
