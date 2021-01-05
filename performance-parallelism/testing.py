import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    
    n_experiments = 10
    n_treads = 128

    # compilando o programa
    os.system('gcc matrix-multi-openMP.c -o matrix-multi-openMP -fopenmp')

    # dicionários para guardar os tempos médios, os speedruns e as eficácias de cada running/n-threads
    avg_times = {}
    speedruns = {}
    efficiency = {}
    for experiment in range(1, n_experiments+1):
        for n in range(1, n_treads+1):
            # setando o número de threads na API OpenMP
            # os.sytem('export OMP_NUM_THREADS='+str(n))
            os.system('./matrix-multi-openMP '+str(n)+'.txt')

    for n in range(1, n_treads+1):
        times_n_file = open(str(n)+'.txt', 'r')
        times_n = [float(line[:-1]) for line in times_n_file.readlines()]
        avg_times[n] = np.mean(times_n)
        speedruns[n] = avg_times[n] / avg_times[1]
        efficiency[n] = speedruns[n] / n

    # removendo os arquivos .txt auxiliares
    os.system('rm *.txt')

    # plotando gráficos
    plt.title('Expected Time')
    plt.scatter(list(range(1, n_treads+1)), list(avg_times.values()), color='navy')
    plt.axvline(x=8, color='black')
    plt.xlabel('Threads')
    plt.ylabel('Time (sec)')
    plt.grid()
    plt.show()
    plt.clf()

    plt.title('Speedrun')
    plt.scatter(list(range(1, n_treads+1)), list(speedruns.values()), color='crimson')
    plt.axvline(x=8, color='black')
    plt.xlabel('Threads')
    plt.grid()
    plt.show()
    plt.clf()

    plt.title('Efficiency')
    plt.scatter(list(range(1, n_treads+1)), list(efficiency.values()), color='lightseagreen')
    plt.axvline(x=8, color='black')
    plt.xlabel('Threads')
    plt.grid()
    plt.show()
    plt.clf()