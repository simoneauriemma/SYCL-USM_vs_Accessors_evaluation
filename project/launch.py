from subprocess import run
import re
import statistics
import matplotlib.pyplot as plt
import numpy as np

programs = [
    "./build/matrix_addition/acc_matrix_addition",
    "./build/matrix_addition/usm_matrix_addition",
    "./build/matrix_addition/usm_matrix_addition_shared",

    "./build/matrix_multiplication/acc_matrix_multiplication",
    "./build/matrix_multiplication/usm_matrix_multiplication",
    
    "./build/stress_test/usm_stress_test",
    "./build/stress_test/acc_stress_test",
]
times = []

# Per ogni programma
for program in programs:
    program_name = program.split("/")[3]

    # Creo un file diverso
    with open('results/' + program_name + '.txt', 'w') as f:
        f.write(program_name.upper() + "\n")
        f.flush()

        # Eseguo X volte il programma
        for i in range(0, 20):
            output = run(program, capture_output=True).stdout
            output = re.findall('[0-9]+\.[0-9]*', str(output))[0]

            # Aggiunto all'array di tempi il tempo di esecuzione (Per il calcolo di media, mediana, ecc...)
            times.append(output)

            # Scrivo nel file il tempo di esecuzione
            f.write(str(i + 1) + '- ' + output + '\n')
            f.flush()

        # Calcolo delle statistiche
        times = list(map(float, times))
        labels = range(1, 21)

        plt.title(program_name.upper())
        plt.plot(times, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration number')
        plt.xticks(np.arange(0, 20, step=1), labels)
        plt.ylabel('Execution Time (sec)')
        plt.ylim((statistics.mean(times) / 2, statistics.mean(times) * 2))
        plt.savefig('results/images/' + program_name + '.png')
        plt.clf()

        f.write('\nMEDIA ARITMETICA: ' + str(statistics.mean(times)) + '\n')
        f.write('MEDIANA: ' + str(statistics.median(times)) + '\n')
        f.write('VARIANZA: ' + str(statistics.variance(times)) + '\n')
        f.write('DEVIAZIONE STANDARD: ' + str(statistics.stdev(times)) + '\n')

        # Svuoto tutto
        times = []
        print(program_name + ' has finished')
        f.close()
