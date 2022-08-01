from subprocess import run
import re
import statistics
import matplotlib.pyplot as plt
import numpy as np

def calculate_statistics(times, program_name, size):
    times = list(map(float, times))
    f.write('\nMEDIA ARITMETICA: ' + str(statistics.mean(times)) + '\n')
    f.write('MEDIANA: ' + str(statistics.median(times)) + '\n')
    f.write('VARIANZA: ' + str(statistics.variance(times)) + '\n')
    f.write('DEVIAZIONE STANDARD: ' + str(statistics.stdev(times)) + '\n\n\n')
    f.flush()

    # Calcolo delle statistiche
    labels = range(1, 21)

    if(size == 16384):
        plt.title(program_name.upper())
        plt.plot(times, 'o-')
        plt.grid(True)
        plt.xlabel('Iteration number')
        plt.xticks(np.arange(0, 20, step=1), labels)
        plt.ylabel('Execution Time (sec)')
        plt.ylim((statistics.mean(times) / 2, statistics.mean(times) * 2))
        plt.savefig('results/images/' + program_name + "_" + str(size) + '.png')
        plt.clf()

programs = [
    "./build/src/matrix_addition/acc_matrix_addition",
    "./build/src/matrix_addition/usm_matrix_addition",
    "./build/src/matrix_addition/usm_matrix_addition_shared",

    "./build/src/matrix_multiplication/acc_matrix_multiplication",
    "./build/src/matrix_multiplication/usm_matrix_multiplication",
    
    "./build/src/data_dependency_test/usm_data_dependency_test_v2",
    "./build/src/data_dependency_test/acc_data_dependency_test_v2",
]
times = []
matrix_sizes = [1024, 2048, 4096, 8192, 16384]

# Per ogni programma
for program in programs:
    program_name = program.split("/")[4]

    # Creo un file diverso
    with open('results/' + program_name + '.txt', 'w') as f:
        f.write(program_name.upper() + "\n")
        f.flush()

        # Se è data_dependency non lo lancio con 
        if("data_dependency" in program_name):
            # Eseguo X volte il programma
            for i in range(0, 20):
                output = run(program, capture_output=True).stdout
                output = re.findall('[0-9]+\.[0-9]*', str(output))[0]

                # Aggiungo all'array di tempi il tempo di esecuzione (Per il calcolo di media, mediana, ecc...)
                times.append(output)

                # Scrivo nel file il tempo di esecuzione
                f.write(str(i + 1) + '- ' + output + '\n')
                f.flush()
            calculate_statistics(times, program_name, 16384)
            # Svuoto tutto
            times = []
        else:
            for size in matrix_sizes:
                f.write("Size: " + str(size) + "\n")

                # Eseguo X volte il programma
                for i in range(0, 20):
                    output = run([program, str(size), str(size)], capture_output=True).stdout
                    output = re.findall('[0-9]+\.[0-9]*', str(output))[0]

                    # Aggiungo all'array di tempi il tempo di esecuzione (Per il calcolo di media, mediana, ecc...)
                    times.append(output)

                    # Scrivo nel file il tempo di esecuzione
                    f.write(str(i + 1) + '- ' + output + '\n')
                    f.flush()
                calculate_statistics(times, program_name, size)
                # Svuoto tutto
                times = []

        print(program_name + ' has finished')
        f.close()