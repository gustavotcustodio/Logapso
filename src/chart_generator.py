import os
import numpy as np
import matplotlib.pyplot as plt


def get_fitness_vals_run(filename):
    fitness_vals_run = []
    with open(filename, 'r') as current_file:
        line = current_file.readline()
        iteration = 0

        while(line):
            if 'Best fitness' in line:
                fitness = float(line.split(": ")[1])
                fitness_vals_run.append(fitness)

                iteration += 1
            line = current_file.readline()
    return fitness_vals_run


def plot_graph(data, filename):
    plt.plot([i for i in range(data.shape[0])], data)
    plt.title(filename)
    plt.show()


def main():
    total_runs = 5
    algorithms = ['pso', 'logapso']

    data_to_plot = {}

    for alg in algorithms:
        fitness_vals_paramset = []

        for current_file in sorted(os.listdir('experiments_results')):

            param = current_file.split('_')[1]
            n_run = int(current_file.split('_')[-1].replace('run', ''))

            filename = os.path.join('./experiments_results',
                                    '%s_%s_run%d' % (alg, param, n_run))
            fitness_vals_run = get_fitness_vals_run(filename)

            fitness_vals_paramset.append(fitness_vals_run)

            if n_run == total_runs:
                dictkey = f'{alg}_{param}'
                data_to_plot[dictkey] = np.array(np.mean(fitness_vals_paramset,
                                                         axis=0))
                fitness_vals_paramset = []

                plot_graph(data_to_plot[dictkey], dictkey)


if __name__ == '__main__':
    main()
