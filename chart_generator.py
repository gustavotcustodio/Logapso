import os
import numpy as np
import yaml
import matplotlib.pyplot as plt


def read_fitness_progression(filename):
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


def read_experiment_info(filename):
    fullfilename = os.path.join('paramsfiles', filename+'.yml')
    with open(fullfilename, 'r') as current_file:
        experiment_info = yaml.safe_load(current_file)

    if 'dataset' in experiment_info:
        dataset = experiment_info['dataset'].split('.')[0]
        n_clusters = experiment_info['n_clusters']
        return '%s %s nclusters: %d' % (experiment_info['fitness'], dataset, n_clusters)
    else:
        return experiment_info['fitness']


def plot_chart(data, filename):
    chart_directory = 'data/experiments_results/charts'

    plt.figure()
    # fig, ax = plt.subplots()
    algorithms = list(data.keys())
    for algorithm in algorithms:
        # plt.xlabel('Time(ms)')
        plt.plot([i for i in range(data[algorithm].shape[0])], data[algorithm])
    plt.ylabel('fitness')
    plt.title(read_experiment_info(filename))
    plt.legend(algorithms)
    plt.savefig(os.path.join(chart_directory, f'{filename}.jpg'))
    plt.close()


def fix_filenames(filename):
    return '_'.join(filename.split('_')[1:])


def filter_mismatched_files(filename, filelist):

    if f'pso_{filename}' in filelist and \
            f'logapso_{filename}' in filelist:
        return True
    else:
        return False


def main():
    total_runs = 4
    # algorithms = ['pso', 'logapso']
    experiments_folder = 'data/experiments_results'

    data_to_plot = {}

    fitness_vals_paramset = []  # Fitness values for parameter set

    filelist_folder = sorted(os.listdir(experiments_folder))
    filelist = map(fix_filenames, filelist_folder)
    filelist = filter(lambda f: filter_mismatched_files(f, filelist_folder), filelist)
    filelist = list(dict.fromkeys(filelist))

    for current_filename in filelist:
        print(current_filename)
        # Current parameters set
        current_param = current_filename.split('_')[0]
        data_to_plot[current_param] = {}

        # Current run for the parameters set
        current_run = int(current_filename.split('_')[-1].replace('run', ''))

        for algorithm in ['pso', 'logapso', 'gapso']:
            full_filename = os.path.join(experiments_folder ,
                                         '%s_%s_run%d' % (algorithm, current_param, current_run))
            fitness_values = read_fitness_progression(full_filename)

            fitness_vals_paramset.append(fitness_values)

            if current_run == total_runs:

                data_to_plot[current_param][algorithm] = np.array(np.mean(fitness_vals_paramset, axis=0))
                fitness_vals_paramset = []  # Reset the fitness values for the next paramset

        if current_run == total_runs:
            plot_chart(data_to_plot[current_param], current_param)


if __name__ == '__main__':
    main()
