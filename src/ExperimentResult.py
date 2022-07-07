import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from src.AlgorithmResult import AlgorithmResult

CHART_DIRECTORY = 'data/experiments_results/charts2'
TABLES_FOLDER = 'data/experiments_results/tables'
EXPERIMENTS_FOLDER = './data/experiments_results/'
ALGORITHMS = ['pso', 'logapso', 'gapso']

class ExperimentResult:

    def __init__(self, params_set: str, n_runs: int):
        self.params_set = params_set
        self.n_runs = n_runs
        self.results = {}
        self.experiment_info = {}

    def read_experiment_info(self):
        fullfilename = os.path.join('paramsfiles', self.params_set +'.yml')
        with open(fullfilename, 'r') as current_file:
            self.experiment_info = yaml.safe_load(current_file)

    def get_experiment_info(self):
        if 'dataset' in self.experiment_info:
            self.experiment_info['dataset'] = self.experiment_info['dataset'].split('.')[0]
            return '%s %s nclusters: %d' % (self.experiment_info['fitness'], self.experiment_info['dataset'], self.experiment_info['n_clusters'])
        else:
            return self.experiment_info['fitness']

    def read_fitness_progression(self, filename: str):
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

    def calc_average_fitness_progression(self):
        for algorithm in ALGORITHMS:
            algorithm_fitness = []

            for current_run in range(1, self.n_runs+1):
                full_filename = os.path.join(
                    EXPERIMENTS_FOLDER, '%s_%s_run%d' % (algorithm, self.params_set, current_run)
                )
                fitness_progression = self.read_fitness_progression(full_filename)

                algorithm_fitness.append(fitness_progression)
            self.results[algorithm] = AlgorithmResult(algorithm, algorithm_fitness)

    def plot_chart(self):
        filename = self.params_set
        data = self.results
        plt.figure()
        # fig, ax = plt.subplots()
        for algorithm in ALGORITHMS:
            # plt.xlabel('Time(ms)')
            plt.plot([i for i in range(data[algorithm].avg.shape[0])], data[algorithm].avg)
        plt.ylabel('fitness')
        plt.title(self.get_experiment_info())
        plt.legend(ALGORITHMS)
        plt.savefig(os.path.join(CHART_DIRECTORY, f'{filename}.jpg'))
        plt.close()
        print('Plot successful.')

def format_table_results():
    return


def create_table(experiments: list, filename: str):
    table_experiment = []
    i = 0

    for experiment in experiments:
        # The current algorithm result being added to the table
        curr_alg = ALGORITHMS[i % len(ALGORITHMS)]

        experiment_info = {}
        experiment_info.update(experiment.experiment_info)
        del experiment_info['ga']
        del experiment_info['pso']
        experiment_info.update(experiment.experiment_info['ga'])
        experiment_info.update(experiment.experiment_info['pso'])
        experiment_info.update({"Algorithm": curr_alg, "Fitness": experiment.results[curr_alg].avg_best})
        table_experiment.append(experiment_info)
        i += 1

    df_latex = pd.DataFrame(table_experiment)
    df_latex.to_latex(filename+'.tex', index=False, multirow=True, multicolumn=True,
                escape=False, caption="zzzzzzzzzzzzzzzz")


def add_to_result_dict(result, result_dict):
    if 'dataset' in result.experiment_info:
        dataset = result.experiment_info['dataset'].split('.')[0]
        fitnessfunc = result.experiment_info['fitness'] + dataset
    else:
        fitnessfunc = result.experiment_info['fitness']

    if fitnessfunc not in result_dict:
        result_dict[fitnessfunc] = []
    result_dict[fitnessfunc].append(result)


if __name__ == '__main__':
    result_dict = {}

    for i in range(1, 226):
        try:
            er = ExperimentResult(f'params{i}', 4)
            er.read_experiment_info()
            er.calc_average_fitness_progression()

        except:
            pass

    # for k in result_dict.keys():
    #     fullfilename = os.path.join(TABLES_FOLDER, f'params{k}')
    #     create_table(result_dict[k], fullfilename)
