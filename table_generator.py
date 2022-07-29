import os
import subprocess
import pandas as pd
from src.ExperimentResult import ExperimentResult

N_RUNS = 4
ALGORITHMS = ['pso', 'gapso', 'logapso']
TABLES_FOLDER = 'data/experiments_results/tables'
N_EXPERIMENTS = 225

EXPERIMENTS_RESULTS_FOLDER = './data/experiments_results'


def create_multirow(cell_value: str | float, n_rows: int) -> list:
    multiple_rows = ["\\multirow{%d}{*}{%s}" % (n_rows, cell_value)]
    # add the empty rows after 
    multiple_rows += [''] * (n_rows - 1)
    return multiple_rows


def format_results_table(experiment: ExperimentResult) -> dict:
    exp_info = experiment.experiment_info
    formatted_results = {}

    if 'dataset' in exp_info:
        formatted_results['Cluster eval.'] = create_multirow(
            exp_info['fitness'].replace('_', ' ').title(), len(ALGORITHMS)
        )
        formatted_results['\\# clusters'] = create_multirow(
            exp_info['n_clusters'], len(ALGORITHMS)
        )
    else:
        formatted_results['Benchmark function'] = create_multirow(
            exp_info['fitness'].replace('_', ' ').title(), len(ALGORITHMS)
        )
        formatted_results['\\# dims'] = create_multirow(
            exp_info['pso']['particle_length'], len(ALGORITHMS)
        )

    formatted_results['Optim. algorithm'] = [algorithm.upper()
                                             for algorithm in ALGORITHMS]
    formatted_results['Avg. fitness'] = [experiment.results[algorithm].avg_best
                                         for algorithm in ALGORITHMS]
    formatted_results['Std. dev.'] = [experiment.results[algorithm].std
                                      for algorithm in ALGORITHMS]
    formatted_results['Pop. size'] = create_multirow(
        exp_info['pso']['swarm_size'], len(ALGORITHMS)
    )
    formatted_results['$\\phi_{1}$'] = create_multirow(
        exp_info['pso']['acc1'], len(ALGORITHMS)
    )
    formatted_results['$\\phi_{2}$'] = create_multirow(
        exp_info['pso']['acc2'], len(ALGORITHMS)
    )
    formatted_results['w'] = create_multirow(
        exp_info['pso']['inertia'], len(ALGORITHMS)
    )
    formatted_results['Mutation rate'] = create_multirow(
        exp_info['ga']['mutation_rate'], len(ALGORITHMS)
    )
    return formatted_results


def main():

    for i in range(1, N_EXPERIMENTS+1):
        try:
            experiment = ExperimentResult(f'params{i}', N_RUNS)
            experiment.read_experiment_info()
            experiment.calc_average_fitness_progression()

            experiment_name = "%s_params%03d" % (
                experiment.experiment_info['fitness'], i)

            if 'dataset' in experiment.experiment_info:
                dataset = experiment.experiment_info['dataset'].replace('.data', '')
                experiment_name = "%s_%s" % (dataset, experiment_name)
                caption_table = "Results of experiments using the %s dataset" % dataset
            else:
                caption_table = "Results of experiments with benchmark functions"

            filename = os.path.join(TABLES_FOLDER, f'{experiment_name}.tex')

            df_latex = pd.DataFrame(format_results_table(experiment))
            df_latex.to_latex(filename, index=False, multirow=True, multicolumn=True,
                              escape=False, caption=caption_table)
            print("%s has been successfully saved." % experiment_name)
        except Exception as e:
            print("Erro: %s" % e)

    subprocess.call(['sh', './merge_tex_tables'])
    print('Tables merged successfully.')

if __name__ == "__main__":
    main()
