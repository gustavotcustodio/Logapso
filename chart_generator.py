import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from src.ExperimentResult import ExperimentResult

N_RUNS = 4
ALGORITHMS = ['pso', 'gapso', 'logapso']
CHARTS_FOLDER = 'data/experiments_results/charts'
COLORS = {
    'pso': 'blue',
    'gapso': 'green',
    'logapso': 'red',
  }
N_EXPERIMENTS = 225


def get_filename(info: dict, id_exp: int) -> str:
    paramname = 'params%d' % id_exp
    if 'dataset' in info:
        return '%s_%s_%d_%s.jpg' % (info['dataset'].split('.')[0], info['fitness'],
                                    info['n_clusters'], paramname)
    else:
        return '%s_%s.jpg' % (info['fitness'], paramname)


def plot_chart(data: dict, fullfilename: str, n_iter = 500) -> None:
    sns.set_theme(style="darkgrid")
    sns.set(rc={'axes.facecolor':'lightblue',
                'figure.facecolor':'lightgrey'})
    plt.figure()

    for algorithm in ALGORITHMS:
        p = sns.lineplot(data=data[algorithm][:n_iter], color=COLORS[algorithm])
        p.set_xlabel("Iteration", fontsize = 10)
        p.set_ylabel("Fitness", fontsize = 10)

    plt.tight_layout()
    plt.legend(ALGORITHMS)
    plt.savefig(os.path.join(CHARTS_FOLDER, fullfilename), dpi=380)
    plt.close()
    print('%s salvo com sucesso.' % fullfilename)


def main():
    file_n_iters = open('iters.json')
    iters = json.load(file_n_iters)
    file_n_iters.close()

    for i in range(1, N_EXPERIMENTS+1):
        try:
            experiment = ExperimentResult(f'params{i}', N_RUNS)
            experiment.read_experiment_info()
            experiment.calc_average_fitness_progression()

            data = {}
            for algorithm in ALGORITHMS:
                data[algorithm] = experiment.results[algorithm].avg
            fullfilename = get_filename(experiment.experiment_info, i)
            plot_chart(data, fullfilename, n_iter=iters[str(i)])
        except Exception as e:
            print("Error %s" % e)


if __name__ == '__main__':
    main()
