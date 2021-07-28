import os
import sys
import yaml
import numpy as np
import getopt
import functions
from pso import Pso
from logapso import Logapso
from genetic_algorithm_logapso import GeneticAlgorithmLogapso


def read_config_file(filename):
    try:
        with open(os.path.join('configfiles', filename), 'r') as stream:
            configs = yaml.load(stream, Loader=yaml.FullLoader)
            return configs
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(2)


def scale_feature(feature, scaled_min=0, scaled_max=1):
    max_val = max(feature)
    min_val = min(feature)
    return (scaled_max - scaled_min) * (feature - min_val) / \
        (max_val - min_val) + scaled_min


def read_dataset(dataset):
    path = os.path.join('datasets', dataset)
    data = np.genfromtxt(path, delimiter=',')
    np.random.shuffle(data)
    for i in range(data.shape[1] - 1):
        data[:, i] = scale_feature(data[:, i], -1, 1)
    return data


def run(optimizer, params, start_from_checkpoint=False):
    # Cluster optimization problem or benchmark function
    if 'dataset' in params:
        data = read_dataset(params['dataset'])
        n_features = data[:, :-1].shape[1]  # number of features
        fitnessfunction = functions.get_fitness_function(
            params['fitness'], data)
        particle_length = n_features * params['n_clusters']
    else:
        fitnessfunction = functions.get_fitness_function(params['fitness'])
        particle_length = params['pso']['particle_length']

    # Select the optimization algorithm
    if optimizer == 'logapso':
        chromosome_length = particle_length
        step_value = 1
        possible_genes = [-1, 0, 1]

        ga = GeneticAlgorithmLogapso(
            params['ga']['pop_size'], chromosome_length,
            params['ga']['mutation_rate'], params['ga']['n_generations'],
            fitnessfunction, step_value, possible_genes)
        pso = Logapso(
            params['pso']['swarm_size'], params['pso']['inertia'],
            params['pso']['acc1'], params['pso']['acc2'],
            params['pso']['maxiters'], step_value, fitnessfunction, ga)

    else:  # optimizer = 'pso'
        pso = Pso(
            params['pso']['swarm_size'], params['pso']['inertia'],
            params['pso']['acc1'], params['pso']['acc2'],
            params['pso']['maxiters'], fitnessfunction)

    if start_from_checkpoint:
        pso.load_checkpoint()
    else:
        pso.generate_particles(particle_length, params['pso']['lbound'],
                               params['pso']['ubound'])
    pso.run()


def main(argv):
    configfile = None
    algorithm = None
    start_from_checkpoint = False

    try:
        opts, _ = getopt.getopt(argv, "hc:a:C", [
            "configfile=", "algorithm=", "checkpoint"])

    except getopt.GetoptError:
        # Print debug info
        print('Invalid arguments.')
        sys.exit(2)

    for option, argument in opts:
        if option in ('-c', '--configfile'):
            configfile = argument
        elif option in ('-a', '--algorithm'):
            algorithm = argument
        elif option in ('-C', '--checkpoint'):
            start_from_checkpoint = True

    parameters = read_config_file(configfile)

    if configfile is None or algorithm is None:
        print('Invalid arguments.')
    else:
        run(algorithm, parameters, start_from_checkpoint)


if __name__ == '__main__':
    main(sys.argv[1:])
