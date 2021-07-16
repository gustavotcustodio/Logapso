import os
import sys
import yaml
import numpy as np
import getopt
import fitness_function
from pso import Pso


def read_config_file(filename):
    try:
        with open(os.path.join('configfiles', filename), 'r') as stream:
            configs = yaml.load(stream, Loader=yaml.FullLoader)
            return configs
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(2)


def read_dataset(dataset):
    path = os.path.join('datasets', dataset)
    data = np.genfromtxt(path, delimiter=',')
    np.random.shuffle(data)
    return data


def get_fitness_function(params):
    if params['centroid_optimization']:
        dataset = read_dataset(params['dataset'])
        return fitness_function.get_fitness_function(
            params['fitness'], dataset=dataset)
    else:
        return fitness_function.get_fitness_function(params['fitness'])


def run_pso(params):
    swarm_size = params['pso']['swarm_size']
    inertia = params['pso']['inertia']
    acc1 = params['pso']['acc1']
    acc2 = params['pso']['acc2']
    max_iters = params['pso']['max_iters']

    if 'dataset' in params:
        data = read_dataset(params['dataset'])
        fitnessfunction = fitness_function.get_fitness_function(
            params['fitness'], data)
        particle_length = data[:, :-1].shape[1] * len(np.unique(data[:, -1]))
    else:
        fitnessfunction = fitness_function.get_fitness_function(
            params['fitness'])
        particle_length = params['pso']['particle_length']

    pso = Pso(swarm_size, inertia, acc1, acc2, max_iters, fitnessfunction)
    pso.generate_particles(particle_length, lbound=-1, ubound=1)
    pso.run()


def run(algorithm, params):
    try:
        if algorithm == 'pso':
            run_pso(params)
    except Exception:
        print('Config file incorrectly formatted.')
        sys.exit(2)


def main(argv):
    configfile = None
    algorithm = None

    try:
        opts, args = getopt.getopt(
            argv, "hc:a:", ["configfile=", "algorithm="])

    except getopt.GetoptError:
        # Print debug info
        print('Invalid arguments.')
        sys.exit(2)

    for option, argument in opts:
        if option in ('-c', '--configfile'):
            configfile = argument
        elif option in ('-a', '--algorithm'):
            algorithm = argument

    parameters = read_config_file(configfile)

    if configfile is None or algorithm is None:
        print('Invalid arguments.')
    else:
        run(algorithm, parameters)


if __name__ == '__main__':
    main(sys.argv[1:])
