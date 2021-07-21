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


def scale_feature(feature, scaled_min=0, scaled_max=1):
    max_val = max(feature)
    min_val = min(feature)
    return (scaled_max - scaled_min) * (feature - min_val) / \
        (max_val - min_val) + scaled_min


def run_pso(params):
    # Getting parameters from config file
    swarm_size = params['pso']['swarm_size']
    inertia = params['pso']['inertia']
    acc1 = params['pso']['acc1']
    acc2 = params['pso']['acc2']
    maxiters = params['pso']['maxiters']
    lbound = params['pso']['lbound']
    ubound = params['pso']['ubound']

    # Cluster optimization problem
    if 'dataset' in params:
        data = read_dataset(params['dataset'])
        n_features = data[:, :-1].shape[1]  # number of features

        features = np.array([scale_feature(data[:, i], -1, 1)
                             for i in range(n_features)]).T
        labels = data[:, -1]

        fitnessfunction = fitness_function.get_fitness_function(
            params['fitness'], features, labels)
        particle_length = n_features * params['n_clusters']
    else:
        fitnessfunction = fitness_function.get_fitness_function(
            params['fitness'])
        particle_length = params['pso']['particle_length']

    pso = Pso(swarm_size, inertia, acc1, acc2, maxiters, fitnessfunction)
    pso.generate_particles(particle_length, lbound, ubound)
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
