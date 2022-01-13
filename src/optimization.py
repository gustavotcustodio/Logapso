import os
import sys
import yaml
import numpy as np
import getopt
import multiprocessing
import src.checkpointmanager as chkpoint
import src.datasetreader as reader
import src.functions as functions
from src.functions import FitnessFunction
from src.pso import Pso
from src.gapso import GaPso
from src.logapso import Logapso
from src.genetic_algorithm_logapso import GeneticAlgorithmLogapso


N_CPUS = 4
CHECKPOINT_DIR = 'checkpoints'


def create_logapso_optimizer(individual_length: int, params: dict, fitnessfunc: FitnessFunction,
                             output_file: str) -> Logapso:
    step_value = 1
    possible_genes = [-1, 0, 1]

    ga = GeneticAlgorithmLogapso(
        params['ga']['pop_size'], individual_length,
        params['ga']['mutation_rate'], params['ga']['n_generations'],
        fitnessfunc, step_value, possible_genes)
    logapso = Logapso(
        params['pso']['swarm_size'], params['pso']['inertia'],
        params['pso']['acc1'], params['pso']['acc2'],
        params['pso']['maxiters'], step_value, fitnessfunc, ga, output_file)
    return logapso


def create_pso_optimizer(params: dict, fitnessfunc: FitnessFunction, output_file: str) -> Pso:
    pso = Pso(
        params['pso']['swarm_size'], params['pso']['inertia'],
        params['pso']['acc1'], params['pso']['acc2'],
        params['pso']['maxiters'], fitnessfunc, output_file)
    return pso

def create_gapso_optimizer(params: dict, fitnessfunc: FitnessFunction, output_file: str) -> GaPso:
    gapso = GaPso(
        params['pso']['swarm_size'], params['pso']['inertia'],
        params['pso']['acc1'], params['pso']['acc2'],
        params['pso']['maxiters'], fitnessfunc, output_file)
    return gapso


def run_optimizer(optimizer: Pso, individual_length: int, params: dict,
                  checkpoint_file: str, start_from_checkpoint=False):

    optimizer.generate_particles(
        individual_length, params['pso']['lbound'], params['pso']['ubound']
    )
    if start_from_checkpoint:
        chkpoint.load_checkpoint(optimizer, checkpoint_file)

    optimizer.run(checkpoint_file)


def run_experiment(algorithm: str, paramsfile: str,
                   start_from_checkpoint: bool, current_run: int):
    # Load experiment parameters
    params = reader.read_param_file(paramsfile)

    # Cluster optimization problem or benchmark function
    if 'dataset' in params:
        data = reader.read_dataset(params['dataset'])
        n_features = data[:, :-1].shape[1]  # number of features

        # Fitness function
        fitnessfunc = functions.get_fitness_function(params['fitness'], data)

        # Length of individuals in the optimization algorithm
        length = n_features * params['n_clusters']
    else:
        fitnessfunc = functions.get_fitness_function(params['fitness'])
        length = params['pso']['particle_length']

    # File where the experiments output are saved.
    output_file = f'{algorithm}_{paramsfile.replace(".yml", "")}_run{current_run}'
    if algorithm == 'logapso':
        optimizer = create_logapso_optimizer(length, params, fitnessfunc, output_file)

    elif algorithm == 'gapso':
        optimizer = create_gapso_optimizer(params, fitnessfunc, output_file)

    else:  # optimizer = 'pso'
        optimizer = create_pso_optimizer(params, fitnessfunc, output_file)

    checkpoint_file = os.path.join(
        CHECKPOINT_DIR, f'chk_{algorithm}_{paramsfile}'
    )
    run_optimizer(optimizer, length, params, checkpoint_file,
                  start_from_checkpoint)
    # Delete checkpoint after finishing
    chkpoint.remove(checkpoint_file)


def run_algorithms(algorithms: list, paramsfiles: list,
                   start_from_checkpoint: bool, nruns):
    '''
    with multiprocessing.Pool(N_CPUS) as pool:
        pool.starmap(
            run_experiment,
            [(algorithm, params, start_from_checkpoint, r)
             for params in paramsfiles
             for algorithm in algorithms
             for r in range(1, nruns+1)]
        )
        pool.close()
        pool.join()
        '''
    algorithm = list(algorithms)[0]
    params = paramsfiles[0]
    for r in range(1, nruns+1):
        run_experiment(algorithm, params, start_from_checkpoint, r)
