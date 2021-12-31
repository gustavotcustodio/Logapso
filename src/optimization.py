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
from src.logapso import Logapso
from src.genetic_algorithm_logapso import GeneticAlgorithmLogapso


N_CPUS = 2
CHECKPOINT_DIR = 'checkpoints'


def create_logapso_optimizer(individual_length: int, params: dict,
                             fitnessfunction: FitnessFunction) -> Logapso:
    step_value = 1
    possible_genes = [-1, 0, 1]

    ga = GeneticAlgorithmLogapso(
        params['ga']['pop_size'], individual_length,
        params['ga']['mutation_rate'], params['ga']['n_generations'],
        fitnessfunction, step_value, possible_genes)
    logapso = Logapso(
        params['pso']['swarm_size'], params['pso']['inertia'],
        params['pso']['acc1'], params['pso']['acc2'],
        params['pso']['maxiters'], step_value, fitnessfunction, ga)
    return logapso


def create_pso_optimizer(params: dict, fitnessfunc: FitnessFunction) -> Pso:
    pso = Pso(
        params['pso']['swarm_size'], params['pso']['inertia'],
        params['pso']['acc1'], params['pso']['acc2'],
        params['pso']['maxiters'], fitnessfunc)
    return pso


def run_optimizer(optimizer: Pso, individual_length: int, params: dict,
                  checkpoint_file: str, start_from_checkpoint=False):

    optimizer.generate_particles(
        individual_length, params['pso']['lbound'], params['pso']['ubound']
    )
    if start_from_checkpoint:
        chkpoint.load_checkpoint(optimizer, checkpoint_file)

    optimizer.run(checkpoint_file)


def run_experiment(algorithm: str, paramsfile: str,
                   start_from_checkpoint: bool):
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

    if algorithm == 'logapso':
        optimizer = create_logapso_optimizer(length, params, fitnessfunc)

    else:  # optimizer = 'pso'
        optimizer = create_pso_optimizer(params, fitnessfunc)

    checkpoint_file = os.path.join(
        CHECKPOINT_DIR, f'chk_{algorithm}_{paramsfile}'
    )
    run_optimizer(optimizer, length, params, checkpoint_file,
                  start_from_checkpoint)
    # Delete checkpoint after finishing
    chkpoint.remove(checkpoint_file)


def run_algorithms(algorithms: list, paramsfiles: list,
                   start_from_checkpoint: bool):

    with multiprocessing.Pool(N_CPUS) as pool:
        pool.starmap(
            run_experiment,
            [(algorithm, params, start_from_checkpoint)
             for algorithm in algorithms for params in paramsfiles])
        pool.close()
        pool.join()
