import os
import json
import numpy as np
from src.particle import Particle


def load_ndarray(data: 'np.ndarray'):
    return np.fromstring(data[1:-1].replace('\n', ''), sep=' ')


def save_json(filename: str, data: dict):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def load_checkpoint(optimizer: 'Pso', checkpoint_file: str):
    """ Continue from a previous stopped experiment.
    """
    # Check if file exists before
    if not os.path.isfile(checkpoint_file):
        return

    # If file exists, load its content
    with open(checkpoint_file, 'r') as json_file:
        data = json.load(json_file)
        lbound = data['lbound']
        ubound = data['ubound']
        length = load_ndarray(data['best']['position']).shape[0]

        for p in range(optimizer.swarm_size):
            p = str(p)
            particle = Particle(length, lbound, ubound)
            particle.position = load_ndarray(data[p]['position'])
            particle.current_fitness = data[p]['fitness']
            particle.best_position = load_ndarray(data[p]['best_position'])
            particle.best_fitness = data[p]['best_fitness']
            particle.velocity = load_ndarray(data[p]['velocity'])
            optimizer.particles.append(particle)

        optimizer.best_particle = Particle(length, lbound, ubound)
        optimizer.best_particle.position = load_ndarray(
            data['best']['position']
        )
        optimizer.best_particle.current_fitness = data['best']['fitness']
        optimizer.best_particle.best_position = load_ndarray(
            data['best']['position']
        )
        optimizer.best_particle.best_fitness = data['best']['fitness']
        optimizer.best_particle.velocity = load_ndarray(
            data['best']['velocity']
        )
        optimizer.best_particle.lbound = data['lbound']
        optimizer.best_particle.ubound = data['ubound']
        optimizer.first_iter = data['iteration']


def save_checkpoint(population: list, best_solution: 'Particle',
                    iteration: int, filename: str):
    checkpoint = {}
    for p in range(len(population)):
        checkpoint[p] = {}
        checkpoint[p]['position'] = str(population[p].position)
        checkpoint[p]['fitness'] = population[p].current_fitness
        checkpoint[p]['best_position'
                      ] = str(population[p].best_position)
        checkpoint[p]['best_fitness'] = population[p].best_fitness
        checkpoint[p]['velocity'] = str(population[p].velocity)

    checkpoint['best'] = {}
    checkpoint['best']['position'] = str(best_solution.best_position)
    checkpoint['best']['fitness'] = best_solution.best_fitness
    checkpoint['best']['velocity'] = str(best_solution.velocity)
    checkpoint['lbound'] = best_solution.lbound
    checkpoint['ubound'] = best_solution.ubound
    checkpoint['iteration'] = iteration + 1

    save_json(filename, checkpoint)

def remove(checkpoint_file: str):
    """ Remove checkpoint file.
    """
    if os.path.isfile(checkpoint_file):
        os.remove(checkpoint_file)
