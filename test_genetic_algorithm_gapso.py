from __future__ import absolute_import

import numpy as np
from src.chromosome import Chromosome
from src.genetic_algorithm_gapso import GeneticAlgorithmGapso
from src.functions import FitnessFunction

def soma(chromosome: list):
    return sum(chromosome)

def test_crossover():
    population = []
    real_solution = np.array([[0.71, 0.34, 0.16 ],
                              [0.66, 0.61,  0.17 ],
                              [0.81, 0.84,  0.27 ],
                              [0.767, 0.508, 0.202]])
    for _ in range(4):
        population.append(Chromosome(3))

    population[0].genes = np.array([0.5, 0.2, 0.3])
    population[1].genes = np.array([0.8, 0.4, 0.1])
    population[2].genes = np.array([0.6, 0.7, 0.2])
    population[3].genes = np.array([0.9, 0.9, 0.3])

    funcao = FitnessFunction(soma, True)

    gapso = GeneticAlgorithmGapso(3, 1, funcao, population)
    gapso.perform_crossover()

    solution = np.vstack([chromosome.genes for chromosome in population])
    np.testing.assert_array_almost_equal(solution, real_solution)

def test_run():
    population = []
    real_solution = [1.51, 1.74, 2.22, 1.777]
    for _ in range(4):
        population.append(Chromosome(3))

    population[0].genes = np.array([0.5, 0.2, 0.3])
    population[1].genes = np.array([0.8, 0.4, 0.1])
    population[2].genes = np.array([0.6, 0.7, 0.2])
    population[3].genes = np.array([0.9, 0.9, 0.3])

    funcao = FitnessFunction(soma, True)

    gapso = GeneticAlgorithmGapso(3, 1, funcao, population)
    gapso.run()

    solution = [ind.fitness_value for ind in gapso.population]

    np.testing.assert_array_almost_equal(solution, real_solution)


if __name__ == "__main__":
    test_crossover()
    test_run()
