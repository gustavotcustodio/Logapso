from __future__ import absolute_import

import pytest
from src.chromosome import Chromosome
from src.genetic_algorithm_gapso import GeneticAlgorithmGapso

def soma(chromosome: list):
    return sum(chromosome)

def test_crossover():
    population = [[0.5, 0.2, 0.3],
                  [0.8, 0.4, 0.1],
                  [0.6, 0.7, 0.2],
                  [0.9, 0.9, 0.3]]
    gapso = GeneticAlgorithmGapso(3, 1, soma, population)

if __name__ == "__main__":
    test_crossover()
