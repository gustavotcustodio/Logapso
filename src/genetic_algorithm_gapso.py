import random
import numpy as np
import src.functions as functions
from src.genetic_algorithm import GeneticAlgorithm
from src.individual import Individual


class GeneticAlgorithmGapso(GeneticAlgorithm):

    def __init__(self, chrom_length, mutation_rate, fitnessfunction,
                 population, operation_type='arithmetic'):

        self.population = population
        pop_size = len(self.population)
        super().__init__(pop_size, chrom_length, mutation_rate,
                         1, fitnessfunction, operation_type)

    def set_population(population: list[Individual]):
        self.population = population
        self.pop_size = len(population)

    def perform_crossover(self):
        """
        Perform crossover operations between multiple individuals.
        """
        for i in range(0, len(self.population)-1):
            self.population[i].crossover(
                self.population[i+1], self.operation_type
            )
        self.population[-1].crossover(
            self.population[0], self.operation_type
        )

    def perform_mutation(self):
        """
        Randomly changes the value of individuals.
        """
        for individual in self.population:
            if random.random() < self.mutation_rate:
                individual.mutate(individual.genes, self.operation_type)

    def run(self):
        self.perform_crossover()
        self.perform_mutation()
        self.calc_population_fitness()


def generate_individuals(pop_size: int, length: int, fitnessfunction,
                         lbound: float, ubound: float):
    individuals = []

    for _ in range(pop_size):
        individual = Individual(length, lbound, ubound)
        fitness_val = fitnessfunction.calc_fitness(individual.genes)

        individual.current_fitness = fitness_val
        individuals.append(individual)

    return individuals


def main():
    pop_size = 100
    length = 50
    mutation_rate = 0.2
    fitnessfunction = functions.get_fitness_function('square_sum')
    lbound = -1.0
    ubound =  1.0

    individuals = generate_individuals(pop_size, length,
                                       fitnessfunction, lbound, ubound)
    gapso = GeneticAlgorithmGapso(length, mutation_rate, fitnessfunction,
                                  individuals)
    gapso.run()


if __name__ == '__main__':
    main()
