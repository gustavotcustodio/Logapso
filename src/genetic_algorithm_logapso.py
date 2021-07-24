import numpy as np
from genetic_algorithm import GeneticAlgorithm


class GeneticAlgorithmLogapso(GeneticAlgorithm):

    def __init__(self, pop_size, chrom_length, mutation_rate, n_generations,
                 fitnessfunction, step_value, possible_genes=[0, 1]):

        super().__init__(pop_size, chrom_length, mutation_rate,
                         n_generations, fitnessfunction, possible_genes)
        self.step_value = step_value

    def calc_population_fitness(self):
        for chromosome in self.population:
            new_position = self.step_value * chromosome.genes + \
                self.particle.best_position

            under_lbound = np.where(new_position < self.particle.lbound)[0]
            over_ubound = np.where(new_position > self.particle.ubound)[0]
            new_position[under_lbound] = self.particle.lbound
            new_position[over_ubound] = self.particle.ubound

            fitness_val = self.fitnessfunction.calc_fitness(new_position)
            chromosome.set_fitness_value(fitness_val)

    def set_particle(self, particle):
        self.particle = particle
