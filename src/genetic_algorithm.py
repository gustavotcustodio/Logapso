import random
import copy
import numpy as np
from chromosome import Chromosome


class GeneticAlgorithm:

    def __init__(self, pop_size, chrom_length, mutation_rate, n_generations,
                 fitnessfunction, possible_genes=[0, 1]):
        self.pop_size = pop_size
        self.chrom_length = chrom_length
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.fitnessfunction = fitnessfunction
        self.possible_genes = possible_genes

    def _generate_random_population(self):
        """
        Returns
        -------
        list(Chromosome)
            List containing the GA's population.
        """
        return [Chromosome(self.chrom_length, self.possible_genes)
                for _ in range(self.pop_size)]

    def _perform_crossover(self, selected_chromosomes):
        """
        Perform crossover operations between multiple chromosomes.
        """
        offsprings = []

        # Remove the first chromosome from the list, in order to have an even
        # number of chromosomes.
        if len(selected_chromosomes) % 2 == 1:
            offsprings.append(selected_chromosomes[0])
            start_index = 1
        else:
            start_index = 0

        for i in range(start_index, len(selected_chromosomes), 2):
            offsprings.append(copy.deepcopy(selected_chromosomes[i]))
            offsprings.append(copy.deepcopy(selected_chromosomes[i+1]))
            offsprings[i].do_single_point_crossover(offsprings[i+1])

        return selected_chromosomes + offsprings

    def _calc_cumsum(self, fitness_values):
        max_val = max(fitness_values)
        min_val = min(fitness_values)

        if max_val == min_val:
            return np.array(range(1, self.pop_size + 1)) / self.pop_size

        if self.fitnessfunction.maximization:
            fitness_cumsum = np.cumsum(fitness_values - min_val)

        if not self.fitnessfunction.maximization:
            fitness_cumsum = np.cumsum(max_val - fitness_values)
        fitness_cumsum /= max(fitness_cumsum)

        return fitness_cumsum  # \
        # if self.fitnessfunction.maximization else 1 - fitness_cumsum

    def calc_population_fitness(self):
        for chromosome in self.population:
            fitness_val = self.fitnessfunction.calc_fitness(chromosome.genes)
            chromosome.set_fitness_value(fitness_val)

    def _do_roulette_selection(self, n_to_select):
        """
        Roulette selection of genetic algorithm.

        Parameters
        ----------
        n_to_select: int
            Number of candidate solutions to crossover

        Returns
        -------
        selected_chromosomes: list(Chromosome)
            'n_to_select' selected chromosomes for crossover.
        """
        selected_chromosomes = []
        fitness_values = np.array([chromosome.fitness_value
                                   for chromosome in self.population])
        fitness_cumsum = self._calc_cumsum(fitness_values)

        for _ in range(n_to_select):
            # Get the index of the first element equal or lower than a random
            # number from 0 to 1.
            index = np.searchsorted(fitness_cumsum, random.random())
            selected_chromosomes.append(copy.deepcopy(self.population[index]))
        return selected_chromosomes

    def _perform_mutation(self):
        """
        Randomly changes the value of chromosomes.
        """
        mutations = np.random.uniform(0, 1, (self.pop_size, self.chrom_length))

        for chromosome, prob_mutations in zip(self.population, mutations):
            selected_genes = np.where(prob_mutations <= self.mutation_rate)[0]
            chromosome.mutate_chromosome(selected_genes)

    def run(self):
        self.population = self._generate_random_population()
        self.calc_population_fitness()

        for _ in range(self.n_generations):
            n_to_select = int(self.pop_size / 2)
            selected_chromosomes = self._do_roulette_selection(n_to_select)
            self.population = self._perform_crossover(selected_chromosomes)
            self._perform_mutation()
            self.calc_population_fitness()
            fitness_vals = [chromosome.fitness_value
                            for chromosome in self.population]
            index_best = self.fitnessfunction.get_index_best(fitness_vals)
        return self.population[index_best]

    def __str__(self):
        return '\n'.join([str(chromosome.genes)
                          for chromosome in self.population]) + \
            '\n--------------------------------------------------------'
