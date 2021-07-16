import random
import numpy as np


class Chromosome:
    def __init__(self, chrom_length, possible_values=[0, 1], genes=None):
        self.chrom_length = chrom_length
        self.possible_values = possible_values
        if genes is None:
            self.genes = self._generate_chromosome()
        else:
            self.genes = genes

    def _generate_chromosome(self):
        return np.array([random.choice(self.possible_values)
                         for _ in range(self.chrom_length)])

    def do_single_point_crossover(self, chromosome):
        if self.chrom_length > 1:
            crossover_point = random.randint(1, self.chrom_length-1)

            tmp = self.genes[:crossover_point]
            self.genes[:crossover_point] = chromosome.genes[:crossover_point]
            chromosome.genes[:crossover_point] = tmp

    def mutate_chromosome(self, indexes):
        if indexes.size > 0:
            mutated_genes = self.genes[indexes]

            for val_to_change in self.possible_values:
                new_values = self.possible_values.copy()
                new_values.remove(val_to_change)

                indexes_to_change = np.where(
                    self.genes[indexes] == val_to_change)[0]
                mutated_genes[indexes_to_change] = np.random.choice(
                    new_values, size=len(indexes_to_change))

            self.genes[indexes] = mutated_genes

    def set_fitness_value(self, fitness_value):
        self.fitness_value = fitness_value


if __name__ == '__main__':
    possible_values = [0, 1]
    genes = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])
    indexes = [0, 2, 4, 6, 8]
    mutated_genes = genes[indexes]
    print(genes)

    for val_to_change in possible_values:
        new_values = possible_values.copy()
        new_values.remove(val_to_change)

        indexes_to_change = np.where(genes[indexes] == val_to_change)[0]
        mutated_genes[indexes_to_change] = np.random.choice(
            new_values, size=len(indexes_to_change))
    genes[indexes] = mutated_genes
    print(genes)

    x = [1, 3, 5, 6, 7, 9]
    print(x.pop(0))
    print(x)
