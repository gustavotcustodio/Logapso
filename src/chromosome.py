import random
import numpy as np


class Chromosome:
    def __init__(self, chrom_length, possible_genes=[0, 1]):
        self.chrom_length = chrom_length
        self.possible_genes = possible_genes
        self.genes = self._generate_chromosome()

    def _generate_chromosome(self):
        return np.array([random.choice(self.possible_genes)
                         for _ in range(self.chrom_length)])

    def _single_point_crossover(self, chromosome):
        crossover_point = random.randint(1, self.chrom_length-1)

        tmp = self.genes[:crossover_point]
        self.genes[:crossover_point] = chromosome.genes[:crossover_point]
        chromosome.genes[:crossover_point] = tmp

    def _uniform_crossover(self, chromosome):
        r = random.random()
        self.genes = r * self.genes + (1 - r) * chromosome.genes

    def crossover(self, chromosome, operation_type='bitstring'):
        if self.chrom_length > 1:
            if operation_type == 'bitstring':
                self._single_point_crossover(chromosome)
            else:
                self._uniform_crossover(chromosome)

    def _normal_mutation(self, genes):
        # random normal distribution number
        r = random.gauss(mu=0, sigma=1)
        pos = random.randint(0, len(genes) - 1)
        genes[pos] = genes[pos] + r
        return genes

    def _bit_string_mutation(self, genes):
        for val_to_change in self.possible_genes:
            new_values = self.possible_genes.copy()
            new_values.remove(val_to_change)

            indexes_to_change = np.where(genes == val_to_change)[0]
            genes[indexes_to_change] = np.random.choice(
                new_values, size=len(indexes_to_change))
        return genes

    def mutate(self, indexes, operation_type='bitstring'):
        if indexes.shape[0] > 0:
            if operation_type == 'bitstring':
                genes = self._bit_string_mutation(self.genes[indexes])
                self.genes[indexes] = genes
            else:
                self.genes = self._normal_mutation(self.genes[:])



if __name__ == '__main__':
    possible_genes = [0, 1]
    genes = np.array([0, 1, 1, 0, 0, 1, 0, 0, 1, 1])
    indexes = [0, 2, 4, 6, 8]
    mutated_genes = genes[indexes]
    print(genes)

    for val_to_change in possible_genes:
        new_values = possible_genes.copy()
        new_values.remove(val_to_change)

        indexes_to_change = np.where(genes[indexes] == val_to_change)[0]
        mutated_genes[indexes_to_change] = np.random.choice(
            new_values, size=len(indexes_to_change))
    genes[indexes] = mutated_genes
    print(genes)

    x = [1, 3, 5, 6, 7, 9]
    print(x.pop(0))
    print(x)
