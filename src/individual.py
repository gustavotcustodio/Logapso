from chromosome import Chromosome
from particle import Particle
import numpy as np

class Individual(Chromosome, Particle):
    def __init__(self, chrom_length, lbound, ubound):
        self.lbound = lbound
        self.ubound = ubound

        self.particle_length = chrom_length
        self.position = self.generate_random_array()
        self.best_position = np.copy(self.position)
        self.velocity = self.generate_random_array()

        self.chrom_length = chrom_length
        self.genes = self.position

    def __str__(self):
        return str(self.genes)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    i = Individual(10, -1, 1)
    print(i)
    i.update_position()
    print(i)
