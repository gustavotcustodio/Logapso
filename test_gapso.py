import random
import numpy as np
from src.chromosome import Chromosome


def test_crossover():
    chrom_length = 3

    chromosome = Chromosome(chrom_length)
    chromosome.genes = np.array([[1.,3,4], [5,1,0], [9,2,1]])

    other_chromosome = Chromosome(chrom_length)
    other_chromosome.genes = np.array([[8.,2,1], [0,1,0], [7,1,0]])

    random.seed(2)

    desired_single_point = np.array([[8.,2,1], [5,1,0], [9,2,1]])
    chromosome.crossover(other_chromosome, operation_type="bitstring")
    np.testing.assert_equal(chromosome.genes, desired_single_point)

    desired_uniform = np.array([[1.64109351, 2.90841521, 3.72524564],
                                [0.45792394, 1.        , 0.        ],
                                [7.18316957, 1.09158479, 0.09158479]])
    chromosome.crossover(other_chromosome, operation_type="uniform")
    np.testing.assert_almost_equal(chromosome.genes, desired_uniform)


def test_mutate():
    correct = np.array([[1., 3., 3.04628299], [5., 1., 0.], [9., 2., 0.04628299]])

    chromosome = Chromosome(3)
    chromosome.genes = np.array([[1.,3,4], [5,1,0], [9,2,1]])

    random.seed(10)
    indexes_to_change = [0, 2]

    # r = random.gauss(mu=0, sigma=1)  # -0.9537170080633371
    # pos = random.randint(0, genes.shape[1]-1)  # 2
    chromosome.mutate(indexes_to_change, operation_type="normal")
    np.testing.assert_almost_equal(chromosome.genes, correct)


if __name__ == "__main__":
    test_mutate()
    test_crossover()
