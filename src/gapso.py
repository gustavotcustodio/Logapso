import numpy as np
from pso import Pso
from genetic_algorithm_gapso import GeneticAlgorithmGapso
from individual import Individual
import functions

MUTATION_RATE = 0.2

class GaPso(Pso):

    """ Hybrid genetic algorithm and particle swarm optimization
    proposed by Kao and Zahara (2008).

    Attributes
    ----------
    swarm_size: int
        Number of particles.
    inertia: float
        Inertia weight.
    acc1, acc2: float
        Accelerarion coefficients.
    maxiters: int
        Maximum number of iterations.
    fitnessfunction: FitnessFunction
        Function for evaluating particles.
    """
    def __init__(self, swarm_size, inertia, acc1, acc2, maxiters,
                 fitnessfunction):

        super().__init__(swarm_size, inertia, acc1, acc2, maxiters,
                         fitnessfunction)

    def generate_particles(self, particle_length: int,
                           lbound: float, ubound: float):
        for _ in range(self.swarm_size):

            individual = Individual(particle_length, lbound, ubound)
            fitness_val = self.fitnessfunction.calc_fitness(
                individual.position)

            individual.current_fitness = fitness_val
            individual.best_fitness = fitness_val
            self.particles.append(individual)

        self.best_particle = self.find_best_particle()

    def _sort_individuals_by_fitness(self):
        self.particles = sorted(self.particles,
                                key=lambda p: p.current_fitness)

    def _create_ga(self, selected_pop: list[Individual], chrom_length: int,
                   mutation_rate: float) -> GeneticAlgorithmGapso:

        n_chromosomes = len(selected_pop)
        ga = GeneticAlgorithmGapso(
            chrom_length, mutation_rate, self.fitnessfunction,
            selected_pop, operation_type='arithmetic'
        )
        return ga

    def _pso_step(self, worst_individuals):

        for individual in worst_individuals:
            individual.update_velocity(self.best_particle.position,
                                       self.inertia, self.acc1, self.acc2)
            # Update the particles's current position and calculate the new
            # fitness value
            individual.update_position()
            individual.current_fitness = \
                self.fitnessfunction.calc_fitness(individual.position)

            # Update the best position found by the particle
            if self.fitnessfunction.is_fitness_improved(
                individual.current_fitness, individual.best_fitness
            ):
                individual.best_position = np.copy(individual.position)
                individual.best_fitness = individual.current_fitness
                # Update the global solution if it is a better candidate
                self.update_best_particle(individual)
            print(self.best_particle.current_fitness)

    def run(self):
        n_to_select = int(self.swarm_size / 2)

        for i in range(100):
            self._sort_individuals_by_fitness()
            best_individuals = [self.particles[i]
                                for i in range(n_to_select)]
            worst_individuals = [self.particles[i]
                                 for i in range(n_to_select, self.swarm_size)]

            length = self.particles[0].particle_length
            ga = self._create_ga(best_individuals, length, MUTATION_RATE)
            ga.run()

            # update best individual
            self.best_particle = self.find_best_particle()

            self._pso_step(worst_individuals)
            self.particles = best_individuals + worst_individuals


def main():
    pop_size = 100
    length = 500
    mutation_rate = 0.2
    max_iters = 10000
    fitnessfunction = functions.get_fitness_function('square_sum')
    lbound = -1.0
    ubound =  1.0

    ga_pso = GaPso(pop_size, 0.1, 0.5, 0.5, max_iters, fitnessfunction)
    ga_pso.generate_particles(length, lbound, ubound)
    ga_pso.run()


if __name__ == '__main__':
    main()
