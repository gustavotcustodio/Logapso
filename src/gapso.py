import numpy as np
from src.pso import Pso
from src.genetic_algorithm_gapso import GeneticAlgorithmGapso
from src.individual import Individual
import src.checkpointmanager as chkpoint
import src.functions as functions

MUTATION_RATE = 0.2

class GaPso(Pso):

    def __init__(self, swarm_size, inertia, acc1, acc2, maxiters,
                 fitnessfunction, output_file):

        super().__init__(swarm_size, inertia, acc1, acc2, maxiters,
                         fitnessfunction, output_file)

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
        if self.fitnessfunction.maximization:
            self.particles = sorted(self.particles,
                                    key=lambda p: p.current_fitness, reverse=True)
        else:
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

    def run(self, checkpoint_file: str):
        n_to_select = int(self.swarm_size / 2)

        for i in range(self.maxiters):
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

            chkpoint.save_checkpoint(self.particles, self.best_particle,
                                     i, checkpoint_file)
            self.outputstream.write('Iteration %d' % i)
            self.outputstream.write(40 * '=')
            self.outputstream.write('Best position: %s' % self.best_particle.best_position)
            self.outputstream.write('Best fitness: %f\n' % self.best_particle.best_fitness)
