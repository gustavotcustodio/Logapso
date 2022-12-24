from copy import deepcopy
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

    def _initialize_ga(self, selected_pop: list[Individual], chrom_length: int,
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
            self._update_best_position_from_particle(individual)

    def _update_best_position_from_particle(self, individual):
        if self.fitnessfunction.is_fitness_improved(
            individual.current_fitness, individual.best_fitness
        ):
            individual.best_position = np.copy(individual.position)
            individual.best_fitness = individual.current_fitness
            # Update the global solution if it is a better candidate
            self.update_best_particle(individual)

    def _update_individuals_if_improved(self, candidates, old_individuals):
        """
        Check if the fitness was improved after running the genetic algorithm.
        If it was, replace the individual by the improved one.
        """
        for i in range(len(candidates)):
            # under_lbound = np.where(candidates[i].position < candidates[i].lbound)[0]
            # over_ubound = np.where(candidates[i].position > candidates[i].ubound)[0]

            # candidates[i].position[under_lbound] = candidates[i].lbound
            # candidates[i].position[over_ubound] = candidates[i].ubound

            candidates[i].velocity = candidates[i].position - old_individuals[i].position

            # candidates[i].velocity[under_lbound] = 0.
            # candidates[i].velocity[over_ubound] = 0.

            self._update_best_position_from_particle(candidates[i])
                # old_individuals[i] = candidates[i]
                # self._update_best_position_from_particle(old_individuals[i])
                # individual.velocity = individual.position - individual.old_position
        # individuals = candidates + old_individuals
        # if self.fitnessfunction.maximization:
        #     sorted(individuals, key = lambda x: x.best_fitness)
        # else:
        #     sorted(individuals, reverse=True, key = lambda x: x.current_fitness)
        # return individuals[:len(candidates)]

    def run(self, checkpoint_file: str):
        n_to_select = int(self.swarm_size / 2)
        length = self.particles[0].particle_length

        for i in range(self.first_iter, self.maxiters):

            self._sort_individuals_by_fitness()
            best_individuals = [self.particles[j]
                                for j in range(n_to_select)]
            worst_individuals = [self.particles[j]
                                 for j in range(n_to_select, self.swarm_size)]
            parents = [deepcopy(individual) for individual in best_individuals]


            for individual in best_individuals:
                individual.genes = individual.position
            # # print("Antes:",best_individuals[0].position)
            ga = self._initialize_ga(best_individuals, length, MUTATION_RATE)
            ga.run()

            for individual in best_individuals:
                individual.position = individual.genes
            # print("Depois:",best_individuals[0].position)

            self._update_individuals_if_improved(best_individuals, parents)

            # print("Antes",[ind.best_fitness for ind in worst_individuals])
            self._pso_step(self.particles)
            # print("Depois:", [ind.best_fitness for ind in worst_individuals])
            # print(100 * "=")
            # individuals_next_gen = best_individuals + worst_individuals + parents

            # if self.fitnessfunction.maximization:
            #     sorted(individuals_next_gen, reverse=True, key = lambda x: x.best_fitness)
            # else:
            #     sorted(individuals_next_gen, key = lambda x: x.best_fitness)
            # self.particles = individuals_next_gen[:self.swarm_size]

            chkpoint.save_checkpoint(self.particles, self.best_particle,
                                     i, checkpoint_file)
            self.outputstream.write('Iteration %d' % i)
            self.outputstream.write(40 * '=')
            self.outputstream.write('Best position: %s' % self.best_particle.best_position)
            self.outputstream.write('Best fitness: %f\n' % self.best_particle.best_fitness)
