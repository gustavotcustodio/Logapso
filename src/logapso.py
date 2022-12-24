import random
import copy
import numpy as np
from src.pso import Pso
from src.genetic_algorithm_logapso import GeneticAlgorithmLogapso
from src.functions import FitnessFunction
import src.checkpointmanager as chkpoint
import src.functions as functions


class Logapso(Pso):

    def __init__(self, swarm_size, inertia, acc1, acc2, maxiters, step_val,
                 fitnessfunction, ga, output_file):

        super().__init__(swarm_size, inertia, acc1, acc2, maxiters,
                         fitnessfunction, output_file)
        self.step_val = step_val
        self.ga = ga
        self.updated_global_solution = True

    def apply_ga(self, particle):
        self.ga.particle = particle
        ga_solution = self.ga.run()

        new_position = ga_solution.genes * self.step_val + \
            particle.best_position
        # new_fitness = ga_solution.fitness_value
        new_fitness = ga_solution.current_fitness

        # Update if the solution is improved after running a GA
        if self.fitnessfunction.is_fitness_improved(new_fitness,
                                                    particle.best_fitness):
            particle.position = new_position
            particle.current_fitness = new_fitness
            particle.best_position = new_position
            particle.best_fitness = new_fitness

            under_lbound = np.where(particle.position < particle.lbound)[0]
            over_ubound = np.where(particle.position > particle.ubound)[0]
            particle.position[under_lbound] = particle.lbound
            particle.position[over_ubound] = particle.ubound

    def update_best_particle(self, particle):
        old_fitness_value = self.best_particle.best_fitness
        new_fitness_value = particle.best_fitness

        if self.fitnessfunction.is_fitness_improved(new_fitness_value,
                                                    old_fitness_value):
            self.best_particle = copy.deepcopy(particle)
            self.updated_global_solution = True

    def run(self, checkpoint_file: str):
        for i in range(self.first_iter, self.maxiters):
            for particle in self.particles:
                particle.update_velocity(self.best_particle.position,
                                         self.inertia, self.acc1, self.acc2)
                particle.update_position()
                particle.current_fitness = self.fitnessfunction.calc_fitness(
                    particle.position)

                # Update the best position found by the particle
                if self.fitnessfunction.is_fitness_improved(
                    particle.current_fitness, particle.best_fitness
                ):
                    particle.best_position = np.copy(particle.position)
                    particle.best_fitness = particle.current_fitness

                    if random.random() < 0.1:  # GA is applied 10% of times
                        # Apply a GA to fine-tune the candidate solution
                        self.apply_ga(particle)

                    # Update the global solution if the found solution is
                    # a better candidate
                    self.update_best_particle(particle)

            if self.updated_global_solution:  # If the global solution updated
                # Applies the GA on global solution
                self.apply_ga(self.best_particle)
                self.updated_global_solution = False

            chkpoint.save_checkpoint(self.particles, self.best_particle,
                                     i, checkpoint_file)
            self.outputstream.write('Iteration %d' % i)
            self.outputstream.write(40 * '=')
            self.outputstream.write('Best position: %s' % self.best_particle.best_position)
            self.outputstream.write('Best fitness: %f\n' % self.best_particle.best_fitness)


if __name__ == '__main__':
    swarm_size = 50
    particle_length = 50
    mutation_rate = 0.02
    n_generations = 30
    step_value = 0.2
    inertia = 0.7
    acc1 = 1.4
    acc2 = 1.4
    lbound = -100.0
    ubound = 100.0
    maxiters = 300

    fitnessfunction = functions.get_fitness_function('square_sum')

    ga = GeneticAlgorithmLogapso(
        swarm_size, particle_length, mutation_rate, n_generations,
        fitnessfunction, step_value, operation_type='bitstring',
        possible_genes=[-1, 0, 1]
    )
    pso_optimizer = Logapso(swarm_size, inertia, acc1, acc2, maxiters,
                            step_value, fitnessfunction, ga)
    pso_optimizer.generate_particles(particle_length, lbound, ubound)
    pso_optimizer.run()
