import random
import copy
import numpy as np
from pso import Pso
import functions
from functions import FitnessFunction
from genetic_algorithm_logapso import GeneticAlgorithmLogapso


class Logapso(Pso):

    def __init__(self, swarm_size, inertia, acc1, acc2, maxiters, step_val,
                 fitnessfunction, ga, checkpoint_file=None):

        super().__init__(swarm_size, inertia, acc1, acc2, maxiters,
                         fitnessfunction, checkpoint_file)
        self.step_val = step_val
        self.ga = ga
        self.updated_global_solution = True

    def apply_ga(self, particle):
        self.ga.set_particle(particle)
        ga_solution = self.ga.run()

        new_position = ga_solution.genes * self.step_val + \
            particle.best_position
        new_fitness = ga_solution.fitness_value

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

    def run(self):
        for _ in range(self.maxiters):
            for particle in self.particles:
                particle.update_velocity(self.best_particle.position,
                                         self.inertia, self.acc1, self.acc2)
                particle.update_position()
                particle.set_current_fitness(
                    self.fitnessfunction.calc_fitness(particle.position))

                # Update the best position found by the particle
                if self.fitnessfunction.is_fitness_improved(
                    particle.current_fitness, particle.best_fitness
                ):
                    particle.update_best_position()
                    particle.set_best_fitness(particle.current_fitness)

                    if random.random() < 0.1:  # GA is applied 10% of times
                        # Apply a GA to fine-tune the candidate solution
                        self.apply_ga(particle)

                    # Update the global solution if the found solution is
                    # a better candidate
                    self.update_best_particle(particle)

            if self.updated_global_solution:  # If the global solution updated
                # Applies the GA on global solution
                # print('========================================')
                # print(self.best_particle.best_fitness)
                self.apply_ga(self.best_particle)
                # self.updated_global_solution = False

                print(self.best_particle.best_fitness)
                print('========================================')


if __name__ == '__main__':
    swarm_size = 50
    particle_length = 15
    mutation_rate = 0.02
    n_generations = 30
    step_value = 0.2
    inertia = 0.7
    acc1 = 1.4
    acc2 = 1.4
    maxiters = 300

    function = functions.griewank
    fitnessfunction = FitnessFunction(function, maximization=False)

    ga = GeneticAlgorithmLogapso(
        swarm_size, particle_length, mutation_rate, n_generations,
        fitnessfunction, step_value, possible_genes=[-1, 0, 1]
    )
    pso_optimizer = Logapso(swarm_size, inertia, acc1, acc2, maxiters,
                            step_value, fitnessfunction, ga)
    pso_optimizer.generate_particles(particle_length, lbound=-1, ubound=1)
    pso_optimizer.run()
