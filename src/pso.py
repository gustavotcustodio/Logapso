import numpy as np
import json
import copy
import src.functions as functions
import src.checkpointmanager as chkpoint
from src.particle import Particle


class Pso:
    """Regular PSO implementation.

    Attributes
    ----------
    swarm_size: int
        Number of particles.
    fitnessfunction: FitnessFunction
        Function for evaluating particles.
    inertia: float
        Inertia weight.
    acc1, acc2: float
        Accelerarion coefficients.
    maxiters: int
        Maximum number of iterations.
    """

    def __init__(self, swarm_size, inertia, acc1, acc2, maxiters,
                 fitnessfunction):
        self.swarm_size = swarm_size
        self.maxiters = maxiters

        self.inertia = inertia
        self.acc1 = acc1
        self.acc2 = acc2

        # Fitness function
        self.fitnessfunction = fitnessfunction
        self.particles = []
        self.first_iter = 0

    def generate_particles(self, particle_length, lbound, ubound):

        for _ in range(self.swarm_size):
            particle = Particle(particle_length, lbound, ubound)
            fitness_val = self.fitnessfunction.calc_fitness(particle.position)

            particle.current_fitness = fitness_val
            particle.best_fitness = fitness_val
            self.particles.append(particle)

        self.best_particle = self.find_best_particle()

    def find_best_particle(self):
        """
        Returns
        -------
        Particle
            Particle with best fitness value.
        """
        fitness_values = [particle.best_fitness for particle in self.particles]

        index_best = self.fitnessfunction.get_index_best(fitness_values)
        return self.particles[index_best]

    def update_best_particle(self, particle):
        """ Check if new particle is a better solution than the current one.

        Parameters
        ----------
        particle: Particle
            Particle to compare with the current best
        """
        old_fitness_value = self.best_particle.best_fitness
        new_fitness_value = particle.best_fitness

        if self.fitnessfunction.is_fitness_improved(new_fitness_value,
                                                    old_fitness_value):
            self.best_particle = copy.deepcopy(particle)

    def run(self, checkpoint_file: str):
        for i in range(self.first_iter, self.maxiters):
            for particle in self.particles:
                particle.update_velocity(
                    self.best_particle.position, self.inertia,
                    self.acc1, self.acc2
                )
                # Update the particles's current position and calculate the new
                # fitness value
                particle.update_position()
                particle.current_fitness = \
                    self.fitnessfunction.calc_fitness(particle.position)

                # Update the best position found by the particle
                if self.fitnessfunction.is_fitness_improved(
                    particle.current_fitness, particle.best_fitness
                ):
                    particle.best_position = np.copy(particle.position)
                    particle.best_fitness = particle.current_fitness

                    # Update the global solution if it is a better candidate
                    self.update_best_particle(particle)

            chkpoint.save_checkpoint(self.particles, self.best_particle,
                                     i, checkpoint_file)
            print('Iteration %d' % i)
            print(40 * '=')
            print('Best position: %s' % self.best_particle.best_position)
            print('Best fitness: %f\n' % self.best_particle.best_fitness)


if __name__ == '__main__':
    swarm_size = 500
    particle_length = 50
    inertia = 0.729
    acc1 = 1.49445
    acc2 = 1.49445
    lbound = -100.0
    ubound = 100.0
    maxiters = 1000

    fitnessfunction = functions.get_fitness_function('square_sum')
    pso_optimizer = Pso(swarm_size, inertia, acc1, acc2, maxiters,
                        fitnessfunction)
    pso_optimizer.generate_particles(particle_length, lbound, ubound)
    pso_optimizer.run()
