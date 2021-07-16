import numpy as np


class Particle:

    def __init__(self, particle_length, lbound, ubound):
        self.particle_length = particle_length
        self.lbound = lbound
        self.ubound = ubound
        self.position = self._generate_random_position()
        self.best_position = np.copy(self.position)
        self.velocity = self._generate_random_velocity()

    def _generate_random_position(self):
        """Generates the particle starting positions ranging from
        lbound to ubound

        Returns
        -------
        ndarray
        """
        return np.random.uniform(self.lbound, self.ubound,
                                 size=self.particle_length)

    def _generate_random_velocity(self):
        """Generates the initial velocity for the particle.

        Returns
        -------
        ndarray
        """
        velocity_limit = abs(self.ubound - self.lbound)
        return np.random.uniform(-velocity_limit, velocity_limit,
                                 size=self.particle_length)

    def update_velocity(self, global_best_position, inertia, acc1, acc2):
        # Matrices of random numbers
        r1 = np.random.uniform(0, 1, self.particle_length)
        r2 = np.random.uniform(0, 1, self.particle_length)

        self.velocity += inertia * self.velocity + \
            acc1 * r1 * (self.best_position - self.position) + \
            acc2 * r2 * (global_best_position - self.position)

    def update_position(self):
        self.position += self.velocity
        # fitness_value = self.fitnessfunction.eval_array(self.position)
        # self.set_current_fitness(fitness_value)

    def update_best_position(self):
        self.best_position = np.copy(self.position)

    def set_current_fitness(self, current_fitness):
        self.current_fitness = current_fitness

    def set_best_fitness(self, best_fitness):
        self.best_fitness = best_fitness
