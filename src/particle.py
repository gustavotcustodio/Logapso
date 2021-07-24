import numpy as np


class Particle:

    def __init__(self, particle_length, lbound, ubound):
        self.particle_length = particle_length
        self.lbound = lbound
        self.ubound = ubound
        self.position = self._generate_random_array()
        self.best_position = np.copy(self.position)
        self.velocity = self._generate_random_array()

    def _generate_random_array(self):
        """Generates the particle starting positions and velocities
        ranging from lbound to ubound

        Returns
        -------
        ndarray
        """
        return np.random.uniform(self.lbound, self.ubound,
                                 size=self.particle_length)

    def update_velocity(self, global_best_position, inertia, acc1, acc2):
        # Matrices of random numbers
        r1 = np.random.uniform(0, 1, self.particle_length)
        r2 = np.random.uniform(0, 1, self.particle_length)

        self.velocity = inertia * self.velocity + \
            acc1 * r1 * (self.best_position - self.position) + \
            acc2 * r2 * (global_best_position - self.position)

        # under_lbound = np.where(self.velocity < self.lbound)[0]
        # over_ubound = np.where(self.velocity > self.ubound)[0]
        # self.velocity[under_lbound] = self.lbound
        # self.velocity[over_ubound] = self.ubound

    def update_position(self):
        self.position += self.velocity
        under_lbound = np.where(self.position < self.lbound)[0]
        over_ubound = np.where(self.position > self.ubound)[0]
        self.position[under_lbound] = self.lbound
        self.position[over_ubound] = self.ubound
        self.velocity[under_lbound] = 0.0
        self.velocity[over_ubound] = 0.0

    def update_best_position(self):
        self.best_position = np.copy(self.position)

    def set_current_fitness(self, current_fitness):
        self.current_fitness = current_fitness

    def set_best_fitness(self, best_fitness):
        self.best_fitness = best_fitness
