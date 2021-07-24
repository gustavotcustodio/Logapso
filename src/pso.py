import functions
import copy
from particle import Particle


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
                 fitnessfunction, checkpoint_file=None):
        """
        Parameters
        ----------
        checkpoint_file: string, optional
            File containing a checkpoint to the experiments.
        """
        self.swarm_size = swarm_size
        self.maxiters = maxiters

        self.inertia = inertia
        self.acc1 = acc1
        self.acc2 = acc2

        # Fitness function
        self.fitnessfunction = fitnessfunction
        self.checkpoint_file = checkpoint_file

        self.particles = []

    def generate_particles(self, particle_length, lbound, ubound):
        for _ in range(self.swarm_size):
            particle = Particle(particle_length, lbound, ubound)
            fitness_val = self.fitnessfunction.calc_fitness(particle.position)

            particle.set_current_fitness(fitness_val)
            particle.set_best_fitness(fitness_val)
            self.particles.append(particle)

        self.best_particle = self._find_best_particle()

    def _find_best_particle(self):
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

    def run(self):
        for _ in range(self.maxiters):
            for particle in self.particles:
                particle.update_velocity(self.best_particle.position,
                                         self.inertia, self.acc1, self.acc2)

                # Update the particles's current position and calculate the new
                # fitness value
                particle.update_position()
                particle.set_current_fitness(
                    self.fitnessfunction.calc_fitness(particle.position))

                # Update the best position found by the particle
                if self.fitnessfunction.is_fitness_improved(
                    particle.current_fitness, particle.best_fitness
                ):
                    particle.update_best_position()
                    particle.set_best_fitness(particle.current_fitness)

                    # Update the global solution if it is a better candidate
                    self.update_best_particle(particle)
            print(self.best_particle.best_fitness)
            print('============================================')


if __name__ == '__main__':
    swarm_size = 500
    particle_length = 100
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
