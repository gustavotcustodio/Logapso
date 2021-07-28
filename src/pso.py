import numpy as np
import json
import copy
import functions
from particle import Particle


CHECKPOINT_FILE = 'checkpoint.json'


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

        self.best_particle = self._find_best_particle()

    def load_checkpoint(self):
        """ Continue from a previous stopped experiment.
        """
        with open(CHECKPOINT_FILE, 'r') as json_file:
            data = json.load(json_file)
            lbound = data['lbound']
            ubound = data['ubound']
            particle_length = self.load_ndarray(data['best']['position']
                                                ).shape[0]
            for i in range(self.swarm_size):
                i = str(i)
                particle = Particle(particle_length, lbound, ubound)
                particle.position = self.load_ndarray(data[i]['position'])
                particle.current_fitness = data[i]['fitness']
                particle.best_position = self.load_ndarray(
                                            data[i]['best_position'])
                particle.best_fitness = data[i]['best_fitness']
                particle.velocity = self.load_ndarray(data[i]['velocity'])
                self.particles.append(particle)

            self.best_particle = Particle(particle_length, lbound, ubound)
            self.best_particle.position = self.load_ndarray(
                                            data['best']['position'])
            self.best_particle.current_fitness = data['best']['fitness']
            self.best_particle.best_position = self.load_ndarray(
                                                data['best']['position'])
            self.best_particle.best_fitness = data['best']['fitness']
            self.best_particle.velocity = self.load_ndarray(
                                            data['best']['velocity'])
            self.best_particle.lbound = data['lbound']
            self.best_particle.ubound = data['ubound']
            self.first_iter = data['iteration']

    def load_ndarray(self, data):
        return np.fromstring(data[1:-1].replace('\n', ''), sep=' ')

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
        for i in range(self.first_iter, self.maxiters):
            for particle in self.particles:
                particle.update_velocity(self.best_particle.position,
                                         self.inertia, self.acc1, self.acc2)

                # Update the particles's current position and calculate the new
                # fitness value
                particle.update_position()
                particle.current_fitness = self.fitnessfunction.calc_fitness(
                    particle.position)

                # Update the best position found by the particle
                if self.fitnessfunction.is_fitness_improved(
                    particle.current_fitness, particle.best_fitness
                ):
                    particle.best_position = np.copy(particle.position)
                    particle.best_fitness = particle.current_fitness

                    # Update the global solution if it is a better candidate
                    self.update_best_particle(particle)

            self.save_checkpoint(i)
            print('Iteration %d' % i)
            print(40 * '=')
            print('Best position: %s' % self.best_particle.best_position)
            print('Best fitness: %f\n' % self.best_particle.best_fitness)

    def save_checkpoint(self, current_iteration):
        checkpoint = {}
        for i in range(len(self.particles)):
            checkpoint[i] = {}
            checkpoint[i]['position'] = str(self.particles[i].position)
            checkpoint[i]['fitness'] = self.particles[i].current_fitness
            checkpoint[i]['best_position'
                          ] = str(self.particles[i].best_position)
            checkpoint[i]['best_fitness'] = self.particles[i].best_fitness
            checkpoint[i]['velocity'] = str(self.particles[i].velocity)
        checkpoint['best'] = {}
        checkpoint['best']['position'] = str(self.best_particle.best_position)
        checkpoint['best']['fitness'] = self.best_particle.best_fitness
        checkpoint['best']['velocity'] = str(self.best_particle.velocity)
        checkpoint['lbound'] = self.best_particle.lbound
        checkpoint['ubound'] = self.best_particle.ubound
        checkpoint['iteration'] = current_iteration + 1

        self.save_json(CHECKPOINT_FILE, checkpoint)

    def save_json(self, filename, data):
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)


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
