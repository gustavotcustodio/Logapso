import time
import numpy as np
from sklearn.metrics import silhouette_score
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from particle import Particle
from fitness_function import FitnessFunction


class Pso:
    """Regular PSO implementation.

    Attributes
    ----------
    swarm_size: int
        Number of particles.
    l_bound, ubound: float
        Lower and upper bounds for a position in the array.
    fitnessfunction: FitnessFunction
        Function for evaluating particles.
    inertia: float
        Inertia weight.
    acc1, acc2: float
        Accelerarion coefficients.
    max_iters: int
        Maximum number of iterations.
    """

    def __init__(self, swarm_size, inertia, acc1, acc2, max_iters,
                 fitnessfunction, checkpoint_file=None):
        """
        Parameters
        ----------
        checkpoint_file: string, optional
            File containing a checkpoint to the experiments.
        """
        self.swarm_size = swarm_size
        self.max_iters = max_iters

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

    def _update_best_particle(self, particle):
        """ Check if new particle is a better solution than the current one.

        Parameters
        ----------
        particle: Particle
            Particle to compare with the current best
        """
        old_fitness_value = self.best_particle.best_fitness
        new_fitness_value = particle.best_fitness

        if self.fitnessfunction.is_fitness_improved(
                new_fitness_value, old_fitness_value):
            self.best_particle = particle

    def run(self):
        for _ in range(self.max_iters):
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
                self._update_best_particle(particle)
            print(self.best_particle.best_fitness)


def silhouette(inputs, labels, precomput_dists):
    n_attrs = inputs.shape[1]

    def wrapper(clusters_arr):
        n_clusters = int(clusters_arr.shape[0] / n_attrs)  # number of clusters
        clusters = np.reshape(clusters_arr, (n_clusters, n_attrs))
        data_clusters_dists = distance_matrix(inputs, clusters)
        labels = np.argmin(data_clusters_dists, axis=1)
        if len(np.unique(labels)) < 2:
            return -1
        return silhouette_score(precomput_dists, labels, metric='precomputed')
    return wrapper


def xie_beni(inputs, labels):
    n = inputs.shape[0]  # number of inputs
    m = inputs.shape[1]  # number of attributes

    def wrapper(particle):
        d = int(particle.shape[0]/m)  # number of clusters
        clusters = np.reshape(particle, (d, m))  # fit each cluster in a row
        distances = distance_matrix(inputs, clusters)
        # 10**(-100) avoids division by 0
        distances = np.where(distances != 0, distances, 10**(-100))
        # Shape of distance matrix:(n x d)
        u = distances**2 / np.sum(distances**2, axis=1)[:, np.newaxis]
        u = 1.0 / u
        u = (u.T / np.sum(u, axis=1)).T
        num = np.sum(u**2 * distances**2)
        den = n * min(distance.pdist(clusters))**2
        return num / den
    return wrapper


def read_dataset(dataset):
    data = np.genfromtxt(dataset, delimiter=',')
    np.random.shuffle(data)
    return data[:, :-1], data[:, -1]


if __name__ == "__main__":
    X, y = read_dataset('wdbc.data')
    particle_length = X.shape[1] * len(np.unique(y)) * 3

    distances = distance_matrix(X, X)
    sf = silhouette(X, y, distances)
    xb = xie_beni(X, y)

    previous = time.time()
    print("XB")
    fitnessfunction = FitnessFunction(xb, maximization=False)
    pso_optimizer = Pso(50, 0.1, 0.2, 0.2, 300, fitnessfunction)
    pso_optimizer.generate_particles(particle_length, lbound=-1, ubound=1)
    pso_optimizer.run()
    print("Tempo: " + str(time.time() - previous))

    previous = time.time()
    print("Silhouette")
    fitnessfunction = FitnessFunction(sf, maximization=True)
    pso_optimizer = Pso(50, 0.1, 0.2, 0.2, 300, fitnessfunction)
    pso_optimizer.generate_particles(particle_length, lbound=-1, ubound=1)
    pso_optimizer.run()
    print("Tempo: " + str(time.time() - previous))
