import math
import numpy as np
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score


class FitnessFunction:
    def __init__(self, function, maximization):
        self.function = function
        self.maximization = maximization

    def calc_fitness(self, solution):
        return self.function(solution)

    def is_fitness_improved(self, new_fitness, old_fitness):
        if self.maximization and new_fitness > old_fitness:
            return True
        elif not self.maximization and new_fitness < old_fitness:
            return True
        else:
            return False

    def get_index_best(self, value_list):
        if self.maximization:
            return value_list.index(max(value_list))
        else:
            return value_list.index(min(value_list))

    def update_function(self, function):
        self.function = function


def silhouette(inputs, precomput_dists):
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
        # Shape of distance matrix:(n x d)
        u = distances**2 / np.sum(distances**2, axis=1)[:, np.newaxis]
        u = 1.0 / u
        u = (u.T / np.sum(u, axis=1)).T
        num = np.sum(u**2 * distances**2)
        # print(clusters)
        den = n * min(distance.pdist(clusters))**2
        # print(den)
        return num / den
    return wrapper


def square_sum(x):
    # -100 to 100
    return sum(np.square(x))


def griewank(x):
    # -600 to 600
    term1 = 1/4000 * sum(np.square(x - 100))
    term2 = np.prod(np.cos((x - 100) / np.sqrt(range(1, len(x)+1))))
    return term1 - term2 + 1


def ackley(x):
    # -32 to 32
    n = len(x)
    term1 = -20 * math.exp(-0.2 * np.sqrt(sum(x**2)/n))
    term2 = -math.exp(sum(np.cos(2*math.pi*x))/n)
    return term1 + term2 + 20 + math.e


def schwefel_226(x):
    # -500 to 500
    absx = np.abs(x)
    const = 418.982887272433799807913601398
    return const*len(x) - sum(x*np.sin(np.sqrt(absx)))


def get_fitness_function(function_name, data=None):
    if data is not None:
        features, labels = data[:, :-1], data[:, -1]

        if function_name == 'silhouette':
            precomput_dists = distance_matrix(features, features)
            function = silhouette(features, precomput_dists)
            return FitnessFunction(function, maximization=True)

        elif function_name == 'xie_beni':
            function = xie_beni(features, labels)
            return FitnessFunction(function, maximization=False)

    elif function_name == 'griewank':
        return FitnessFunction(griewank, maximization=False)

    elif function_name == 'square_sum':
        return FitnessFunction(square_sum, maximization=False)
