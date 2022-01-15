import math
import sys
import random
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


def fuku_sugeno(inputs: np.ndarray):
    n = inputs.shape[0]  # number of inputs
    m = inputs.shape[1]  # number of attributes

    def wrapper(particle):
        d = int(particle.shape[0]/m) # number of clusters
        clusters = np.reshape(particle, (d, m))
        intra_clusters_dists = distance.pdist(clusters)
        if 0 in intra_clusters_dists:
            return sys.maxsize  # If clusters are the same

        inputs_clusters_dists = distance_matrix(inputs, clusters)
        inputs_clusters_dists = np.where(inputs_clusters_dists!=0, inputs_clusters_dists, 10**(-100))
        # Shape of distance matrix:(n x d)
        u = inputs_clusters_dists**2 / np.sum(inputs_clusters_dists**2, axis=1)[:, np.newaxis]
        u = 1.0 / u
        u = (u.T / np.sum(u, axis=1)).T

        dists_grandmean = distance_matrix(
            np.mean(inputs, axis=0)[np.newaxis,:], clusters
        )
        return np.sum(u**2 * (inputs_clusters_dists**2 - dists_grandmean**2))
    return wrapper


def xie_beni(inputs: np.ndarray):
    n = inputs.shape[0]  # number of inputs
    m = inputs.shape[1]  # number of attributes

    def wrapper(particle):
        d = int(particle.shape[0]/m)  # number of clusters
        clusters = np.reshape(particle, (d, m))  # fit each cluster in a row
        intra_clusters_dists = distance.pdist(clusters)
        if 0 in intra_clusters_dists:
            return sys.maxsize  # If clusters are the same

        # Check if all clusters are equal
        inputs_clusters_dists = distance_matrix(inputs, clusters)
        inputs_clusters_dists = np.where(inputs_clusters_dists!=0, inputs_clusters_dists, 10**(-100))
        # Shape of distance matrix: (n x d)
        u = inputs_clusters_dists**2 / np.sum(inputs_clusters_dists**2, axis=1)[:, np.newaxis]
        u = 1.0 / u
        u = (u.T / np.sum(u, axis=1)).T
        num = np.sum(u**2 * inputs_clusters_dists**2)
        den = n * min(intra_clusters_dists)**2
        return num / den
    return wrapper


def square_sum(x):
    # -100 to 100
    return sum(np.square(x))


def rosenbrock(x):
    # -100 to 100
    x2 = np.array(x[1:])
    x1 = np.array(x[:-1])
    return sum(100*(x2-x1**2)**2 + (x1-1)**2)


def quartic_noise(x):
    # -1.28 to 1.28
    indices = np.array(range(len(x)))
    return sum(indices*x**4) + random.random()


def rastrigin(x):
    # -100 to 100
    return sum(np.square(x) - 10*np.cos(2*math.pi*x) + 10)


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


def schwefel_222(x):
    # -10 to 10
    absx = np.abs(x)
    return sum(absx) + np.prod(absx)


def schwefel_226(x):
    # -500 to 500
    absx = np.abs(x)
    const = 418.982887272433799807913601398
    return const*len(x) - sum(x*np.sin(np.sqrt(absx)))


def u(a, k, m, x_i):
    if   x_i >  a:
        return k * (x_i-a)**m
    elif x_i < -a:
        return k * (-x_i-a)**m
    else:
        return 0


def penalty_1(x):
    # -50 to 50
    k, m, a = 100, 4, 10
    n = len(x)
    PI = math.pi
    y = 1.25 + x/4
    term1 = 10*PI/n * math.sin(PI*y[0])**2
    term2 = sum((y[:-1]-1)**2 * (1+10*np.sin(PI*y[1:])**2))
    term3 =(y[-1]-1)**2
    term4 = sum([u(a, k, m, x_i) for x_i in x])
    return term1 + term2 + term3 + term4


def penalty_2(x):
    # -50 to 50
    k, m, a = 100, 4, 5
    PI = math.pi
    term1 = 0.1 * math.sin(3*PI*x[0])**2
    term2 = sum((x-1)**2 * (1 + np.sin(3*PI*x+1)**2))
    term3 =(x[-1]-1)**2 * (1 + math.sin(2*PI*x[-1]))**2
    term4 = sum([u(a, k, m, x_i) for x_i in x])
    return term1 + term2 + term3 + term4


def get_fitness_function(function_name: str, data=None):
    if data is not None:
        features, labels = data[:, :-1], data[:, -1]

        if function_name == 'silhouette':
            precomput_dists = distance_matrix(features, features)
            function = silhouette(features, precomput_dists)
            return FitnessFunction(function, maximization=True)

        elif function_name == 'xie_beni':
            function = xie_beni(features)
            return FitnessFunction(function, maximization=False)

        elif function_name == 'fuku_sugeno':
            function = fuku_sugeno(features)
            return FitnessFunction(function, maximization=False)

    elif function_name == 'griewank':
        return FitnessFunction(griewank, maximization=False)

    elif function_name == 'square_sum':
        return FitnessFunction(square_sum, maximization=False)

    elif function_name == 'quartic_noise':
        return FitnessFunction(quartic_noise, maximization=False)

    elif function_name == 'rosenbrock':
        return FitnessFunction(rosenbrock, maximization=False)

    elif function_name == 'rastrigin':
        return FitnessFunction(rastrigin, maximization=False)

    elif function_name == 'ackley':
        return FitnessFunction(ackley, maximization=False)

    elif function_name == 'schwefel_222':
        return FitnessFunction(schwefel_222, maximization=False)

    elif function_name == 'schwefel_226':
        return FitnessFunction(schwefel_226, maximization=False)

    elif function_name == 'penalty_1':
        return FitnessFunction(penalty_1, maximization=False)

    elif function_name == 'penalty_2':
        return FitnessFunction(penalty_2, maximization=False)
