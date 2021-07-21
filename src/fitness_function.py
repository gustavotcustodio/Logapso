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
        # 10**(-100) avoids division by 0
        # distances = np.where(distances != 0, distances, 10**(-100))
        # Shape of distance matrix:(n x d)
        u = distances**2 / np.sum(distances**2, axis=1)[:, np.newaxis]
        u = 1.0 / u
        u = (u.T / np.sum(u, axis=1)).T
        num = np.sum(u**2 * distances**2)
        den = n * min(distance.pdist(clusters))**2
        return num / den
    return wrapper


def get_fitness_function(function_name, features=None, labels=None):
    if function_name == 'silhouette':
        precomput_dists = distance_matrix(features, features)
        function = silhouette(features, precomput_dists)
        return FitnessFunction(function, maximization=True)
    else:
        function = xie_beni(features, labels)
        return FitnessFunction(function, maximization=False)
