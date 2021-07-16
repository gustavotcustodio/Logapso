import numpy as np
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


def get_fitness_function(function_name, data=None):
    if function_name == 'silhouette':
        inputs, labels = data[:, :-1], data[:, -1]
        precomput_dists = distance_matrix(inputs, inputs)
        function = silhouette(inputs, labels, precomput_dists)
        return FitnessFunction(function, maximization=True)
