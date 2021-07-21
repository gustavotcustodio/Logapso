import os
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score

dataset = 'iris.data'

path = os.path.join('datasets', dataset)
data = np.genfromtxt(path, delimiter=',')
np.random.shuffle(data)

inputs = data[:, :-1]
labels = data[:, -1]

precomput_dists = distance_matrix(inputs, inputs)
value = silhouette_score(inputs, labels)
print(value)
