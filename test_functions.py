import random
import numpy as np
import os
from src import functions


def test_fuku_sugeno():
    path = os.path.join('data', 'datasets', 'iris.data')
    data = np.genfromtxt(path, delimiter=',')[:,:-1]
    centroids = np.array([
        6.336480,    2.905626,     5.013636,    1.727721,
        5.023318,    3.380671,     1.571838,    0.290483,
    ])
    print("2 clusters:", functions.fuku_sugeno(data)(centroids))

    centroids = np.array([
        6.775011, 3.052382, 5.646782, 2.0535467,
        5.003966, 3.414089, 1.482816, 0.2535463,
        5.888932, 2.761069, 4.363952, 1.3973150,
    ])
    print("3 clusters:", functions.fuku_sugeno(data)(centroids))

    centroids = np.array([
     6.254564,    2.885530,     4.909424,   1.6927103,
     5.000653,    3.418748,     1.469936,   0.2474031,
     6.999453,    3.103582,     5.890137,   2.1185975,
     5.637767,    2.655592,     4.024186,   1.2417211,
    ])
    print("4 clusters:", functions.fuku_sugeno(data)(centroids))

    centroids = np.array([
      4.998878,    3.418086,     1.467942,   0.2461613,
      7.437371,    3.079715,     6.277322,   2.0528023,
      6.190780,    2.877973,     4.709953,   1.5571730,
      6.525154,    3.037419,     5.447162,   2.0841764,
      5.585001,    2.616656,     3.949176,   1.2121261,
    ])
    print("5 clusters:", functions.fuku_sugeno(data)(centroids))

if __name__ == "__main__":
    test_fuku_sugeno()
