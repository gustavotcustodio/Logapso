import os
import sys
import yaml
import numpy as np


def read_param_file(filename: str):
    try:
        with open(os.path.join('paramsfiles', filename),
                  'r') as stream:
            params = yaml.load(stream, Loader=yaml.FullLoader)
            return params
    except Exception as exc:
        print('Error reading file:', exc)
        sys.exit(2)


def scale_dataset(data: np.ndarray, scaled_min=0, scaled_max=1):
    max_columns = np.max(data, axis=0)
    min_columns = np.min(data, axis=0)
    return (scaled_max - scaled_min) * (data - min_columns) / \
        (max_columns - min_columns) + scaled_min


def read_dataset(dataset: str):
    path = os.path.join('data', 'datasets', dataset)
    data = np.genfromtxt(path, delimiter=',')
    np.random.shuffle(data)

    labels = data[:, -1:]
    data = scale_dataset(data[:, :-1], -1, 1)
    return np.hstack((data, labels))
