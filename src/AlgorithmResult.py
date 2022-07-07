import numpy as np

class AlgorithmResult:
    """docstring for AlgorithmResult"""
    def __init__(self, algorithm: str, fitness_values: list[list[float]]):
        self.algorithm = algorithm
        self.runs = fitness_values
        self.avg = np.average(np.asarray(fitness_values), axis=0)
        self.avg_best = self.avg[-1]
        self.std = np.std([fitness_run[-1] for fitness_run in fitness_values])
