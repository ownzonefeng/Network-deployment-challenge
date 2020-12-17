"""
Team name: ThE raNDom WALkERS
Members: Jonathan Doenz, Wentao Feng, Yuxuan Wang
"""

import numpy as np
import scipy as sp

from MH import objective

class BetaOptimizerCompetition():
    def __init__(self, dataset, lambda_, n_iterations, verbose=False):
        self.N = len(dataset.v)
        self.g = dataset
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.verbose = verbose

        # variables recording results
        self.f_vec = []

    def __str__(self):
        s = f"generator: {self.g}\n" +\
            f"lambda: {self.lambda_}\n" +\
            f"N: {self.N}\n" +\
            f"n_iterations: {self.n_iterations}"
        return s

    def run(self):
        for i in range(self.n_iterations):
            # Generate state composed by only 2 cities
            extreme_ids = np.random.choice(self.N, size=2)
            state = np.zeros_like(self.g.indices)
            state[extreme_ids] = 1.
            state = state.astype(bool)

            # Compute objective value for this state
            f_val = objective(self.g.v, self.g.x, self.lambda_, state)

            self.f_vec.append(f_val.copy())

class BetaOptimizer():
    def __init__(self, generator_class, lambda_, N, n_iterations, verbose=False):
        self.N = N
        # if 
        self.g = generator_class(N)
        self.lambda_ = lambda_
        self.n_iterations = n_iterations
        self.distances_matrix = sp.spatial.distance_matrix(self.g.x, self.g.x)
        self.verbose = verbose

        # variables recording results
        self.f_vec = []

    def __str__(self):
        s = f"generator: {self.g}\n" +\
            f"lambda: {self.lambda_}\n" +\
            f"N: {self.N}\n" +\
            f"n_iterations: {self.n_iterations}"
        return s

    def compute_max_distance(self, included_ids):
        max_distance = self.distances_matrix[np.ix_(included_ids, included_ids)].max()
        return max_distance

    def compute_f(self, max_distance, population):
        circular_area = np.pi * (max_distance / 2)**2
        return self.lambda_ * self.N * circular_area - population

    def run(self):
        for i in range(self.n_iterations):
            # Generate circle with random radius and circle in the [0, 1]^2 square
            extreme_ids = np.random.choice(self.N, size=2)
            max_distance = np.linalg.norm(self.g.x[extreme_ids[0]]
                                          - self.g.x[extreme_ids[1]])
            radius = max_distance / 2
            center = self.g.x[extreme_ids, :].mean(axis=0)

            # identify nodes inside the circle
            included_ids = np.where(sp.spatial.distance.cdist(center.reshape(1, -1), self.g.x) < radius)[1]
            population = self.g.v[included_ids].sum()
            f_val = self.compute_f(max_distance, population)

            self.f_vec.append(f_val.copy())
