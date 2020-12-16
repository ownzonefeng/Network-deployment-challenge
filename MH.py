"""
Team name: ThE raNDom WALkERS
Members: Jonathan Doenz, Wentao Feng, Yuxuan Wang
"""

import math, random
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix
from funcs import getOptBetaSeq, interpBetas
from dataset import G1, G2
from tqdm import trange

def minimize(func):
    def to_maximize(*args, **kwargs):
        value = func(*args, **kwargs)
        return -value
    return to_maximize


def maximize(func):
    def do_nothing(*args, **kwargs):
        value = func(*args, **kwargs)
        return value
    return do_nothing


class optimizer(object):
    """ 
    Metropolis-Hastings optimizer based on random walk (maximize objective function)
    """
    def __init__(self, obj_func, trans_func, beta, init_state, **kwargs):
        self.beta = beta
        self.obj_func = obj_func
        self.trans_func = trans_func
        self._kwargs = kwargs
        self.state = init_state
        self.obj_val = self.obj_func(state=init_state, **self._kwargs)

        self._init_state = init_state
    
    def reset(self):
        self.state = self._init_state
        self.obj_val = self.obj_func(state=self.state, **self._kwargs)
    
    def step(self):
        i = self.state
        j = self.trans_func(i)
        obj_j = self.obj_func(state=j, **self._kwargs)
        obj_i = self.obj_func(state=i, **self._kwargs)
        acceptance = self.accpet(obj_j, obj_i)
        if random.random() <= acceptance:
            self.state = j
            self.obj_val = obj_j
        return self.state, self.obj_val

    def accpet(self, j, i):
        if self.beta * (j - i) >= 0:
            return 1
        else:
            return math.exp(self.beta * (j - i))
    
    def run(self, iters=100, beta_schedule=None, reset=True):
        if reset:
            self.reset()
        vals = []
        states = []
        for i in trange(iters):
            if beta_schedule is not None:
                interval = iters // len(beta_schedule)
                beta_i = i // interval
                self.beta = beta_schedule[beta_i]
            curr_state, curr_val = self.step()
            vals.append(curr_val)
            states.append(curr_state)
        return vals, states

@maximize
def objective(v, x, lam, state):
    indicator = state
    n = len(indicator)

    def diameter(array):
        if len(array) <= 1:
            return 0
        if len(array) <= 128:
            return np.max(distance_matrix(array, array))
        hull = ConvexHull(array, incremental=False)
        edges = array[hull.vertices]
        diam = np.max(distance_matrix(edges, edges))
        return diam
    
    population = np.sum(v * indicator)
    cities = x[indicator]
    radius = diameter(cities) / 2
    area = math.pi * radius ** 2
    
    obj = population - lam * n * area
    return obj

def transition(state):
    ind = random.randint(0, len(state) - 1)
    new_state = np.copy(state)
    new_state[ind] = not state[ind]
    return new_state

def demo_run(gname: str = 'G1', lambda_: float = 1):
    num_iters = 2000
    datasets = {'G1': G1(100), 'G2': G2(100)}
    G = datasets[gname]

    beta_sequence = getOptBetaSeq(gname, lambda_)
    beta_interp = interpBetas(beta_sequence, num_iters, smooth=False)
    beta = beta_interp[0]

    inputs = {'v': G.v, 'x': G.x, 'lam': lambda_}
    init_state = np.random.randint(0, 2, size=len(G.v), dtype=bool)
    optim = optimizer(objective, transition, beta, init_state, **inputs)

    val, num_cities = optim.run(iters=num_iters, beta_schedule=beta_interp, reset=True)
    max_ind = np.argmax(val)
    chosen_cities = np.where(num_cities[max_ind] == 1)
    return chosen_cities